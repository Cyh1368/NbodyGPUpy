import datetime, os, sys, pickle, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree
import numpy as np
import cupy as cp
import requests

def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=text')
        response.raise_for_status()  # Check for HTTP errors
        return response.text
    except requests.RequestException as e:
        print(f"Error when retrieving IP: {e}")
        return -1

def send_status_update(message):
    try:
        url = "http://ec2-13-250-97-80.ap-southeast-1.compute.amazonaws.com:5000/status"
        headers = {"Content-Type": "application/json"}
        data = {
            "IP":get_public_ip(),
            "message": message,
            "timestamp": time.ctime()
        }
        response = requests.post(url, json=data, headers=headers)
        print(f"Status sent: {response.status_code}")
    except Exception as e:
        print(f"Error when sending status update: {e}")

def seconds_to_hms(execution_time):
    """
    Convert execution time in seconds to hours, minutes, and seconds format (H:M:S).

    Parameters:
    execution_time (float): Time in seconds.

    Returns:
    str: Time in "H:M:S" format.
    """
    ms = str(round(execution_time - np.floor(execution_time), 3))[2:]
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    
    return f"{hours}:{minutes:02}:{seconds:02}.{ms}"

def retrieve_arrays(folder_path, N, estimated_Nt_3D, estimated_Nt_1D):
    # Define the shapes and data types of the arrays you want to read
    # If OSError: [WinError 8] 記憶體資源不足，無法處理此命令。
    # means values of N or/and Nt is wrong
    save_shape = (N, 3, estimated_Nt_3D + 1)

    print(os.path.getsize(f'{folder_path}/pos.npy'))
    print(os.path.getsize(f'{folder_path}/vel.npy'))
    print(os.path.getsize(f'{folder_path}/KE.npy'))
    print(os.path.getsize(f'{folder_path}/PE.npy'))
    print(os.path.getsize(f'{folder_path}/t_3D.npy'))
    print(os.path.getsize(f'{folder_path}/t_1D.npy'))
    print(os.path.getsize(f'{folder_path}/mass.npy'))

    # Open the memory-mapped files in read mode
    pos_save = np.memmap(f'{folder_path}/pos.npy', dtype='float64', mode='r', shape=save_shape)
    vel_save = np.memmap(f'{folder_path}/vel.npy', dtype='float64', mode='r', shape=save_shape)
    KE_save = np.memmap(f'{folder_path}/KE.npy', dtype='float64', mode='r', shape=(estimated_Nt_1D + 1,))
    PE_save = np.memmap(f'{folder_path}/PE.npy', dtype='float64', mode='r', shape=(estimated_Nt_1D + 1,))
    mass_save = np.memmap(f'{folder_path}/mass.npy', dtype='float64', mode='r', shape=(N, ))
    t_3D_save = np.memmap(f'{folder_path}/t_3D.npy', dtype='float64', mode='r', shape=(estimated_Nt_3D + 1))
    t_1D_save = np.memmap(f'{folder_path}/t_1D.npy', dtype='float64', mode='r', shape=(estimated_Nt_1D + 1))

    return pos_save, vel_save, KE_save, PE_save, t_3D_save, t_1D_save, mass_save

def retrieve_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def find_closest_index(t_save, t):
    idx = np.searchsorted(t_save, t)
    if idx == len(t_save):
        closest_idx = len(t_save) - 1
    elif idx == 0:
        closest_idx = 0
    else:
        if abs(t_save[idx] - t) < abs(t_save[idx - 1] - t):
            closest_idx = idx
        else:
            closest_idx = idx - 1
    return closest_idx

def store_array_to_memmap(array, path):
    """
    Store a numpy array into a memory-mapped file.

    Parameters:
    - array: The numpy array to be stored.
    - path: The path where the memory-mapped file will be saved.
    """
    # Determine the shape and data type of the array
    shape = array.shape
    dtype = array.dtype

    print(f"Storing {dtype} array with shape {shape}.")

    # Create a memory-mapped file to store the array
    mmapped_array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)

    # Copy data to the memory-mapped file
    np.copyto(mmapped_array, array)

    # Flush changes to disk
    mmapped_array.flush()

def getAcc(pos, mass, G, softening=0):

    # print(f"Getacc processes = {num_processes}")
    N = pos.shape[0]
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    ax = cp.zeros((N, 1))
    ay = cp.zeros((N, 1))
    az = cp.zeros((N, 1))

    dx = x - x.T
    dy = y - y.T
    dz = z - z.T

    inv_r3 = (dx**2 + dy**2 + dz**2 + softening*82)

    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    ax = - G * (dx * inv_r3) @ mass
    ay = - G * (dy * inv_r3) @ mass
    az = - G * (dz * inv_r3) @ mass

    a = cp.hstack((ax, ay, az))
    return a

def getEnergy(pos, vel, mass, G=1):

    # Kinetic Energy:
	KE = 0.5 * cp.sum(cp.sum( mass * vel**2 ))

	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r for all particle pairwise particle separations 
	inv_r = cp.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	# sum over upper triangle, to count each interaction only once
	PE = G * cp.sum(cp.sum(cp.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE

def get_formatted_time():
     # Get the current time
    current_time = datetime.datetime.now()
    # Format the current time into a human-readable string
    formatted_time = current_time.strftime("%H%M%S-%Y%m%d")
    return formatted_time

def find_pairwise_min_timestep_KDtree(positions, velocities, scaleFactor=0.1):
    N = positions.shape[0]
    positions = positions.get()
    velocities = velocities.get()
    center_of_mass = np.mean(positions, axis=0)
    distances_from_com = np.linalg.norm(positions - center_of_mass, axis=1)

    fractional_N = N
    closest_indices = np.argsort(distances_from_com)[:fractional_N]
    closest_pos = positions[closest_indices]
    closest_vel = velocities[closest_indices]

    closest_pos_np = closest_pos

    kdtree = KDTree(closest_pos_np)
    
    # Query all points at once, k=2 to skip the point itself
    distances, indices = kdtree.query(closest_pos_np, k=2)
    
    # Take the second distance (closest neighbor) for each point
    min_distances = np.array(distances[:, 1])
    
    # Calculate the minimum timestep across all points
    min_timesteps = min_distances / np.linalg.norm(closest_vel, axis=1)
    
    return np.min(min_timesteps) * scaleFactor

def main_cont(memmap_name, metadata_name, min_dt, tStart, tEnd, store_pickle, store_interval = 1, scaleFactor = 0.1, softening=0):
    print("Utilizing multithreading. Continuing from a previous simulation. Softening=2e-4.")
    print(f"Retrieving frm memmap: {memmap_name}")
    # Here dt refers to the initial dt each frame
    # since it may be divided to smaller time steps if needed
    # output_interval: Every [how many] frames log one line of status update
    # Assuming this function is ran in /Memmaps
    current_directory = os.getcwd()
    """ N-body simulation """
    
    metadata_prev = retrieve_metadata(metadata_path = os.path.join(current_directory, f"{memmap_name}/{metadata_name}"))

    estimated_Nt_1D_prev = int(metadata_prev["estimated_Nt_1D"])
    estimated_Nt_3D_prev = int(metadata_prev["estimated_Nt_3D"])

    # Simulation parameters
    N = metadata_prev["N"]
    omega = metadata_prev["omega"]
    G = 1
    M = 1 # Total mass of the cluster

    pos_save_prev, vel_save_prev, KE_save_prev, PE_save_prev, t_3D_save_prev, t_1D_save_prev, mass_save_prev = retrieve_arrays(folder_path = os.path.join(current_directory, f"{memmap_name}"), N=N, estimated_Nt_3D=estimated_Nt_3D_prev, estimated_Nt_1D=estimated_Nt_1D_prev)

    # Trim trailing zeros
    KE_save_prev = np.trim_zeros(np.array(KE_save_prev), 'b')
    PE_save_prev = np.trim_zeros(np.array(PE_save_prev), 'b')
    t_3D_save_prev = np.trim_zeros(np.array(t_3D_save_prev), 'b')
    t_1D_save_prev = np.trim_zeros(np.array(t_1D_save_prev), 'b')

    tStart_index = find_closest_index(t_3D_save_prev, tStart)
    tStart = t_3D_save_prev[tStart_index]

    # Simulation parameters
    t = tStart      # current time of the simulation
    mass = mass_save_prev
    
    folder_path = f"Memmap_CONT_omega={omega}_{get_formatted_time()}"
    os.makedirs(folder_path, exist_ok=True)

    print(f"MAIN(): omega={omega}.")

    print("===== Simulation Data =====")
    print(f"Continuing from a previous simulation. N={N}, omega={omega}, dt=irregular, G={G}, M={M}, min_dt={min_dt}, tEnd={tEnd}")

    pos, vel = pos_save_prev[:, :, tStart_index].copy(), vel_save_prev[:, :, tStart_index].copy()
    pos = cp.asarray(pos)
    vel = cp.asarray(vel)
    mass = cp.asarray(mass.reshape(N, 1))

    print(f"Pos and Vel arrays loaded from previous simulation with shape {pos.shape}, {vel.shape} with types {type(pos)} and {type(vel)}. Mass shape: {mass.shape}")
    # calculate initial gravitational accelerations
    
    print(type(pos), type(vel), type(mass), type(pos_save_prev))
    acc = getAcc( pos, mass, G, softening)
    print("\nInitial acceleration obtained.")
    # calculate initial energy of system
    KE, PE  = getEnergy( pos, vel, mass, G)
    print("\nInitial KE, PE obtained.")
    # save energies, particle orbits for plotting trails
    # Create memory-mapped arrays for storage

    estimated_Nt_3D = (tEnd - t) // min_dt // store_interval # Max frames allowed, upper bound
    estimated_Nt_1D = (tEnd - t) // min_dt # Max frames allowed, upper bound

    print(f"\nEstimated Frames Upper Bound: 1D={estimated_Nt_1D}, 3D={estimated_Nt_3D}")

    pos_save = np.memmap(f'{folder_path}/pos.npy', dtype='float64', mode='w+', shape=(N, 3, int(estimated_Nt_3D + 1)))
    print(f"pos array created with size {os.path.getsize(f'{folder_path}/pos.npy') / 1024 / 1024 : .4f} MB")
    vel_save = np.memmap(f'{folder_path}/vel.npy', dtype='float64', mode='w+', shape=(N, 3, int(estimated_Nt_3D + 1)))
    print("vel array created")
    KE_save = np.memmap(f'{folder_path}/KE.npy', dtype='float64', mode='w+', shape=(int(estimated_Nt_1D + 1),))
    print(f"KE array created with size {os.path.getsize(f'{folder_path}/KE.npy') / 1024 / 1024: .4f} MB")
    PE_save = np.memmap(f'{folder_path}/PE.npy', dtype='float64', mode='w+', shape=(int(estimated_Nt_1D + 1),))
    print("PE array created")
    E_save = np.memmap(f'{folder_path}/E.npy', dtype='float64', mode='w+', shape=(int(estimated_Nt_1D + 1),))
    print("E array created")
    t_3D_save = np.memmap(f'{folder_path}/t_3D.npy', dtype='float64', mode='w+', shape=(int(estimated_Nt_3D + 1),))
    print("t_3D array created")
    t_1D_save = np.memmap(f'{folder_path}/t_1D.npy', dtype='float64', mode='w+', shape=(int(estimated_Nt_1D + 1),))
    print("t_1D array created")
    mass_save = np.memmap(f'{folder_path}/mass.npy', dtype='float64', mode='w+', shape=(N,))
    print("mass array created")

    i_save_3D = 0 # (SAVED) Frame count for the 3D arrays
    i_every = 0

    pos_save[:,:,0] = pos.get()
    vel_save[:,:,0] = vel.get()
    KE_save[0] = KE
    PE_save[0] = PE
    E_save[0] = PE+KE
    t_1D_save[0] = t
    t_3D_save[0] = t
    mass_save[:] = mass.get().reshape(N,)

    if store_pickle:
        # Store metadata or small arrays as needed, not large memmap files

        # Example:
        metadata = {
            "seed": f"Continuing from a previous simulation {memmap_name}. Seed = {metadata_prev['seed']}",
            "omega": omega,
            "N": N,
            "real_total_mass": metadata_prev['real_total_mass'],
            "dt": "irregular",
            "min_dt": min_dt,
            "record_dt": metadata_prev["record_dt"],
            "store_interval": store_interval,
            "scaleFactor": scaleFactor,
            "softening": softening,
            "tEnd": tEnd,
            "timestamp": get_formatted_time(),
            "workers": f"Using GPU, no parallization. IP: {get_public_ip()}",
            "IMF": metadata_prev['IMF'],
            "Nt_3D": i_save_3D, 
            "Nt_1D": i_every,
            "estimated_Nt_3D": estimated_Nt_3D,
            "estimated_Nt_1D": estimated_Nt_1D
        }

        # Serialize and save metadata
        metadata_filename = f'metadata.pkl'
        metadata_file_path = os.path.join(folder_path, metadata_filename)
        print(f"Memmap stored at {folder_path}")
        with open(metadata_file_path, 'wb') as f:
            pickle.dump(metadata, f)

        print("Sending first status email...")
        print(f"Metadata: {metadata}")
        send_status_update(f"Simulation Initialized for {get_public_ip()}, \nMetadata: {metadata}")

    # try:

    print(get_formatted_time())
    print("\n\nTime\t\t\t\tSince last (s)\t\tSaved Frame\tt (Henon)\t\tKE\t\tPE\t\tE")
    frame_time = time.time()

    while t<tEnd:
        sys.stdout.flush()
        current_time = datetime.datetime.now()
        print(f"{current_time}\t\t{time.time() - frame_time : .3f}\t\t{i_save_3D}\t\t{t:.5f} / {tEnd}\t\t{KE:.5f}\t\t{PE:.5f}\t\t{KE+PE:.5f}", end="\n")
        frame_time = time.time()

        irregular_dt = find_pairwise_min_timestep_KDtree(pos, vel, scaleFactor)
        # (1/2) kick
        vel += acc * irregular_dt/2.0
        # drift
        pos += vel * irregular_dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)
        # (1/2) kick
        vel += acc * irregular_dt/2.0
        # update time
        t += irregular_dt
        # get energy of system
        KE, PE  = getEnergy(pos, vel, mass, G)

        KE_save[i_every+1] = KE
        PE_save[i_every+1] = PE
        E_save[i_every+1] = PE+KE
        t_1D_save[i_every+1] = t
        
        if not i_every % store_interval: # Every some number of frames
            # save energies, positions for plotting trail
            # print(pos)
            pos_save[:, :, i_save_3D+1] = pos.get()
            vel_save[:, :, i_save_3D+1] = vel.get()
            t_3D_save[i_save_3D+1] = t
            i_save_3D+=1
            sys.stdout.flush() 

        if not i_every % (100 * store_interval):

            # Reset the device's memory pool (optional)
            print("Resetting GPU memory...")
            # Synchronize to ensure all computations are done
            cp.cuda.Device().synchronize()

            # Free all unused memory blocks
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print("GPU memory reest.")

            print(f"Sending status update at {i_every // store_interval} store intervals...")
            send_status_update(f"Status update for {get_public_ip()} at {i_every // store_interval} store intervals.\nSimulation is running smoothly. \n\nTime\t\t\t\tSince last (s)\t\tSaved Frame\tt (Henon)\t\tKE\t\tPE\t\tE\n{current_time}\t\t{time.time() - frame_time : .3f}\t\t{i_save_3D}\t\t{t:.5f} / {tEnd}\t\t{KE:.5f}\t\t{PE:.5f}\t\t{KE+PE:.5f}\nMetadata: {metadata}")
            
        i_every+=1

    formatted_time = get_formatted_time()
    print(f"{formatted_time}")

    # except Exception as e:
    #     print(f"Error caught: {e}")
    #     send_status_update(f"Error caught for {get_public_ip()}\n{e}")
                
    # except KeyboardInterrupt:
    #     print("\nSimulation interrupted by user.")      
    #     send_status_update(f"KeyboardInterrupt caught for {get_public_ip()}\n{e}")      

    print("Storing Metadata...")

    if store_pickle:
        # Store metadata or small arrays as needed, not large memmap files

        # Example:
        metadata = {
            "seed": f"Continuing from a previous simulation {memmap_name}. Seed = {metadata_prev['seed']}",
            "omega": omega,
            "N": N,
            "dt": "irregular",
            "min_dt": min_dt,
            "record_dt": "every frame",
            "scaleFactor": scaleFactor,
            "tStart": tStart,
            "tEnd": tEnd,
            "timestamp": get_formatted_time(),
            "workers": "CPU",
            "IMF": metadata_prev["IMF"],
            "Nt_3D": i_save_3D, # Actual
            "Nt_1D": i_every,
            "softening": softening,
            "estimated_Nt_3D": estimated_Nt_3D,
            "estimated_Nt_1D": estimated_Nt_1D
        }

        # Serialize and save metadata
        metadata_filename = f'metadata.pkl'
        metadata_file_path = os.path.join(folder_path, metadata_filename)
        print(f"Memmap stored at {folder_path}")
        with open(metadata_file_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Metadata (finale) and Memmap stored successfully at {folder_path}")

        print(f"Sending simulation complete signal...")
        send_status_update(f"Simulation Completed for {get_public_ip()}, \nMetadata: {metadata}")

    sys.stdout.flush() 

    return 0

if __name__ == "__main__":

    current_directory = os.getcwd()

    start_time = 0
    end_time = 0

    # Redirect stdout to the file
    console_stdout = sys.stdout
    print(console_stdout)

    log_file = f"{current_directory}/log/{get_formatted_time()}-output.txt"

    print(f"Log stored at {log_file}\nRunning...")

    with open(log_file, 'w') as file:

        sys.stdout = file

        parser = argparse.ArgumentParser(description='Example script.')
        parser.add_argument('--memmapname', type=str, help='Memmap Name of previous simulation', default="")
        parser.add_argument('--metadataname', type=str, help='Metadata Name.', default="metadata.pkl")
        parser.add_argument('--tstart', type=float, help='Simulation end time', default=0)
        parser.add_argument('--tend', type=float, help='Simulation end time', default=0)
        parser.add_argument('--storeinterval', type=int, help='Store interval of 3D arrays', default=40)
        parser.add_argument('--scalefactor', type=float, help='Time step scale factor.', default=0.4)
        parser.add_argument('--softening', type=float, help='Softening value.', default=0)


        args = parser.parse_args()
        
        start_time = time.time()

        main_cont(memmap_name=args.memmapname, metadata_name=args.metadataname, min_dt=5e-5, tStart=args.tstart, tEnd=args.tend, store_pickle=True, store_interval=args.storeinterval, scaleFactor=args.scalefactor, softening=args.softening)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")

    sys.stdout = console_stdout
    print("Execution completed.")
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")