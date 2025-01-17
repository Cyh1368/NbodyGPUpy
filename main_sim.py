import datetime, os, sys, pickle, time, argparse
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
    mmapped_array = cp.memmap(path, dtype=dtype, mode='w+', shape=shape)

    # Copy data to the memory-mapped file
    cp.copyto(mmapped_array, array)

    # Flush changes to disk
    mmapped_array.flush()

def get_formatted_time():
     # Get the current time
    current_time = datetime.datetime.now()
    # Format the current time into a human-readable string
    formatted_time = current_time.strftime("%H%M%S-%Y%m%d")
    return formatted_time

def generate_IMF(imf_type):
    
    if imf_type == "Kroupa":

        # 0.08~0.5: 0.076917
        # 0.5~1.0: 0.0075193
        # 1~10: 0.00415299
        # 1~100: 0.00436113

        X = np.random.uniform(0, 1)

        """
        mH	0.2			Integral of IMF	Ratio
        m0	0.5		mH~m0	0.129887557	0.677532047
        m1	1		m0~m1	0.039126354	0.204094676
        m2	100		m1~m2	0.022692972	0.118373276
            		Total	0.191706882	1

        mH	0.25			Integral of IMF	Ratio
        m0	0.5		mH~m0	0.133015882	0.590152082
        m1	1		m0~m1	0.058466542	0.259398732
        m2	100		m1~m2	0.033910125	0.150449186
        alpha1	1.8		Total	0.225392549	1



        """
        
        if X < 0.590152082:  # m0~m1: 0.08~0.5, Exponent: -1.8, Probability: 0.86824
            m0, m1 = 0.25, 0.5
            exponent = -1.8
        elif X < 0.590152082+0.259398732:  # m0~m1: 0.5~1.0, Exponent: -2.7, Probability: 0.084878
            m0, m1 = 0.5, 1
            exponent = -2.7
        else:  # m0~m1: 1~100, Exponent: -2.3, Probability: 0.046879
            m0, m1 = 1, 100
            exponent = -2.3

        norm_const = (m1**(exponent + 1) - m0**(exponent + 1)) / (exponent + 1)
        u = np.random.uniform(0, 1)
        N = ((u * norm_const * (exponent + 1)) + m0**(exponent + 1))**(1 / (exponent + 1))
        
        return N

    elif imf_type == "Even":
        return np.random.uniform(0.08, 10)
    
    elif imf_type == "Single":
        return 1.0

    else:
        raise AssertionError("Unsupported IMF type. Use 'Kroupa', 'Single' or 'Even'.")

def generate_q():
    while True:
        x_4 = np.random.rand()
        x_5 = np.random.rand()

        q = x_4
        
        if 0.1 * x_5 < q**2 * (1 - q**2)**3.5:
            break
    
    return q

def plummer_model(N, omega=0, M=1.0, r_0=1.0, sd=1):
    np.random.seed(sd)
    """
    Generate initial coordinates and velocities for N-body simulation using the Plummer model.

    Parameters:
    N (int): Number of stars
    M (float): Total mass of the system
    r_0 (float): Scale length

    Returns:
    positions (numpy.ndarray) & velocities: Arrays of shape (N, 3) containing initial positions of stars
    """
    # Generate initial positions
    positions = cp.zeros((N, 3))
    r = cp.zeros(N)

    reject_count = 0
    for i in range(N):
        while True:
            X_1 = np.random.rand()
            X_2 = np.random.rand()
            X_3 = np.random.rand()

            # print(X_1, X_2, X_3)

            r_i = r_0 * ((X_1 ** (-2/3)) - 1) ** (-1/2)

            if r_i <= 10:
                z = np.float64((1 - 2 * X_2) * r_i)
                x = np.float64((r_i**2 - z**2)**0.5 * cp.cos(2 * cp.pi * X_3))
                y = np.float64((r_i**2 - z**2)**0.5 * cp.sin(2 * cp.pi * X_3))

                positions[i] = cp.array([x, y, z])
                r[i] = r_i
                break
            else:
                reject_count+=1
                
    print("Rejections above 10: ", reject_count)
    # Calculate escape velocities
    v_e = 2**0.5 * (1 + r**2) ** (-0.25)

    q = cp.asarray(np.asarray([generate_q() for _ in range(N)]))
    
    v = v_e * q
    
    X_6 = cp.asarray(np.random.rand(N))
    X_7 = cp.asarray(np.random.rand(N))
    
    vz = (1 - 2 * X_6) * v
    vx = (v**2 - vz**2)**0.5 * cp.cos(2 * cp.pi * X_7) - omega * positions[:, 1]
    vy = (v**2 - vz**2)**0.5 * cp.sin(2 * cp.pi * X_7) + omega * positions[:, 0]

    velocities = cp.vstack((vx, vy, vz)).T
    
    assert(len(positions)==len(velocities))
    
    return positions, velocities

# Vectorizing for CUPY

def getAcc(pos, mass, G, softening=0):

    # print(f"Getacc processes = {num_processes}")
    print(mass.shape)
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

def scalePosVel(pos, vel, mass, q_vir = 0.5, E_0 = -0.25, loquacious=True):
    KE, PE  = getEnergy( pos, vel, mass, G=1)
    V = PE # Virial energy
    if loquacious: print(f"Before scaling KE={KE:.5f}, PE={PE:.5f}, E={KE+PE:.5f}")
    vel_scaled = vel * cp.sqrt( abs ( q_vir * V / KE ))

    beta = abs ( (1 - q_vir) * PE / E_0 )
    pos_scaled = pos * beta
    vel_scaled = vel_scaled / cp.sqrt(beta)
    KE_scaled, PE_scaled  = getEnergy( pos_scaled, vel_scaled, mass, G=1)
    if loquacious: print(f"After scaling KE={KE_scaled:.5f}, PE={PE_scaled:.5f}, E={KE_scaled+PE_scaled:.5f}")

    return pos_scaled, vel_scaled

def find_pairwise_min_timestep_KDtree(positions, velocities, masses, scaleFactor=0.1):
    N = positions.shape[0]
    positions = positions.get()
    velocities = velocities.get()
    masses = masses.get()

    center_of_mass = np.sum(positions.T * masses.T, axis=1) / np.sum(masses)

    distances_from_com = np.linalg.norm(positions - center_of_mass, axis=1)

    closest_indices = np.argsort(distances_from_com)[:N]
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

def main_irregular_step(seed, omega, N, min_dt, tEnd, store_pickle, imf_type, store_interval = 1, scaleFactor = 0.1, softening = 0):
    print(f"Utilizing CUPY.")
    folder_path = f"Memmap_IMF_omega={omega}_{get_formatted_time()}"
    os.makedirs(folder_path, exist_ok=True)
    print(f"MAIN(): omega={omega}.")

    # Simulation parameters
    t         = 0      # current time of the simulation
    G = 1
    M = 1 # Total mass of the cluster

    # Generate Initial Conditions
    np.random.seed(int(seed))            # set the random number generator seed
    
    mass = cp.array([generate_IMF(imf_type) for _ in range(N)]).reshape(N, 1)
    real_total_mass = float(sum(mass)[0])
    print(f"Total mass before normalization: {real_total_mass}")
    mass /= sum(mass)
    
    print("===== Simulation Data =====")

    pos, vel = plummer_model(N, omega, sd=seed)
    print(pos.shape)
    print("Plummer model applied to pos, vel arrays.")

    pos, vel = scalePosVel(pos, vel, mass, q_vir = 0.5, E_0 = -0.25, loquacious=True)

    print(pos, vel)

    print("Position and velocity scaled.")

    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening)
    print(f"\n\nMass shape: {mass.shape}")
    print("\nInitial acceleration obtained.")
    # calculate initial energy of system
    KE, PE  = getEnergy( pos, vel, mass, G)
    print("\nInitial KE, PE obtained.")

    assert( abs(KE-0.25) < 0.01)
    assert( abs(PE- (-0.50)) < 0.01)

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

    mass_save[:] = mass.get().reshape(N,)
    # mass_save = mass

    i_save_3D = 0 # (SAVED) Frame count for the 3D arrays
    i_every = 0

    pos_save[:,:,0] = pos.get()
    vel_save[:,:,0] = vel.get()
    KE_save[0] = KE
    PE_save[0] = PE
    E_save[0] = PE+KE
    t_1D_save[0] = t
    t_3D_save[0] = t

    print("Storing Metadata...")

    if store_pickle:
        # Store metadata or small arrays as needed, not large memmap files with excessive space wasted

        # Example:
        metadata = {
            "seed": seed,
            "omega": omega,
            "N": N,
            "real_total_mass": real_total_mass,
            "dt": "irregular",
            "min_dt": min_dt,
            "record_dt": "every frame",
            "store_interval": store_interval,
            "scaleFactor": scaleFactor,
            "softening": softening,
            "tEnd": tEnd,
            "timestamp": get_formatted_time(),
            "workers": f"Using GPU. IP: {get_public_ip()}",
            "IMF": imf_type,
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

        # print("Sending first status email...")
        print(f"Simulation Initialized for {get_public_ip()}\n")
        for index, (key, value) in enumerate(metadata.items()):
            print(f"{key}\t\t{value}")
        print()

    sys.stdout.flush() 

    print(f"Metadata (preliminary) and Memmap stored successfully at {folder_path}.")

    # try:

    print(get_formatted_time())
    print("\n\nTime\t\t\t\tSince last (s)\t\tSaved Frame\tt (Henon)\t\tKE\t\tPE\t\tE")
    frame_time = time.time()

    while t<tEnd:
        sys.stdout.flush()
        current_time = datetime.datetime.now()
        print(f"{current_time}\t\t{time.time() - frame_time : .3f}\t\t{i_save_3D}\t\t{t:.5f} / {tEnd}\t\t{KE:.5f}\t\t{PE:.5f}\t\t{KE+PE:.5f}", end="\n")
        frame_time = time.time()

        # Un-comment for debugging
        # start_time = time.time()
        
        irregular_dt = find_pairwise_min_timestep_KDtree(pos, vel, mass, scaleFactor)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Pairwise Time Step took {execution_time:.10f} seconds")

        # (1/2) kick

        vel += acc * irregular_dt/2.0

        # drift

        pos += vel * irregular_dt

        # update accelerations

        # start_time = time.time()

        acc = getAcc(pos, mass, G, softening)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"GetAcc took {execution_time:.10f} seconds")

        # (1/2) kick

        vel += acc * irregular_dt/2.0

        # update time

        t += irregular_dt

        # get energy of system

        # start_time = time.time()
        
        KE, PE  = getEnergy(pos, vel, mass, G)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"GetEnergy took {execution_time:.10f} seconds")

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

            # print(f"Sending status update at {i_every // store_interval} store intervals...")
            # send_status_update(f"Status update for {get_public_ip()} at {i_every // store_interval} store intervals.\nSimulation is running smoothly. \n\nTime\t\t\t\tSince last (s)\t\tSaved Frame\tt (Henon)\t\tKE\t\tPE\t\tE\n{current_time}\t\t{time.time() - frame_time : .3f}\t\t{i_save_3D}\t\t{t:.5f} / {tEnd}\t\t{KE:.5f}\t\t{PE:.5f}\t\t{KE+PE:.5f}\nMetadata: {metadata}")
            
        i_every+=1

    formatted_time = get_formatted_time()
    print(f"Seed={seed}, {formatted_time}")

    # except Exception as e:
    #     print(f"Error caught: {e}")
    #     # send_status_update(f"Error caught for {get_public_ip()}\n{e}")
                
    # except KeyboardInterrupt:
    #     print("\nSimulation interrupted by user.")      
    #     # send_status_update(f"KeyboardInterrupt caught for {get_public_ip()}")      

    print("Storing Metadata...")

    if store_pickle:
        # Store metadata or small arrays as needed, not large memmap files

        # Example:
        metadata = {
            "seed": seed,
            "omega": omega,
            "N": N,
            "real_total_mass": real_total_mass,
            "dt": "irregular",
            "min_dt": min_dt,
            "record_dt": "every frame",
            "store_interval": store_interval,
            "scaleFactor": scaleFactor,
            "softening": softening,
            "tEnd": tEnd,
            "timestamp": get_formatted_time(),
            "workers": f"Using GPU, no parallization. IP: {get_public_ip()}",
            "IMF": "Three-Segment Power Law: 0.08~0.5 & 0.5~1.0 & 1.0~10.0",
            "Nt_3D": i_save_3D, 
            "Nt_1D": i_every,
            "estimated_Nt_3D": estimated_Nt_3D,
            "estimated_Nt_1D": estimated_Nt_1D
        }

        print(metadata)

        # Serialize and save metadata
        metadata_filename = f'metadata.pkl'
        metadata_file_path = os.path.join(folder_path, metadata_filename)
        print(f"Memmap stored at {folder_path}")
        with open(metadata_file_path, 'wb') as f:
            pickle.dump(metadata, f)

    sys.stdout.flush() 

    print(f"Metadata (finale) and Memmap stored successfully at {folder_path}")

    print(f"Sending simulation complete signal...")
    # send_status_update(f"Simulation Completed for {get_public_ip()}, \nMetadata: {metadata}")

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
        parser.add_argument('--omega', type=float, help='Float, the (rigid body) angular velocity', default=0)
        parser.add_argument('--seed', type=int, help='Seed', default=1)
        parser.add_argument('--N', type=int, help='Number of stars', default=1000)
        parser.add_argument('--tend', type=float, help='Simulation end time', default=0)
        parser.add_argument('--storeinterval', type=float, help='Store interval of 3D arrays', default=400)
        parser.add_argument('--scalefactor', type=float, help='Time step scale factor.', default=0.02)
        parser.add_argument('--softening', type=float, help='Softening value.', default=0)
        parser.add_argument('--imf', type=str, help='IMF type', default="")

        args = parser.parse_args()

        print(f"Arguments: {args.omega}, {args.tend}")
        
        start_time = time.time()

        main_irregular_step(seed=args.seed, omega=args.omega, N=args.N, min_dt=5e-5, tEnd=args.tend, imf_type = args.imf, store_pickle=True, store_interval=args.storeinterval, scaleFactor=args.scalefactor, softening = args.softening)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {seconds_to_hms(execution_time)}")

    sys.stdout = console_stdout
    print("Execution completed.")
    execution_time = end_time - start_time
    print(f"Execution time: {seconds_to_hms(execution_time)}")