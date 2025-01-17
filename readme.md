# Python GPU-accelerated NBODY Code

`main_sim.py` is a Python script designed to perform N-body simulations leveraging GPU acceleration with CuPy and advanced computational techniques. The script includes functionality for generating initial conditions, scaling positions and velocities, and performing dynamic simulations of astrophysical systems.

`cont_sim.py` complements `main_sim.py` by allowing users to continue a previously saved simulation, seamlessly resuming from stored data.

This code is inspired by the excellent open-source N-body simulation project by [Philipp Mocz](https://github.com/pmocz/nbody-python).

---

## Features

The N-body simulation framework provides a comprehensive set of features, with a focus on efficiency, scalability, and astrophysical accuracy. Below are the key features explained with relevant mathematical equations.

---

### 1. **Initial Condition Generation**
The script generates initial conditions for the N-body system using the **Plummer model** and various **Initial Mass Functions (IMFs)**.

#### Plummer Model
The positions and velocities of particles are generated based on the Plummer density profile:

$$
\rho(r) = \mathbf{a}_i\frac{3M}{4\pi r_0^3} \left( 1 + \frac{r^2}{r_0^2} \right)^{-5/2}
$$

Where:
- $M$ is the total mass,
- $r_0$ is the scale length,
- $r$ is the radial distance from the center.

The escape velocity $v_e$ for a particle at radial distance $r$ is computed as:

$$
v_e = \sqrt{2} \left(1 + r^2\right)^{-1/4}
$$

#### Initial Mass Functions (IMFs)
The masses of particles are sampled based on user-defined IMF types:
1. **Kroupa IMF**: A three-segment power-law distribution:

$$
P(m) \propto 
\begin{cases} 
m^{-1.8}, & 0.08 \leq m < 0.5 \\
m^{-2.7}, & 0.5 \leq m < 1.0 \\
m^{-2.3}, & 1.0 \leq m < 100 
\end{cases}
$$

2. **Single IMF**: All particles have the same mass, $m = 1.0$.
3. **Even IMF**: Masses are uniformly distributed between $m = 0.08$ and $m = 10.0$.

---

### 2. **GPU Acceleration**
The script uses the **CuPy** library to offload computationally intensive tasks to the GPU, accelerating matrix operations. For instance:

#### Acceleration Calculation
The gravitational acceleration \( \mathbf{a}_i \) for each particle is computed as:

$$
\mathbf{a}_i = -G \sum_{j \neq i} \frac{m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left( |\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2 \right)^{3/2}}
$$

Where:
- \( G \) is the gravitational constant,
- \( \mathbf{r}_i \) and \( \mathbf{r}_j \) are positions of particles \( i \) and \( j \),
- \( m_j \) is the mass of particle \( j \),
- \( \epsilon \) is a softening parameter to avoid singularities.

This computation is vectorized using CuPy, significantly reducing execution time.

---

### 3. **Irregular Time Steps**
The simulation dynamically adjusts time steps based on the closest pair of particles. The minimum time step \( \Delta t_{\text{min}} \) is calculated as:
$$
\Delta t_{\text{min}} = \frac{d_{\text{min}}}{v_{\text{max}}}
$$
Where:
- \( d_{\text{min}} \) is the minimum distance between two particles,
- \( v_{\text{max}} \) is the maximum relative velocity.

---

### 4. **Scalable Storage**
To handle large-scale simulations efficiently, the script uses **memory-mapped files** to store particle positions, velocities, and energies without loading all data into RAM. For example, the position data is saved as:
$$
\text{pos\_save}[i, :, t] = \mathbf{r}_i(t)
$$
Where \( \mathbf{r}_i(t) \) is the position of particle \( i \) at time \( t \).

---

### 5. **Energy Conservation**
The script calculates the kinetic energy (KE) and potential energy (PE) to monitor energy conservation:
$$
KE = \frac{1}{2} \sum_{i=1}^N m_i v_i^2
$$
$$
PE = -\frac{G}{2} \sum_{i \neq j} \frac{m_i m_j}{|\mathbf{r}_j - \mathbf{r}_i|}
$$
The total energy \( E \) of the system is:
$$
E = KE + PE
$$

---

### 6. **Status Reporting**
The script sends periodic updates about the simulation status to a REST API. This includes:
- Current simulation time,
- Kinetic and potential energy,
- Progress as a percentage of total runtime.

---

### 7. **Scalable Position and Velocity Scaling**
The positions \( \mathbf{r} \) and velocities \( \mathbf{v} \) are scaled to satisfy the virial theorem:
$$
\frac{2 KE}{|PE|} = q_{\text{vir}}
$$
Where \( q_{\text{vir}} \) is the virial ratio. Scaling is performed as:
$$
\mathbf{v}_{\text{scaled}} = \mathbf{v} \sqrt{\frac{|q_{\text{vir}} \cdot PE|}{KE}}
$$
$$
\mathbf{r}_{\text{scaled}} = \mathbf{r} \cdot \beta
$$
Where \( \beta \) ensures total energy matches the desired value.

---

## Continuing a Simulation with `cont_sim.py`

`cont_sim.py` enables users to resume a simulation from a previously saved state. It requires the memory-mapped files and metadata from the earlier run.

### Usage
```bash
python cont_sim.py --memmapname Memmap_IMF_omega=0.0_123456-20250117 --metadataname metadata.pkl --tstart 10 --tend 20 --storeinterval 40 --scalefactor 0.4 --softening 0.1
