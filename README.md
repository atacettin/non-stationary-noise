# Power Spectra of Non-Stationary Noises
This is the repository containing all code and date used for the MSc thesis 'Power Spectra of Non-Stationary Noises'. The repository includes simulation code for generating trajectories of the conditioned Ornstein-Uhlenbeck (OU) and two-level fluctuator (TLF) ensemble processes. 

## Code
The code is organised as follows:

```notebooks``` contains all Jupyter notebooks used in the generation of figures. ```figs.ipynb``` contains explanatory figures, ```ou.ipynb``` contains figures for the Ornstein-Uhlenbeck process (stationary and conditioned), and ```tlf_bath.ipynb``` contains figures for the TLF ensemble process (stationary and conditioned). 

```scripts``` contains files required to generate trajectories of the processes and save them to a desired output

```src/nonstationarynoise``` contains the source code used in the generation of the conditioned TLF ensemble and OU process.

## Data
The data is organised as follows:

The parent directory indicates the type of process (``ou`` is the OU process, ``stat_tlf_bath`` is the stationary TLF ensemble, etc.). The next directory indicates the number of samples in the trajectory (```N_8192``` indicates trajectories have 8192 timesteps). Then, the final directory divides the processes by the observation duration $T$, indicating the timesteps $\delta T$ used for each process. Each file is named after the level of conditioning (```1_sigma.npz``` for $1\sigma$ conditioning).

The data is stored in the Numpy ```.npz``` format and contains two fields, ```signal``` and ```extra_args```. The former is the samples of the noise, the second is the parameters of the process, following their definition in the source code.
