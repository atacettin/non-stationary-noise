import numpy as np

def sample_ou(N: float, gamma: float, alpha: float, initial_pos: float, dt:float = 1) -> np.ndarray: 
    """
    Generates a sample trajectory of an Ornstein-Uhlenbeck process

    Parameters
    ----------
    N : float
        Length of the sample trajectory
    gamma : float
        Rate of decay parameter
    alpha : float
        Multiplicative factor of in front of the white noise
    init_pos : float
        Initial position of the process
    dt : float
        Sampling interval
        
    Returns
    -------
    trajectory : np.array of floats
        The sample trajectory
    """
    N = int(N)
    traj = [initial_pos]
    exponent = np.exp(-gamma*dt)
    std = np.sqrt(alpha**2/(2*gamma)*(1-np.exp(-2*gamma*dt)))
    for i in range(0, N-1):
        mean = traj[i]*exponent
        traj.append(np.random.normal(mean, std))

    return np.array(traj)

def sample_stationary_ou(N: float, gamma: float, alpha: float, dt:float = 1) -> np.ndarray:
    """
    Generates a sample trajectory of an Ornstein-Uhlenbeck process starting from an initial position sampled from the stationary distribution.

    Parameters
    ----------
    N : int
        Length of the sample trajectory
    gamma : float
        Rate of decay parameter
    alpha : float
        Multiplicative factor of in front of the white noise
    dt : float
        Sampling interval
        
    Returns
    -------
    trajectory : np.array of floats
        The sample trajectory
    """
    initial_pos = np.random.normal(0, alpha**2/(2*gamma))
    traj = sample_ou(N, gamma, alpha, initial_pos, dt)
    return traj


def legacy_sample_ou(N: int, gamma: float, alpha: float, initial_pos: float, dt:float = 1) -> np.ndarray:
    """
    Generates a sample trajectory of an Ornstein-Uhlenbeck process of length N using Euler-Maruyama

    Parameters
    ----------
    N : int
        Length of the sample trajectory
    gamma : float
        Rate of decay parameter
    alpha : float
        Multiplicative factor of in front of the white noise
    init_pos : float
        Initial position of the process
    dt : float
        Sampling interval
        
    Returns
    -------
    trajectory : np.array of floats
        The sample trajectory
    """
    gs = np.random.normal(loc=0, scale=np.sqrt(dt), size=N-1)
    ou = np.zeros(N)
    ou[0] = initial_pos

    a = (1-gamma*dt)
    b = alpha
    for i in range(1,N):
        ou[i] = a*ou[i-1] + b*gs[i-1]

    return np.array(ou)

def lorentzian(fs, gamma, alpha):
    return alpha**2 / (gamma**2 + (2*np.pi*fs)**2)

def finite_time_correction(fs, N, dt, gamma, alpha):
    T = N*dt
    os = 2*np.pi*fs

    G = gamma**2 + os**2 
    H = gamma**2 - os**2
    E = np.exp(-gamma*T)

    return - (alpha**2 / (gamma * T * G**2)) * (H - E * (H * np.cos(os * T) + 2 * gamma * os * np.sin(os * T)))

def nonstationary_correction(fs, N, dt, gamma, alpha, initial_pos):
    T = N*dt
    os = 2*np.pi*fs 

    G = gamma**2 + os**2 
    E = np.exp(-gamma*T)

    return (initial_pos**2 - alpha**2/(2*gamma))*(1/T)*1/G*(1-2*E*np.cos(os*T)+E**2)