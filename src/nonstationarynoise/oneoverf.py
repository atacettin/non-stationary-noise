import numpy as np
from scipy import integrate
from functools import partial


def sample_one_over_f(N: float, N_tlf: float, gamma_min: float, gamma_max: float, sigma: float, z: float, dt: float) -> np.ndarray:
    N = int(N)
    N_tlf = int(N_tlf)
    strength = sigma/np.sqrt(N_tlf)
    
    # Sample switching rates according to a log-uniform distribution
    gammas = np.exp(np.random.uniform(np.log(gamma_min), np.log(gamma_max), N_tlf))
    gammas = np.sort(gammas)

    # Initialise the states of the TLFs
    delta = np.log(gamma_max/gamma_min)
    gamma_c = gamma_min*np.exp(z*delta/np.sqrt(N_tlf))
    states = np.random.choice([-1, 1], size=N_tlf)
    states = np.where(gammas <= gamma_c, 1, states)

    # Precompute flips
    flips = np.random.rand(N, N_tlf) < 0.5*(1 - np.exp(-gammas*dt))

    # Generate the trajectory
    traj = np.zeros(N)
    traj[0] = np.sum(states)*strength
    for i in range(1, N):
        states = np.where(flips[i], -states, states)
        traj[i] = np.sum(states)*strength
    return traj

def sample_one_over_f_equal(N: float, N_tlf: float, gamma_min: float, gamma_max: float, sigma: float, z: float, dt: float) -> np.ndarray:
    N = int(N)
    N_tlf = int(N_tlf)
    strength = sigma/np.sqrt(N_tlf)
    
    # Sample switching rates according to a log-uniform distribution
    gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), N_tlf)

    # Initialise the states of the TLFs
    delta = np.log(gamma_max/gamma_min)
    gamma_c = gamma_min*np.exp(z*delta/np.sqrt(N_tlf))
    states = np.random.choice([-1, 1], size=N_tlf)
    states = np.where(gammas <= gamma_c, 1, states)

    # Precompute flips
    flips = np.random.rand(N, N_tlf) < 0.5*(1 - np.exp(-gammas*dt))

    # Generate the trajectory
    traj = np.zeros(N)
    traj[0] = np.sum(states)*strength
    for i in range(1, N):
        states = np.where(flips[i], -states, states)
        traj[i] = np.sum(states)*strength
    return traj


def _finite_time_integrand(u, T, omegas):
    gamma = np.exp(u)
    denom = gamma**2 + omegas**2
    return (1 - np.exp(-gamma * T)) * (-1 / (gamma * denom) + 2 * gamma / denom**2) * gamma

def finite_time_correction(fs, N, dt, gamma_min, gamma_max, N_tlf, sigma):
    delta = np.log(gamma_max / gamma_min)
    T = N * dt
    omegas = 2 * np.pi * fs

    u_min = np.log(gamma_min)
    u_max = np.log(gamma_max)
    
    result, _ = integrate.quad_vec(
        partial(_finite_time_integrand, T=T, omegas=omegas),
        u_min, u_max,
        epsabs=1e-10,
        epsrel=1e-8,
        workers=-1,
    )

    return -(sigma**2 / N_tlf) / (2 * delta * T) * result

def _I(gamma, omega, T):
    z = gamma - 1j * omega
    return (1 - np.exp(-z * T)) / z

def _term1_integrand(u, T, omegas):
    gamma = np.exp(u)  # u = log(gamma), Jacobian factor gamma included below
    I = _I(gamma, omegas, T)
    return (np.abs(I) ** 2 / gamma) * gamma

def _term2_integrand(u, T, omegas):
    gamma = np.exp(u)
    I = _I(gamma, omegas, T)
    return (I / gamma) * gamma 

def nonstationary_correction(fs, N, dt, gamma_min, gamma_max, gamma_c, N_tlf, sigma):
    delta = np.log(gamma_max / gamma_min)
    T = N * dt
    omegas = 2 * np.pi * fs

    u_min = np.log(gamma_min)
    u_max = np.log(gamma_c)

    # Term 1: integral of |I|^2 / gamma  (real)
    int1, _ = integrate.quad_vec(
        partial(_term1_integrand, T=T, omegas=omegas),
        u_min, u_max,
        epsabs=1e-10, epsrel=1e-8,
    )

    # Term 2: integral of I / gamma  (complex -> square the modulus)
    int2_re, _ = integrate.quad_vec(
        lambda u: _term2_integrand(u, T, omegas).real,
        u_min, u_max,
        epsabs=1e-10, epsrel=1e-8,
    )
    int2_im, _ = integrate.quad_vec(
        lambda u: _term2_integrand(u, T, omegas).imag,
        u_min, u_max,
        epsabs=1e-10, epsrel=1e-8,
    )
    int2_sq = int2_re**2 + int2_im**2  # |int2|^2

    return (1 / T) * (
        -(sigma**2 / delta) * int1
        + (N_tlf * sigma**2 / delta**2) * int2_sq
    )

def sample_one_over_f_same_gamma(N: float, N_tlf: float, gamma: float, sigma: float, z: float, dt: float) -> np.ndarray:
    N = int(N)
    N_tlf = int(N_tlf)
    strength = sigma/np.sqrt(N_tlf)

    states = np.random.choice([-1, 1], size=N_tlf)

    # Initialise a percentage of states to 1 based on z
    states[:int(z*np.sqrt(N_tlf))] = 1

    # Precompute flips
    flips = np.random.rand(N, N_tlf) < 0.5*(1 - np.exp(-gamma*dt))

    # Generate the trajectory
    traj = np.zeros(N)
    traj[0] = np.sum(states)*strength
    for i in range(1, N):
        states = np.where(flips[i], -states, states)
        traj[i] = np.sum(states)*strength
    return traj

def tlf_ensemble(fs, gamma_min, gamma_max, sigma, N_tlf):
    fs = np.asarray(fs, dtype=float)
    
    b_sq = sigma**2 / N_tlf
    delta = np.log(gamma_max / gamma_min)

    psd = np.zeros(len(fs))
    
    mask = fs != 0
    psd[~mask] = (2 * N_tlf * b_sq / delta) * (1/gamma_min - 1/gamma_max)
    psd[mask] = (2 * N_tlf * b_sq / delta) * (np.arctan(gamma_max / (2*np.pi*fs[mask])) - np.arctan(gamma_min / (2*np.pi*fs[mask]))) / (2*np.pi*fs[mask])
    
    return psd