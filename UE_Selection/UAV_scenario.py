import numpy as np

# def init_uav_positions(N, area_size=500):
#     # fixed horizontal positions
#     x = np.random.uniform(-area_size/2, area_size/2, N)
#     y = np.random.uniform(-area_size/2, area_size/2, N)
#     return x, y

def init_random_xy_trajectory(N, T, area_size=500.0, seed=0):
    rng = np.random.default_rng(seed)
    traj_x = rng.uniform(-area_size/2, area_size/2, size=(T, N))
    traj_y = rng.uniform(-area_size/2, area_size/2, size=(T, N))
    return traj_x, traj_y

def init_random_walk_xy_trajectory(N, T, area_size=500.0, step_std=10.0, seed=0):
    rng = np.random.default_rng(seed)

    traj_x = np.zeros((T, N))
    traj_y = np.zeros((T, N))

    traj_x[0] = rng.uniform(-area_size/2, area_size/2, size=N)
    traj_y[0] = rng.uniform(-area_size/2, area_size/2, size=N)

    for t in range(1, T):
        traj_x[t] = traj_x[t-1] + rng.normal(0, step_std, size=N)
        traj_y[t] = traj_y[t-1] + rng.normal(0, step_std, size=N)

        traj_x[t] = np.clip(traj_x[t], -area_size/2, area_size/2)
        traj_y[t] = np.clip(traj_y[t], -area_size/2, area_size/2)

    return traj_x, traj_y 
def init_circular_xy_trajectory(N, T, R_mean=200.0, R_jitter=40.0, seed=0): #260 meters coverage area
    rng = np.random.default_rng(seed)
    
    traj_x = np.zeros((T, N))
    traj_y = np.zeros((T, N))

    R0 = R_mean + rng.uniform(-R_jitter, R_jitter, size=N)
    drift = rng.uniform(-0.2, 0.2, size=N)

    for j in range(N):
        phi = 2 * np.pi * j / N
        for t in range(T):
            Rt = R0[j] + drift[j] * t
            traj_x[t, j] = Rt * np.cos(0.02 * t + phi)
            traj_y[t, j] = Rt * np.sin(0.02 * t + phi)

    return traj_x, traj_y

def init_predefined_height_trajectory(N, T, h_min, h_max, seed=0):
    rng = np.random.default_rng(seed)
    traj_h = np.zeros((T, N))
    phase = rng.uniform(0, 2*np.pi, size=N)

    for j in range(N):
        for t in range(T):
            traj_h[t, j] = h_min + (h_max - h_min) * (0.5 + 0.5 * np.sin(0.01 * t + phase[j]))

    return traj_h

def init_altitudes(N, h_min, h_max):
    return np.random.uniform(h_min, h_max, N)

def update_altitudes(h, actions, h_min, h_max):
    h_new = h + actions
    return np.clip(h_new, h_min, h_max)