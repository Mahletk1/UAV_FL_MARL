import numpy as np

def init_uav_positions(N, area_size=500):
    # fixed horizontal positions
    x = np.random.uniform(-area_size/2, area_size/2, N)
    y = np.random.uniform(-area_size/2, area_size/2, N)
    return x, y

def init_altitudes(N, h_min, h_max):
    return np.random.uniform(h_min, h_max, N)

def update_altitudes(h, actions, h_min, h_max):
    h_new = h + actions
    return np.clip(h_new, h_min, h_max)