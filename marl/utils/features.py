# marl/utils/features.py
import numpy as np

def normalize_angle_deg(theta_deg: np.ndarray) -> np.ndarray:
    # theta in [0, 90] roughly, normalize to [0,1]
    return np.clip(theta_deg / 90.0, 0.0, 1.0)

def normalize_db(x_db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # Simple clip+scale to [0,1]
    x = np.clip(x_db, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

def normalize_pos(x: np.ndarray, scale: float = 500.0) -> np.ndarray:
    # scale meters to roughly [-1,1]
    return np.clip(x / scale, -1.0, 1.0)

def clip_altitude(h: np.ndarray, h_min: float, h_max: float) -> np.ndarray:
    return np.clip(h, h_min, h_max)
