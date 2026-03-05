import numpy as np

def elevation_angle(x_bs, y_bs, h_bs, x_uav, y_uav, h_uav):
    d = np.sqrt((x_uav - x_bs)**2 + (y_uav - y_bs)**2 + (h_uav - h_bs)**2)
    
    theta = 180/np.pi * np.arcsin((h_uav - h_bs) / d)
    return theta, d

def plos(theta, a, b):
    return 1 / (1 + a * np.exp(-b * (theta - a)))

def avg_pathloss_db(d, plos, fc, eta1_db, eta2_db, c=3e8):
    """
    Expected ATG path loss in dB (LoS/NLoS weighted).
    Average in linear domain, then convert to dB.
    """
    FSPL_db = 20 * np.log10(4 * np.pi * fc * d / c)

    PL_LoS_db  = FSPL_db + eta1_db
    PL_NLoS_db = FSPL_db + eta2_db

    L_L = 10 ** (PL_LoS_db / 10.0)
    L_N = 10 ** (PL_NLoS_db / 10.0)

    L_avg = plos * L_L + (1.0 - plos) * L_N
    PL_avg_db = 10.0 * np.log10(np.maximum(L_avg, 1e-30))
    return PL_avg_db

def snr_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm):
    """
    SNR in dB: SNR = Rx_power(dBm) - Noise(dBm)
    """
    rx_power_dbm = P_tx_dbm - PL_db
    snr_db = rx_power_dbm - noise_dbm
    return snr_db

def db_to_linear(x_db):
    return 10 ** (x_db / 10.0)

def linear_to_db(x_lin):
    return 10.0 * np.log10(np.maximum(x_lin, 1e-30))

def snr_rayleigh_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm, rng=np.random):
    """
    Instantaneous SNR with Rayleigh small-scale fading (power gain ~ Exp(1)).
    Uses deterministic large-scale path loss PL_db (in dB), then applies fading.
    Returns SNR in dB.
    """
    # Average SNR (no fading) in linear
    snr_avg_db = (P_tx_dbm - PL_db) - noise_dbm
    snr_avg_lin = db_to_linear(snr_avg_db)

    # Rayleigh fading power gain: Exp(1)
    fad = rng.exponential(scale=1.0, size=np.shape(snr_avg_lin))

    snr_lin = snr_avg_lin * fad
    return linear_to_db(snr_lin)