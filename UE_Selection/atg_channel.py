import numpy as np

def elevation_angle(x_bs, y_bs, h_bs, x_uav, y_uav, h_uav):
    d = np.sqrt((x_uav - x_bs)**2 + (y_uav - y_bs)**2 + (h_uav - h_bs)**2)
    
    theta = 180/np.pi * np.arcsin((h_uav - h_bs) / d)
    return theta, d

def plos(theta, a, b):
    return 1 / (1 + a * np.exp(-b * (theta - a)))

def avg_pathloss_db(d, plos, fc, eta1_db, eta2_db, c=3e8):
    """
    Average ATG path loss in dB (LoS/NLoS weighted) — Al-Hourani style
    """
    FSPL_db = 20 * np.log10(4 * np.pi * fc * d / c)

    PL_LoS_db  = FSPL_db + eta1_db
    PL_NLoS_db = FSPL_db + eta2_db

    PL_avg_db = plos * PL_LoS_db + (1 - plos) * PL_NLoS_db
    return PL_avg_db

def snr_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm):
    """
    SNR in dB: SNR = Rx_power(dBm) - Noise(dBm)
    """
    rx_power_dbm = P_tx_dbm - PL_db
    snr_db = rx_power_dbm - noise_dbm
    return snr_db