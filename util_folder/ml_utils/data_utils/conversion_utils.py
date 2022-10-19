import numpy as np

def convert_seconds_to_24h(seconds):
    hours = seconds // 60 // 60
    minutes = seconds // 60 % 60
    seconds = seconds % 60
    return hours, minutes, seconds

def trig_transform(z, max_val):
    cos_val = np.cos(2*np.pi*z/max_val)
    sin_val = np.sin(2*np.pi*z/max_val)
    return cos_val, sin_val