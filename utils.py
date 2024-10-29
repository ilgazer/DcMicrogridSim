import numpy as np
import pandas as pd


def limit(x, min: float, max: float):
    if x < min:
        return min
    elif x > max:
        return max
    return x


def linear(x1: float, y1: float, x2: float, y2: float):
    return lambda x: (x - x1) * (y2 - y1) / (x2 - x1) + y1


def generate_amounts_array(indices, values, delta_t, sim_length, start_timestamp):
    amounts = np.zeros(sim_length)

    end_timestamp = start_timestamp + pd.Timedelta(hours=sim_length * delta_t)
    timestamps = pd.date_range(start_timestamp, end_timestamp, periods=sim_length + 1)
    target_inds = np.searchsorted(indices, timestamps, side="left") - 1
    amounts_per_range = np.split(values, target_inds)[1:-1]
    mean = 0
    for i, rng in enumerate(amounts_per_range):
        if len(rng) != 0:
            mean = np.mean(rng)
        amounts[i] = mean
    return amounts


def find_emptys(target_inds, timestamps):
    emptys = np.r_[False, target_inds[:-1] == target_inds[1:], False]
    starts = (~emptys[:-1]) & emptys[1:]
    ends = emptys[:-1] & (~emptys[1:])
    print("\n".join([f"{s} --> {e}" for s, e in zip(timestamps[starts], timestamps[ends])]))
