from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from async_lru import alru_cache

from utils import generate_amounts_array


class Load:
    consumption_amounts: np.array
    delta_t: float

    async def initialize(self, sim_length: int, delta_t: float):
        self.consumption_amounts = np.zeros(sim_length)
        self.delta_t = delta_t

    def consume(self, index: int) -> float:
        """
        :param index: the index of the time slice
        :return: the amount of electricity consumed in W
        """
        pass


class StaticLoad(Load):
    amount: float

    def __init__(self, amount: float):
        self.amount = amount

    def consume(self, index: int) -> float:
        self.consumption_amounts[index] += self.amount
        return self.amount


@lru_cache(maxsize=None)
def get_data():
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1",
            "Sub_metering_2", "Sub_metering_3"]

    data = pd.read_csv('household_power_consumption.csv', sep=";")
    data = data[data["Global_active_power"] != "?"]
    for col in cols:
        data.loc[:, col] = pd.to_numeric(data[col], errors='coerce')
    data["Nonheat_active_power"] = data["Global_active_power"] * 1000 - data["Sub_metering_3"] * 60
    data["Timestamp"] = pd.to_datetime(data["Date"] + " " + data["Time"], dayfirst=True)
    data.set_index(["Timestamp"])
    indices = data["Timestamp"].values.copy()
    print("Done with loading data")
    return data, indices


if __name__ == '__main__':
    data, indices = get_data()
    first = data["Timestamp"][0]
    x = (data["Timestamp"] - first) / pd.Timedelta('1h')
    plt.plot(x, data["Nonheat_active_power"])
    plt.show()


@alru_cache(maxsize=None)
async def generate_uci_amounts(sim_length: int, delta_t: float, start_timestamp: pd.Timestamp):
    data, indices = get_data()
    return await generate_amounts_array(indices, data["Nonheat_active_power"], sim_length, delta_t, start_timestamp)


class UCIDataLoad(Load):
    """
    Load load data from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    """

    start_timestamp: pd.Timestamp

    def __init__(self, start_timestamp: pd.Timestamp):
        self.start_timestamp = start_timestamp

    async def initialize(self, sim_length: int, delta_t: float):
        await super().initialize(sim_length, delta_t)
        self.consumption_amounts = await generate_uci_amounts(
            sim_length,
            delta_t,
            self.start_timestamp
        )

    def consume(self, index: int) -> float:
        return self.consumption_amounts[index]
