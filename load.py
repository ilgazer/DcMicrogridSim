import numpy as np
import pandas as pd

from utils import generate_amounts_array


class Load:
    consumption_amounts: np.array
    delta_t: float

    def initialize(self, sim_length: int, delta_t: float):
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


class UCIDataLoad(Load):
    """
    Load load data from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    """

    start_timestamp: pd.Timestamp

    def __init__(self, start_timestamp: pd.Timestamp):
        self.start_timestamp = start_timestamp

    def initialize(self, sim_length: int, delta_t: float):
        super().initialize(sim_length, delta_t)
        self.consumption_amounts = generate_amounts_array(
            indices,
            data["Nonheat_active_power"],
            delta_t,
            sim_length,
            self.start_timestamp
        )

    def consume(self, index: int) -> float:
        return self.consumption_amounts[index]
