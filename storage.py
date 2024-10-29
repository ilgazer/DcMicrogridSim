import numpy as np


class Storage:
    storage_capacities: np.ndarray
    storage_amounts: np.ndarray
    initial_capacity: float
    charge_efficiency: float
    degradation: float
    delta_t: float

    def __init__(self, initial_capacity: float, charge_efficiency: float, degradation: float):
        """
        :param initial_capacity: The capacity of the battery at t=0.
        :param efficiency: The charge/discharge efficiency where 0.9 means 100Wh input translates to 90 Wh stored and available.
        :param degradation: Degradation per hour
        """
        self.initial_capacity = initial_capacity
        self.charge_efficiency = charge_efficiency
        self.degradation = degradation

    async def initialize(self, sim_length: int, delta_t: float):
        self.delta_t = delta_t

        self.storage_capacities = np.zeros(sim_length)
        self.storage_amounts = np.zeros(sim_length)

        self.storage_capacities[0] = self.initial_capacity
        self.storage_amounts[0] = self.initial_capacity / 2

    def update(self, index: int, power_ratio: float) -> tuple[float, float]:
        """
        :param delta_t: The time step size in hours.
        :param index: The current number of ticks.
        :param power_ratio: The ratio of net power the battery will store(+)/provide(-) this tick to its capacity.
        :return:
        """
        share_of_net_power = power_ratio * self.storage_capacities[index - 1]

        net_power = share_of_net_power * self.charge_efficiency if share_of_net_power > 0 else share_of_net_power
        capacity = self.storage_capacities[index - 1] * self.degradation
        amount = self.storage_amounts[index - 1] + net_power * self.delta_t

        self.storage_capacities[index] = capacity
        self.storage_amounts[index] = min(amount, capacity)

        if amount < 0:
            raise Exception(f"Battery depleted at t={index * self.delta_t}")

        return capacity, min(amount, capacity)
