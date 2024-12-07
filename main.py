import asyncio
import traceback

import pandas as pd
import tqdm
from pygad import pygad

from generator import DieselGenerator, SolarPanel, Grid
from load import UCIDataLoad
from storage import Storage
from system import System

diesel_eur_per_l = 1.23

generators = [
    {"name": "Hyundai DHY45KSE", "capacity": 32000, "price": 12599, "l_per_h": 10.9},
    {"name": "Hyundai DHY34KSE", "capacity": 28000, "price": 10599, "l_per_h": 9.5},
    {"name": "Hyundai DHY22KSE", "capacity": 18000, "price": 8699, "l_per_h": 6.5},
    {"name": "Hyundai DHY22KSE", "capacity": 10000, "price": 7700, "l_per_h": 4},
    {"name": "Hyundai DHY12000XSE", "capacity": 9000, "price": 6199, "l_per_h": 4.3},
    {"name": "Hyundai DHY8600SE-T", "capacity": 5500, "price": 1849, "l_per_h": 2.4},
]


def add_loads(global_start, system):
    system.add(UCIDataLoad(pd.Timestamp("2006-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2007-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2008-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2009-12-26 0:00:00") + global_start))


async def simulate(generator, solar_wp, storage_wh, show=False):
    days = 300
    global_start = pd.Timedelta(days=0)

    system = System()
    system.add(Grid(10_000, 9, 7, 0))
    eur_per_wh = generator["l_per_h"] * diesel_eur_per_l / generator["capacity"]
    system.add(DieselGenerator(generator["price"], eur_per_wh, generator["capacity"] * 0.75, 0.2, 0.6))
    system.add(SolarPanel(solar_wp, pd.Timestamp("2021-12-26 0:00:00") + global_start))
    add_loads(global_start, system)
    system.add(Storage(storage_wh, 0.9, 1))

    try:
        await system.simulate(days * 24, 0.05)
    except Exception:
        txt = traceback.format_exc()
        if "Battery depleted" not in txt:
            print(txt)
        elif show:
            print(f"[{generator}, {solar_wp:.0f}, {storage_wh:.0f}] Failed")
        return False

    initial_cost = system.get_initial_cost()
    operation_cost = system.get_operation_cost()
    cost10y = initial_cost + operation_cost * 3650 / days

    if show:
        system.graph(show_each=False,
                     figname=f"{generator['name']}_{solar_wp / 1000:.2f}kWp_{storage_wh / 1000:.2f}kWh")
        system.stats(show_each=True)
        print(f"[{solution[0]}, {solution[1]:.0f}, {solution[2]:.0f}]")
        print(f"{initial_cost:.0f}€ initial + {operation_cost / days:.0f}€/day")
        print(f"10 Year Total Cost: {cost10y:.0f}€")

    return cost10y


async def test20h(generator, solar_wp, storage_wh):
    global_start = pd.Timedelta(days=0)

    system = System()
    eur_per_wh = generator["l_per_h"] * diesel_eur_per_l / generator["capacity"]
    system.add(DieselGenerator(generator["price"], eur_per_wh, generator["capacity"] * 0.75, 0.2, 0.6))
    add_loads(global_start, system)
    system.add(Storage(storage_wh, 0.9, 1))

    try:
        await system.simulate(20, 0.05)
    except Exception:
        txt = traceback.format_exc()
        if "Battery depleted" not in txt:
            print(txt)

        return False
    return True


async def test7h(generator, solar_wp, storage_wh):
    global_start = pd.Timedelta(days=0)

    system = System()
    add_loads(global_start, system)
    system.add(Storage(storage_wh, 0.9, 1, storage_wh))

    try:
        await system.simulate(7, 0.05)
    except Exception:
        txt = traceback.format_exc()
        if "Battery depleted" not in txt:
            print(txt)

        return False
    return True


def fitness_function(ga_instance, solution, solution_idx):
    passed7h = asyncio.run(test7h(generators[solution[0]], solution[1], solution[2]))
    passed20h = asyncio.run(test20h(generators[solution[0]], solution[1], solution[2]))
    cost10y = asyncio.run(simulate(generators[solution[0]], solution[1], solution[2]))
    if False in [passed7h, passed20h, cost10y]:
        # print(f"[{solution[0]}, {solution[1]:.0f}, {solution[2]:.0f}] Failed")
        return 0
    # print(f"[{solution[0]}, {solution[1]:.0f}, {solution[2]:.0f}] Cost: {cost10y:.0f}€")

    return 1 / cost10y


ga_instance = None
if __name__ == "__main__":
    for e in range(10):
        gene_space = [
            {"low": 0, "high": len(generators), "step": 1},
            {"low": 0, "high": 20_000},
            {"low": 0, "high": 20_000}
        ]
        tq = tqdm.tqdm(total=30)
        ga_instance = pygad.GA(
            num_generations=30,
            num_parents_mating=50,
            fitness_func=fitness_function,
            sol_per_pop=400,
            num_genes=3,
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_probability=0.6,
            gene_space=gene_space,
            gene_type=[int, float, float],
            on_generation=lambda _: tq.update(),
        )
        ga_instance.run()
        tq.close()

        solution, _, _ = ga_instance.best_solution()
        asyncio.run(simulate(generators[solution[0]], solution[1], solution[2], show=True))
