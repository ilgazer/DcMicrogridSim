import asyncio

import pandas as pd

from generator import DieselGenerator, SolarPanel, Grid
from load import UCIDataLoad
from storage import Storage
from system import System


async def main():
    global_start = pd.Timedelta(days=0)

    system = System()
    system.add(Grid(10_000, 7, 5, 1))
    system.add(DieselGenerator(7500, 0.2, 0.6))
    system.add(SolarPanel(4 * 3000, pd.Timestamp("2021-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2006-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2007-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2008-12-26 0:00:00") + global_start))
    system.add(UCIDataLoad(pd.Timestamp("2009-12-26 0:00:00") + global_start))
    system.add(Storage(4_000, 0.9, 1))

    try:
        await system.simulate(300 * 24, 0.05)
    finally:
        system.graph()
        system.stats()


if __name__ == '__main__':
    asyncio.run(main())
