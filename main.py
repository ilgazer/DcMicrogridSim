import pandas as pd

from generator import Grid, DieselGenerator
from load import UCIDataLoad
from storage import Storage
from system import System

system = System()
system.add(Grid(10_000, 7, 5, 1.5))
system.add(DieselGenerator(7500, 0.2, 0.6))
system.add(UCIDataLoad(pd.Timestamp("2006-12-20 9:00:00")))
system.add(UCIDataLoad(pd.Timestamp("2007-12-24 9:00:00")))
system.add(UCIDataLoad(pd.Timestamp("2008-12-27 9:00:00")))
system.add(UCIDataLoad(pd.Timestamp("2009-12-29 9:00:00")))
system.add(Storage(4_000, 0.8, 1))

try:
    system.simulate(90 * 24, 0.05)
finally:
    system.graph()
    system.stats()
