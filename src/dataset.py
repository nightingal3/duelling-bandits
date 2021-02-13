from datetime import datetime
import pandas as pd
import numpy as np


class Dataset():
    def __init__(self, filename, effect_size, num_actions, combinations, duel_log, ts_result, uni_result, env):
        self.tables = pd.HDFStore(filename, append = False)
        self.tables['ts_combinations'] = pd.DataFrame(data = combinations['ts'])
        self.tables['uni_combinations'] = pd.DataFrame(data = combinations['uni'])
        self.tables['ts_power_over_time'] = pd.DataFrame(data = duel_log['ts']['power_over_time'])
        self.tables['uni_power_over_time'] = pd.DataFrame(data = duel_log['uni']['power_over_time'])
        self.tables['ts_result'] = pd.DataFrame(ts_result)
        self.tables['uni_result'] = pd.DataFrame(uni_result)
        self.tables['env'] = pd.DataFrame(data = env)
