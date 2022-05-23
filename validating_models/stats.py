from contextlib import contextmanager
import pandas as pd
import time
from functools import partial
import numpy as np
import os
from functools import wraps
import multiprocessing as mp

categories = []

class StatsCollector():
    def __init__(self, stats_queue = None):
        self.active = False
        self.entry_used = True
        self.categories = []
        self.hp_names = []
        self.stats_queue=stats_queue

    def activate(self, hyperparameters : list = list()):
        self.active = True
        if self.stats_queue == None:
            self.stats_queue = mp.Manager().Queue(-1)
        self.hyperparameters = pd.DataFrame(columns=hyperparameters)
        self.stats = pd.DataFrame(columns=self.categories)
        self.hp_names = hyperparameters
    
    def add_category(self, category):
        self.categories.append(category)

    def error(self):
        self.stats.loc[len(self.stats.index) -1] = [np.nan for i in range(len(self.stats.columns))]
        self.entry_used = True

    def new_run(self, hyperparameters : dict = {}):
        if self.active and self.entry_used:
            self.stats.loc[len(self.stats.index)] = [0.0 for i in range(len(self.stats.columns))]
            self.entry_used = False
            self.hyperparameters = pd.concat([self.hyperparameters, pd.DataFrame([hyperparameters], columns = self.hp_names)])

    def new_time(self, category, time):
        if self.active:
            if category not in self.categories:
                self.add_category(category)
                self.stats[category] = 0.0
            self.stats.loc[len(self.stats.index) - 1, category] += time
            self.entry_used = True
    
    def new_entry(self, category, entry):
        if self.active:
            if category not in self.categories:
                self.add_category(category)
                self.stats[category] = np.nan
            self.stats.loc[len(self.stats.index) - 1, category] = entry
            self.entry_used = True
    
    def _receive(self):
        self.stats_queue.put('EOF')
        item = self.stats_queue.get()
        while item != 'EOF':
            category, time = item
            self.new_time(category, time)
            item = self.stats_queue.get()
    
    def get_stats_queue(self):
        return self.stats_queue
    
    def to_file(self, file, categories: list = None):
        self._receive()
        if categories:
            include = list(set(self.stats).intersection(set(categories)))
            if len(include) > 0:
                self.stats[include].to_csv(file, index=False,mode = 'a', header=not os.path.isfile(file))
        else:
            self.stats.to_csv(file, index=False,mode = 'a', header=not os.path.isfile(file))
        self.hyperparameters.to_csv(f'{file}_hps.csv', index = False, mode = 'a', header=not os.path.isfile(f'{file}_hps.csv'))
        self._shutdown()
    
    def _shutdown(self):
        del self.stats_queue
        self.stats_queue = None

STATS_COLLECTOR = StatsCollector()

def process_stats_initializer(stats_collector):
    global STATS_COLLECTOR
    STATS_COLLECTOR = stats_collector

def get_process_stats_initalizer_args():
    global STATS_COLLECTOR
    return (STATS_COLLECTOR,)

def timeit(category,func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with measure_time(category):
            res = func(*args, **kwargs)
        return res
    return wrapper

@contextmanager
def measure_time(category):
    try:
        start = time.time()
        yield None
    finally:
        end = time.time()
        global STATS_COLLECTOR
        q = STATS_COLLECTOR.get_stats_queue()
        if q:
            q.put((category,end-start))
        else:
            pass

def get_decorator(category):
    global STATS_COLLECTOR
    STATS_COLLECTOR.add_category(category)
    return partial(timeit, category)

def get_hyperparameter_value(name):
    try:
        value = STATS_COLLECTOR.hyperparameters.loc[len(STATS_COLLECTOR.hyperparameters.index) - 1,name]
    except:
        value = None
    return value

def new_entry(category, value):
    global STATS_COLLECTOR
    q = STATS_COLLECTOR.get_stats_queue()
    if q:
        q.put((category, value))
    else:
        pass
        #print("Queue not set!")
