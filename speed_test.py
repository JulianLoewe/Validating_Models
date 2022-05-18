from argparse import ArgumentParser
from pathlib import Path
from socket import timeout
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from validating_models.dataset import BaseDataset
from validating_models.constraint import ShaclSchemaConstraint
from sklearn.tree import DecisionTreeClassifier
from validating_models.shacl_validation_engine import ReducedTravshaclCommunicator
import validating_models.visualizations as viz
from validating_models.checker import Checker
import pickle
import os
from pyinstrument import Profiler
from functools import wraps
import random 
from pebble import ProcessExpired, concurrent
from concurrent.futures import TimeoutError
from validating_models.models.decision_tree import get_shadow_tree_from_checker
from validating_models.stats import new_entry
import time

random.seed(42)
np.random.seed(42)

# Dataset Parameters
N_CLUSTERS_PER_CLASS = 1
N_INFORMATIVE = 2
N_REDUNDANT = 0
N_REPEATED = 0
N_FEATURES = 2

CACHE = '.test_cache'
#################################################################################################
# Experiment Setup Functions                                                                    #
#################################################################################################

def file_cache(what):
    WHAT_CACHE = os.path.join(CACHE, what)
    def outer_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(WHAT_CACHE, exist_ok=True)
            what_id = '_'.join([str(arg) for arg in list(args) + list(kwargs.values()) if '<' not in str(arg)])
            what_cache_path = os.path.join(WHAT_CACHE, f'{what_id}.pickle')
            if not os.path.isfile(what_cache_path):
                res = func(*args, **kwargs)
                with open(what_cache_path, 'wb') as f:
                    pickle.dump(res,f)
            else:
                print(f'Reusing {what_cache_path}')
                with open(what_cache_path, 'rb') as f:
                    res = pickle.load(f)
            return res
        return wrapper
    return outer_wrapper

@file_cache('dataframe')
def create_data_frame(n_samples, n_classes):
    X, y = make_classification(n_samples=n_samples, n_features=N_FEATURES, n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT, n_repeated=N_REPEATED, n_classes=n_classes, n_clusters_per_class=N_CLUSTERS_PER_CLASS)
    y = y.reshape((-1,1))
    column_names = [f'f_inf{i}' for i in range(N_INFORMATIVE)] + [f'f_red{i}' for i in range(N_REDUNDANT)] + [f'f_rep{i}' for i in range(N_REPEATED)] + [f'f_noise{i}' for i in range(N_FEATURES - N_INFORMATIVE - N_REDUNDANT - N_REPEATED)] + ['TARGET']
    df = pd.DataFrame(data=np.hstack((X,y)), columns=column_names)
    return df

@file_cache('dataset')
def create_dataset(n_samples, n_classes, n_nodes, node_range = None):
    print('Creating Dataset...', end='')
    df = create_data_frame(n_samples, n_classes)
    if node_range == None:
        node_range = range(n_nodes)
    nodes = [f'entity_{int(i):010d}' for i in node_range]
    sample_to_node_mapping = pd.DataFrame(data=np.random.choice(nodes, size=(n_samples,1)), columns=['x'])
    dataset = BaseDataset(df, target_name = 'TARGET', seed_query = None, seed_var = 'x', sample_to_node_mapping = sample_to_node_mapping, random_schema_validation = True, categorical_mapping={'TARGET': {0:'0', 1: '1'}})
    print('Done')
    return dataset

@file_cache('model')
def train_decision_tree(max_depth, n_samples, n_classes, dataset):
    print('Training Tree...', end='')
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth)
    tree_classifier.fit(dataset.x_data(), dataset.y_data())
    print('Done')
    return tree_classifier

def profile(func, id, *args, pass_id=False, **kwargs):
    profiler = Profiler()
    profiler.start()
    start = time.time()
    if pass_id:
        res = func(id, *args, **kwargs)
    else:
        res = func(*args, **kwargs)
    end = time.time()
    new_entry('overall',end-start)
    profiler.stop()
    os.makedirs(f".profiles/{func.__name__}", exist_ok=True)
    with open(f".profiles/{func.__name__}/profil_{id}.html","w") as f:
        f.write(profiler.output_html())
    
    return res


#################################################################################################
# The different Experiments                                                                     #
#################################################################################################

def exp_validating_models_dtreeviz(id, tree_classifier, checker, constraints,non_applicable_counts=True, coverage=True, fancy=True, visualize_in_parallel=True):
    plot = viz.decision_trees.dtreeviz(tree_classifier, checker, constraints, non_applicable_counts=non_applicable_counts, coverage=coverage, fancy=fancy, visualize_in_parallel=visualize_in_parallel)
    os.makedirs(f".output", exist_ok=True)
    plot.save(f'.output/test_{id}.svg')

@concurrent.process(timeout=300, daemon=False)
def dtreeviz_experiment(id,visualize_in_parallel, max_depth = 5, n_classes=2, n_samples=4**10, n_nodes=4**10, n_constraints=5, fancy=True, coverage=True):
    from validating_models.stats import STATS_COLLECTOR 
    STATS_COLLECTOR.activate(hyperparameters=['visualize_in_parallel','max_depth','n_constraints','n_samples','n_nodes','fancy','coverage'])
    STATS_COLLECTOR.new_run(hyperparameters=[visualize_in_parallel, max_depth, n_constraints, n_samples, n_nodes, fancy, coverage])

    print(f'Running experiment {id} with [max_depth = {max_depth}, n_classes = {n_classes}, n_samples = {n_samples}, n_nodes = {n_nodes}, n_constraints = {n_constraints}, parallel = {visualize_in_parallel}, fancy = {fancy}, coverage = {coverage}]')
    
    constraints = [ ShaclSchemaConstraint(name=f'Constraint {i}',shape_schema_dir=f'dir{i}', target_shape='ts') for i in range(n_constraints)]
    dataset = create_dataset(n_samples, n_classes, n_nodes)
    tree_classifier = train_decision_tree(max_depth, n_samples, n_classes, dataset)
    checker = Checker(tree_classifier.predict, dataset)

    profile(exp_validating_models_dtreeviz,id, tree_classifier, checker, constraints, non_applicable_counts=True, coverage=coverage, fancy=fancy, visualize_in_parallel=visualize_in_parallel, pass_id=True)

    STATS_COLLECTOR.to_file('parallel_vs_serial_times.csv')

@concurrent.process(timeout=300, daemon=False)
def samples_to_node_experiment(id, node_to_samples_non_optimized, max_depth = 5, n_samples = 4**10, n_classes = 2, n_nodes=4**10):
    from validating_models.stats import STATS_COLLECTOR
    STATS_COLLECTOR.activate(hyperparameters=['max_depth', 'n_samples', 'node_to_samples_non_optimized'])
    STATS_COLLECTOR.new_run(hyperparameters=[max_depth, n_samples, node_to_samples_non_optimized])

    print(f'Running experiment samples_to_node_experiment with [max_depth = {max_depth}, n_samples = {n_samples}, optimized = {not node_to_samples_non_optimized}]')
    
    dataset = create_dataset(n_samples, n_classes, n_nodes)
    tree_classifier = train_decision_tree(max_depth, n_samples, n_classes, dataset)
    checker = Checker(tree_classifier.predict, dataset)
    skd_tree = get_shadow_tree_from_checker(tree_classifier,checker)

    from validating_models.models.decision_tree import get_node_samples
    profile(get_node_samples,id, skd_tree)

    STATS_COLLECTOR.to_file('samples_to_node_times.csv')

@concurrent.process(timeout=300, daemon=False)
def join_strategie_experiment(id, use_outer_join, order_by_cardinality, n_samples = 4**10, n_nodes = 4**10, n_constraints = 5, n_classes=2):
    from validating_models.stats import STATS_COLLECTOR
    STATS_COLLECTOR.activate(hyperparameters = ['n_samples','n_nodes','n_constraints','use_outer_join', 'order_by_cardinality'])
    STATS_COLLECTOR.new_run(hyperparameters = [n_samples, n_nodes, n_constraints, use_outer_join, order_by_cardinality])

    constraints = [ ShaclSchemaConstraint(name=f'Constraint {i}', shape_schema_dir=f'dir{i}', target_shape='ts') for i in range(n_constraints)]
    dataset = create_dataset(n_samples, n_classes, n_nodes)
    profile(dataset.get_shacl_schema_validation_results, id, constraints)

    STATS_COLLECTOR.to_file('join_strategie_times.csv')



def star_schema_experiment_generator(endpoint, api_config, shape_schema_dir, n_nodes=4000, n_constraints=24, n_samples=20000):
    constraints = [ ShaclSchemaConstraint(name=f'Constraint {i}',shape_schema_dir=shape_schema_dir, target_shape=f'Class{i+1}') for i in range(n_constraints)]
    dataset = create_dataset(n_samples=n_samples, n_nodes=n_nodes, n_classes=2, node_range=[4000,8000])
    dataset.communicator = ReducedTravshaclCommunicator("",endpoint, api_config)
    dataset.seed_query = "SELECT ?x WHERE { ?x a <http://example.com/Qs> }"
    dataset.random_schema_validation = False
    return dataset, constraints



@concurrent.process(timeout=300, daemon=False)
def validation_engine_experiment(id, experiment_generator, endpoint, api_config, shape_schema_dir, n_nodes, n_constraints, n_samples, constraints_separate):
    from validating_models.stats import STATS_COLLECTOR

    STATS_COLLECTOR.activate(hyperparameters = ['n_samples','n_nodes','shape_schema_dir','api_config'])
    STATS_COLLECTOR.new_run(hyperparameters = [n_samples, n_nodes, shape_schema_dir, api_config])
    dataset, constraints = experiment_generator(endpoint, api_config, shape_schema_dir, n_nodes, n_constraints, n_samples)
    if constraints_separate:
        for constraint in constraints:
            profile(dataset.get_shacl_schema_validation_results, id, [constraint])
    else:
        profile(dataset.get_shacl_schema_validation_results, id, constraints)
    STATS_COLLECTOR.to_file('validation_engine.csv')

#################################################################################################
# Running the experiments                                                                       #
#################################################################################################

def main():
    parser = ArgumentParser(description='Runs the specified experiment')
    parser.add_argument('experiment', type=str)
    args = parser.parse_args()


    NUM_REPS = 5
    
    if args.experiment == "VizDataset":
        dataset = create_dataset(200, 2, 200)
        x = dataset.x_data()
        y = dataset.y_data().squeeze()
        x_class_0 = x[(y == 0),:]
        x_class_1 = x[(y == 1),:]

        import matplotlib.pyplot as plt
        plt.plot(x_class_0[:,0],x_class_0[:,1],'ro', label='Class 0')
        plt.plot(x_class_1[:,0],x_class_1[:,1], 'bo', label='Class 1')
        plt.ylabel('Feature 2')
        plt.xlabel('Feature 1')
        plt.legend()
        plt.savefig('artificial_dataset.png')

    elif args.experiment == "validation_engine":
        experiment_generators = [star_schema_experiment_generator]
        all_heuristics_config = 'speed_test_shacl_api_configs/all_heuristics_config.json'
        no_heuristics_config = 'speed_test_shacl_api_configs/no_heuristics.json'
        endpoint = 'http://localhost:14000/sparql'
        try:
            validation_engine_experiment(f"star_graph_25_{no_heuristics_config.replace('/','_')}_False", star_schema_experiment_generator, endpoint, no_heuristics_config, 'speed_test_shape_schemes/star_graph_25', n_nodes=4000, n_constraints=24, n_samples=20000, constraints_separate=False).result()
        except Exception as e:
            print(e)

        # for generator in experiment_generators:
        #     for config in Path('speed_test_shacl_api_configs').glob('*.json'):
        #         api_config = str(config)
        #         validation_engine_experiment(f"star_graph_25_{no_heuristics_config.replace('/','_')}_False", generator, endpoint, no_heuristics_config, 'speed_test_shape_schemes/star_graph_25', n_nodes=4000, n_constraints=24, n_samples=20000, constraints_separate=False).result()
        #     validation_engine_experiment(f"star_graph_25_{all_heuristics_config.replace('/','_')}_True", generator, endpoint, all_heuristics_config, 'speed_test_shape_schemes/star_graph_25', n_nodes=4000, n_constraints=24, n_samples=20000, constraints_separate=False).result()

    elif args.experiment == "nodesamples":
        nsamples_list = np.logspace(4,12, base=4, num = 20, dtype=np.int_)
        max_depths = np.array([2,4,6,8,10])
        for non_optimized in [False, True]:
                for n_samples in nsamples_list:
                    for max_depth in max_depths:
                        for k in range(NUM_REPS):
                            samples_to_node_experiment(f'node_samples_{non_optimized}_{n_samples}_{max_depth}_{k}', n_samples=n_samples, max_depth=max_depth, node_to_samples_non_optimized=non_optimized).result()

    elif args.experiment == "join":    
        nsamples_list = np.linspace(4**4,4**11, num = 20, dtype=np.int_)
        n_nodes_list = np.linspace(4**4,4**11, num= 20, dtype=np.int_)
        n_constraints_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        for n_samples in nsamples_list:
            for join_outer in [False, True]:
                if join_outer:
                    order_by_cardinality_set = [True,False]
                else:
                    order_by_cardinality_set = [False]
                for order_by_cardinality in order_by_cardinality_set:
                    for k in range(NUM_REPS):
                        join_strategie_experiment(f'{join_outer}-{order_by_cardinality}-nsamples{n_samples}-{k}',join_outer, order_by_cardinality, n_samples=n_samples).result()
            
        for n_nodes in n_nodes_list:
            for join_outer in [False, True]:
                if join_outer:
                    order_by_cardinality_set = [True,False]
                else:
                    order_by_cardinality_set = [False]
                for order_by_cardinality in order_by_cardinality_set:
                    for k in range(NUM_REPS):
                        join_strategie_experiment(f'{join_outer}-{order_by_cardinality}-nnodes{n_nodes}-{k}',join_outer, order_by_cardinality, n_nodes=n_nodes).result()
        
        for n_constraints in n_constraints_list:
            for join_outer in [False, True]:
                if join_outer:
                    order_by_cardinality_set = [True,False]
                else:
                    order_by_cardinality_set = [False]
                for order_by_cardinality in order_by_cardinality_set:
                    for k in range(NUM_REPS):
                        join_strategie_experiment(f'{join_outer}-{order_by_cardinality}-nconstraints{n_constraints}-{k}',join_outer, order_by_cardinality, n_constraints=n_constraints).result()

    elif args.experiment == "treevizParallelSerial":
        nsamples_list = np.linspace(4**4,4**11, num = 20, dtype=np.int_)
        n_nodes_list = np.linspace(256,4**10, num= 20, dtype=np.int_)
        n_constraints_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        max_depths = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

        for n_samples in nsamples_list:
            for visualize_in_parallel in [False, True]:
                for k in range(NUM_REPS):
                    try:
                        dtreeviz_experiment(f'{visualize_in_parallel}-nsamples{n_samples}-{k}',visualize_in_parallel, n_samples=n_samples).result()
                    except Exception as e:
                        print(e)
        
        for n_nodes in n_nodes_list:
            for visualize_in_parallel in [False, True]:
                for k in range(NUM_REPS):
                    try:
                        dtreeviz_experiment(f'{visualize_in_parallel}-nnodes{n_nodes}-{k}',visualize_in_parallel, n_nodes=n_nodes).result()
                    except Exception as e:
                        print(e)
        
        for n_constraints in n_constraints_list:
            for visualize_in_parallel in [False, True]:
                for k in range(NUM_REPS):
                    try:
                        dtreeviz_experiment(f'{visualize_in_parallel}-nconstraints{n_constraints}-{k}',visualize_in_parallel, n_constraints=n_constraints).result()
                    except Exception as e:
                        print(e)

        for max_depth in max_depths:
            for visualize_in_parallel in [False, True]:
                for k in range(NUM_REPS):
                    try:
                        dtreeviz_experiment(f'{visualize_in_parallel}-maxdepth{max_depth}-{k}',visualize_in_parallel, max_depth=max_depth).result()
                    except Exception as e:
                        print(e)

if __name__ == '__main__':
    main()
    




