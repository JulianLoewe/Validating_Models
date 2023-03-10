# Multiple models --> varying importants (weights)
# Bagging -> Boostrap Aggregation => Taking random sub-samples with replacement and train a number of models
# Boosting -> Depending on the last model build new one, Adding importants to missclassified samples

from collections import Counter
from typing import List
from ..checker import Checker, DecisionNodeChecker
from ..colors import adjust_colors
from ..constraint import Constraint
import numpy as np
from ..groupings.classification import group_by_gt_class
from ..models.random_forest import get_shadow_forest_from_checker
from ..models.decision_tree import get_shadow_tree_from_checker
from ..groupings.general import group_by_complete_dataset
from ..drawing_utils import draw_legend
from ..frequency_distribution_table import FrequencyDistributionTable
from ..visualizations import graphviz_helper 
from ..visualizations import decision_trees
from tqdm import tqdm
from ..models.decision_tree import get_single_node_samples

def node_name(i):
    return f'estimator{i}'

def fix_svg_file(path):
    missing_header = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    with open(path, 'r') as original:
        svg_content = original.read()
    
    with open(path, 'w') as new:
        new.write(missing_header)
        new.write(svg_content)

def group_results_by_clustering(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores = []
    all_labels = []
    unique_samples = np.unique(X,axis=0)
    if unique_samples.shape[0] >= 2:
        for i in tqdm(range(2,unique_samples.shape[0] + 1),desc='Grid Search for best clustering'):
            labels = KMeans(n_clusters=i).fit_predict(X)
            try:
                scores.append(silhouette_score(X, labels))
            except:
                continue
            all_labels.append(labels)
        best_idx = np.argmax(np.array(scores))
        print("Best Silhouette Score: " + str(np.max(np.array(scores))))
    else:
        return np.zeros((X.shape[0],)), 1
    return all_labels[best_idx], best_idx + 2 


def random_forest_viz(model,
                    checker: Checker,
                    constraints: List[Constraint],
                    group_functions = group_by_gt_class,
                    coverage: bool = False,
                    non_applicable_counts=False,
                    perform_clustering=False,
                    only_use_train_instances=False,
                    X = None,
                    colors = None):
    colors = adjust_colors(colors)

    shadow_forest = get_shadow_forest_from_checker(model, checker)
    n_estimators = len(shadow_forest.estimators)
    nodes = []
    fdts = []
    plots = []

    if isinstance(X,np.ndarray):
        y_pred_forest = shadow_forest.predict(X.reshape((1,-1)))[0]
        y_preds_tree = []
    else:
        y_pred_forest = np.inf

    for i in tqdm(range(n_estimators), desc='Evaluating Decision Trees'): 
        estimator_checker = Checker(shadow_forest.estimators[i].predict, checker.dataset, use_gt=checker._use_gt)
        shadow_tree = get_shadow_tree_from_checker(shadow_forest.estimators[i], estimator_checker)

        if isinstance(X,np.ndarray):
            y_preds_tree.append(shadow_forest.estimators[i].predict(X.reshape((1,-1)))[0])
            node = shadow_tree.predict_path(X)[-1]
            estimator_checker = DecisionNodeChecker(
                node, checker.dataset, use_gt=checker._use_gt)
        else:
            node = shadow_tree.root
        
        if only_use_train_instances:
            samples = set(get_single_node_samples(node,only_calculate_single_node=True)).intersection(set(shadow_forest.get_bootstrap_indices(i)))
        else:
            samples = get_single_node_samples(node, only_calculate_single_node=True)

        fdt = FrequencyDistributionTable(estimator_checker,constraints, list(samples), group_functions, coverage=coverage, non_applicable_counts=non_applicable_counts)
        fdts.append(fdt)
    
    if perform_clustering:
        labels, n_clusters = group_results_by_clustering(np.stack(list(map(lambda fdt: fdt.fdt.values.flatten(), fdts))))
        group_fdt_idx = {}
        for i in range(n_clusters):
            group_fdt_idx[i] = list(np.where(labels == i)[0])
    else:
        group_fdt_idx = {i:[i] for i in range(len(fdts))}

    for i in tqdm(range(len(group_fdt_idx.keys())), desc='Visualizing Results'):
        if len(group_fdt_idx[i]) > 0:
            plot = fdts[group_fdt_idx[i][0]].visualize('DT ' + ','.join([str(j) for j in group_fdt_idx[i]]))
            plots.append(plot)
            viz_path = graphviz_helper.get_image_path(node_name(i))
            plot.save(viz_path)
            cluster_pred = Counter([y_preds_tree[j] for j in group_fdt_idx[i]]).most_common(1)[0][0]
            highlight = cluster_pred == y_pred_forest if isinstance(X,np.ndarray) else False
            nodes.append(graphviz_helper.node_stmt(node_name(i),graphviz_helper.html_image('',viz_path),highlight ,colors))

    legend = draw_legend(plots)
    legend_path = graphviz_helper.get_image_path('legend')
    legend.save(legend_path)

    return graphviz_helper.grid_layout('Random Forest', nodes,[],legend_path, colors, orientation='TD')

def compare_decision_trees(model, checker: Checker, constraints: List[Constraint], tree_indizes: list, colors=None, **args):
    colors = adjust_colors(colors)
    shadow_forest = get_shadow_forest_from_checker(model, checker)
    nodes = []
    for i in tqdm(tree_indizes, desc='Evaluating Constraints and Visualizing Decision Trees'):
        shadow_tree = get_shadow_tree_from_checker(shadow_forest.estimators[i], checker)
        viz = decision_trees.dtreeviz(shadow_tree, checker, constraints,indices=shadow_forest.get_bootstrap_indices(i), **args)
        viz_path = graphviz_helper.get_image_path(node_name(i))
        viz.save(viz_path)
        nodes.append(graphviz_helper.node_stmt(node_name(i),graphviz_helper.html_image('DT ' + str(i),viz_path),False,colors))
    return graphviz_helper.grid_layout('Selected Estimators', nodes,[],None,colors,size=(1,len(tree_indizes)), orientation='TD')




    
    

