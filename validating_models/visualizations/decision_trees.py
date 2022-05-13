from re import X
import numpy as np
from typing import List, Union
import tempfile
import os
from sklearn.neighbors import KernelDensity

from ..frequency_distribution_table import FrequencyDistributionTable
from ..checker import Checker, ConstantModelChecker, DecisionNodeChecker
from ..constraint import TRUTH_LABELS, Constraint
from ..drawing_utils import Function, Scatter, extract_labels_handles, new_draw_legend
from ..colors import VAL_COLORS, adjust_colors

from ..groupings.general import group_by_complete_dataset
from ..groupings.decision_trees import group_by_decision_tree_leaves, group_by_decision_tree_nodes, group_by_node_split_feature
from .regression import get_feature_target_for_indices_plot
from ..models.decision_tree import get_shadow_tree_from_checker, get_node_samples, get_single_node_samples
from dtreeviz.models.shadow_decision_tree import ShadowDecTreeNode
from ..visualizations.graphviz_helper import DTreeVizConv
from dtreeviz.utils import myround
from tqdm import tqdm
import multiprocessing as mp
from pebble import ProcessPool

from validating_models.stats import get_process_stats_initalizer_args, process_stats_initializer

##################################################################
# Plots based on constraint validation counts:                   #
# Regression                                                     #
##################################################################

def get_kde_plot(node: ShadowDecTreeNode,
                 checker: Checker,
                 constraint: Constraint,
                 non_applicable_counts: bool = False,
                 graph_colors=None,
                 fontname: str = "Arial",
                 fontsize: int = 9,
                 title: str = None):

    graph_colors = adjust_colors(graph_colors)
    figsize = (.8, .75)
    y = checker.dataset.y_data()[get_single_node_samples(node)]
    m = np.mean(y)

    constraint_validation_results = checker.get_constraint_validation_result(
        [constraint], non_applicable_counts=False)[get_single_node_samples(node)]

    # Split
    invalid_idx = np.where(constraint_validation_results == 0)[0]
    valid_idx = np.where(constraint_validation_results == 1)[0]

    y_invalid = y[invalid_idx].reshape((-1, 1))
    y_valid = y[valid_idx].reshape((-1, 1))
    x_range = [np.min(y), np.max(y)]
    x_plot = np.linspace(x_range[0], x_range[1], 1000)[:, np.newaxis]

    plot = Function(figure_size=figsize)

    # if len(y_invalid) > 1:
    #     kde_invalid = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y_invalid)
    #     dens_invalid = np.exp(kde_invalid.score_samples(x_plot))
    #     plot.add_dataset(x_plot, dens_invalid, VAL_COLORS['invalid'])

    # if len(y_valid) > 1:
    #     kde_valid = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y_valid)
    #     dens_valid = np.exp(kde_valid.score_samples(x_plot))
    #     plot.add_dataset(x_plot, dens_valid, VAL_COLORS['valid'])
    y = y.reshape((-1, 1))
    kde_all = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(y)
    dens = np.exp(kde_all.score_samples(x_plot))
    plot.add_dataset(x_plot, dens, 'black', 'density')

    plot.draw()
    #plot.ax.plot(y_valid, -0.005 - 0.01 * np.random.random(y_valid.shape),'+' ,c=VAL_COLORS['valid'])
    plot.ax.plot(y_invalid, -0.01 - 0.02 *
                 np.random.random(y_invalid.shape), '+', c=VAL_COLORS['invalid'])

    plot.add_wedge(m, x_range)

    max_prop_dens = np.max(np.stack([dens], 0))

    plot.ax.plot([m, m], [0, max_prop_dens], '--',
                 color=graph_colors['split_line'], linewidth=1)

    text = f"n={node.nsamples()}"
    plot.set_xlabel(text)
    return plot


def get_target_feature_for_node_plot(node: ShadowDecTreeNode,  # Feature_name / Index,
                                     checker: Checker,
                                     constraint: Constraint,
                                     non_applicable_counts=False,
                                     figsize=(2.5, 1.1),
                                     graph_colors=None,
                                     fontname: str = "Arial",
                                     fontsize: int = 9,
                                     title: str = None):
    graph_colors = adjust_colors(graph_colors)

    plot = get_feature_target_for_indices_plot(feature=node.feature(), indices=get_single_node_samples(node), checker=checker, constraint=constraint,
                                               non_applicable_counts=non_applicable_counts, figsize=figsize, fontsize=fontsize, fontname=fontname, title=title, colors=graph_colors)

    x_feature = checker.dataset.x_data()[:, node.feature()]
    y_data = checker.dataset.y_data()[get_single_node_samples(node)]

    overall_feature_range = (
        np.min(x_feature), np.max(x_feature))

    categorical_labels = None
    if node.is_categorical_split() or node.feature_name() in checker.dataset.categorical_mapping:
        categorical_labels = checker.dataset.categorical_mapping[node.feature_name(
        )]

    y_range = plot.ax.get_ylim()

    left, right = node.split_samples()
    left = y_data[left]
    right = y_data[right]
    split = node.split()

    plot.add_wedge(split, overall_feature_range, categorical_labels)

    # ax.scatter(X_feature, y_train, s=5, c=colors['scatter_marker'], alpha=colors['scatter_marker_alpha'], lw=.3)
    plot.ax.plot([overall_feature_range[0], split], [np.mean(left), np.mean(left)], '--', color=graph_colors['split_line'],
                 linewidth=1)
    plot.ax.plot([split, split], [*y_range], '--',
                 color=graph_colors['split_line'], linewidth=1)
    plot.ax.plot([split, overall_feature_range[1]], [np.mean(right), np.mean(right)], '--', color=graph_colors['split_line'],
                 linewidth=1)
    return plot


def get_target_satisfaction_for_node_plot(node: ShadowDecTreeNode,
                                          checker: Checker,
                                          constraint: Constraint,
                                          non_applicable_counts: bool = False,
                                          graph_colors=None,
                                          fontname: str = "Arial",
                                          fontsize: int = 9,
                                          title: str = None):
    graph_colors = adjust_colors(graph_colors)

    figsize = (.75, .8)
    y = checker.dataset.y_data()[get_single_node_samples(node)]
    m = np.mean(y)
    x = np.random.normal(size=y.shape, scale=0.5)

    constraint_validation_results = checker.get_constraint_validation_result(
        [constraint], non_applicable_counts=non_applicable_counts)[get_single_node_samples(node)]
    color_idx = constraint_validation_results + 1
    plot = Scatter(x, y, color_idx, figsize)
    plot.draw(colors=[VAL_COLORS[truth_value] for truth_value in TRUTH_LABELS], labels=[
              'not applicable', 'invalid', 'valid'])

    plot.ax.plot([-1, 1], [m, m], '--',
                 color=graph_colors['split_line'], linewidth=1)

    text = f"n={node.nsamples()}"
    plot.set_xlabel(text)
    plot.ax.spines['bottom'].set_visible(False)
    plot.ax.set_xticks([])
    return plot

def visualize_fdt(fdt, node_title, group_by, split, feature_range, categorical_split, save_path):
    #print(f'Start {save_path}')
    plot = fdt.visualize(node_title)
    if group_by == group_by_node_split_feature:
        plot.add_wedge(split, feature_range, categorical_split)
    #print(f'Saving {save_path}')
    plot.save(save_path)
    labels_handles = extract_labels_handles(plot)
    #print(f'Done {save_path}')
    del plot
    return labels_handles

def generate_node_plot(node, prepare_plotting = False, *args, **kwargs):
    if node.isleaf():
        plot = generate_leaf_plot(node, prepare_plotting, *args, **kwargs)
    else:
        plot = generate_internal_plot(node, prepare_plotting, *args, **kwargs)
    if prepare_plotting:
        return plot # Now is a tuple of arguments to visualize fdt
    labels_handles = extract_labels_handles(plot)
    return labels_handles

def generate_internal_plot(node, prepare_plotting, shadow_tree, checker, constraints, non_applicable_counts, depth_range_to_display, coverage, use_node_predictions,label_fontsize, leaf_grouping_function, node_samples, path, pid):
    if len(constraints) == 1 or coverage:
        group_by = group_by_node_split_feature
        node_title = ''
    else:
        group_by = group_by_complete_dataset
        node_title = f'{node.feature_name()}@{node.split()}'
    if use_node_predictions:
        prediction = shadow_tree.get_prediction(node.id)
        if checker.dataset.is_categorical(checker.dataset.target_name):
            class_name_mapping = checker.dataset.class_names
            node_title = node_title + \
                f' Prediction: {class_name_mapping[prediction]}'
        else:
            node_title = node_title + \
                f' Prediction: {myround(prediction)}'
    if shadow_tree.is_classifier():
        fdt = FrequencyDistributionTable(checker, constraints, node_samples[node.id], group_by, all_indices_groups=node_samples, coverage=coverage, non_applicable_counts=non_applicable_counts, node=node, only_cached_results=True)
        
        if prepare_plotting:
            if group_by == group_by_node_split_feature:
                return fdt, node_title, group_by, node.split(), checker.dataset.feature_range(node.feature_name()), checker.dataset.categorical_split(node.feature_name(), node.split()), f"{path}/node{node.id}_{pid}.svg"
            else:
                return fdt, node_title, group_by, None, None, None, f"{path}/node{node.id}_{pid}.svg"
        
        plot = fdt.visualize(node_title)
    else:
        if coverage or len(constraints) > 1:
            fdt = FrequencyDistributionTable(checker, constraints, node_samples[node.id], group_by, all_indices_groups=node_samples, coverage=coverage, non_applicable_counts=non_applicable_counts, node=node, only_cached_results=True)
            
            if prepare_plotting:
                if group_by == group_by_node_split_feature:
                    return fdt, node_title, group_by, node.split(), checker.dataset.feature_range(node.feature_name()), checker.dataset.categorical_split(node.feature_name(), node.split()), f"{path}/node{node.id}_{pid}.svg"
                else:
                    return fdt, node_title, group_by, None, None, None, f"{path}/node{node.id}_{pid}.svg"
            
            plot = fdt.visualize(node_title)
        else:
            plot = get_target_feature_for_node_plot(
                node, checker, constraints[0], fontsize=label_fontsize, non_applicable_counts=non_applicable_counts)
    if group_by == group_by_node_split_feature:
        plot.add_wedge(node.split(), checker.dataset.feature_range(node.feature_name(
        )), checker.dataset.categorical_split(node.feature_name(), node.split()))
    plot.save(f"{path}/node{node.id}_{pid}.svg")
    return plot

def generate_leaf_plot(node, prepare_plotting, shadow_tree, checker, constraints, non_applicable_counts, depth_range_to_display, coverage, use_node_predictions, label_fontsize, leaf_grouping_function, node_samples, path, pid):
    prediction = shadow_tree.get_prediction(node.id)
    if checker.dataset.is_categorical(checker.dataset.target_name):
        class_name_mapping = checker.dataset.class_names
        node_title = f'Prediction: {class_name_mapping[shadow_tree.get_prediction(node.id)]}'
    else:
        node_title = f'Prediction: {myround(prediction)}'
    if not shadow_tree.is_classifier() and len(constraints) == 1 and not coverage:
        plot = get_target_satisfaction_for_node_plot(
            node, checker, constraints[0], fontsize=label_fontsize, non_applicable_counts=non_applicable_counts)
        plot.title(node_title)
    else:
        fdt = FrequencyDistributionTable(checker, constraints, node_samples[node.id], leaf_grouping_function, all_indices_groups=node_samples, coverage=coverage, non_applicable_counts=non_applicable_counts, node=node, only_cached_results=True)
        if prepare_plotting:
            return fdt, node_title, None, None, None, None, f"{path}/leaf{node.id}_{pid}.svg"
        plot = fdt.visualize(node_title)
    plot.save(f"{path}/leaf{node.id}_{pid}.svg")
    return plot

def dtreeviz(model,
             checker: Checker,
             constraints: List[Constraint],
             coverage: bool = False,
             non_applicable_counts=False,
             use_node_predictions=False,
             leaf_grouping_function = group_by_complete_dataset, # good alternative: group_by_gt_class
             tree_index: int = None,
             orientation: Union['TD', 'LR'] = "TD",
             instance_orientation: Union["TD", "LR"] = "LR",
             show_root_edge_labels: bool = True,
             show_node_labels: bool = False,
             show_just_path: bool = False,
             fancy: bool = True,  # No function a non fancy representation would be not interesting
             highlight_path: List[int] = [],
             X: np.ndarray = None,
             max_X_features_LR: int = 10,
             max_X_features_TD: int = 20,
             depth_range_to_display: tuple = None,
             label_fontsize: int = 12,
             fontname: str = "Arial",
             title: str = None,
             title_fontsize: int = 14,
             colors: dict = None,
             scale=1.0,
             visualize_in_parallel=False) \
        -> DTreeVizConv:

    shadow_tree = get_shadow_tree_from_checker(model, checker, tree_index)
    checker.validate(constraints)

    def html_label(label, color, size):
        '''Returns the html representation to show a label in a html table.
        '''
        return f'<font face="Helvetica" color="{color}" point-size="{size}"><i>{label}</i></font>'

    def html_image(html_label, img_path):
        '''Returns the html representation to show the label above the given image.
        '''
        html_label_row = f'<tr><td CELLPADDING="0" CELLSPACING="0">{html_label}</td></tr>'
        return f"""<table border="0" CELLBORDER="0">
            {html_label_row}
            <tr>
                    <td><img src="{img_path}"/></td>
            </tr>
            </table>"""

    def html_node_label(node, color, size):
        return html_label(f"Node {node.id}",color, size)

    def html_legend():
        return html_image('',f'{tmp}/legend_{os.getpid()}.svg')

    def node_name(node: ShadowDecTreeNode) -> str:
        '''Returns the name of the corresponding graphviz node given the node in the tree.
        '''
        return f"node{node.id}"

    def node_stmt(node_name, html_content, highlight:bool, colors):
        if highlight:
            return f'{node_name} [margin="0" shape=box penwidth=".5" color="{colors["highlight"]}" style="dashed" label=<{html_content}>]' 
        else: 
            return f'{node_name} [margin="0" shape=box penwidth="0" color="{colors["text"]}" label=<{html_content}>]'

    def split_node(node):
        '''Returns the graphviz node_stmt for the given split node.
        '''
        if fancy:
            label = html_node_label(node, colors["node_label"], 14) if show_node_labels else ''
            html = html_image(label,f'{tmp}/node{node.id}_{os.getpid()}.svg')
        else:
            if checker.dataset.is_categorical(node.feature_name()):
                html = html_label(node.feature_name(), '#444443', 12)
            else:
                split = myround(node.split()) if not node.is_categorical_split() else node.split()[0]
                html = html_label(f'{node.feature_name()}@{split}', '#444443', 12)
        
        return node_stmt(node_name(node), html, node.id in highlight_path, colors)

    def leaf_node(node):
        html_label = html_node_label(node, colors["node_label"], 14) if show_node_labels else ''        
        html = html_image(html_label,f'{tmp}/leaf{node.id}_{os.getpid()}.svg')
        
        return node_stmt(f'leaf{node.id}', html, node.id in highlight_path, colors)

    def class_legend_gr():
        return f"""
            subgraph cluster_legend {{
                style=invis;
                legend [penwidth="0" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {html_legend()}
                >]
            }}
            """

    def instance_html(path, instance_fontsize: int = 11):
        headers = []
        features_used = [node.feature()
                         for node in path[:-1]]  # don't include leaf
        display_X = X
        display_feature_names = shadow_tree.feature_names
        highlight_feature_indexes = features_used
        if (orientation == 'TD' and len(X) > max_X_features_TD) or \
                (orientation == 'LR' and len(X) > max_X_features_LR):
            # squash all features down to just those used
            display_X = [X[i] for i in features_used] + ['...']
            display_feature_names = [node.feature_name()
                                     for node in path[:-1]] + ['...']
            highlight_feature_indexes = range(0, len(features_used))

        for i, name in enumerate(display_feature_names):
            if i in highlight_feature_indexes:
                color = colors['highlight']
            else:
                color = colors['text']
            headers.append(f'<td cellpadding="1" align="right" bgcolor="white">'
                           f'<font face="Helvetica" color="{color}" point-size="{instance_fontsize}">'
                           f'{name}'
                           '</font>'
                           '</td>')

        values = []
        for i, v in enumerate(display_X):
            if i in highlight_feature_indexes:
                color = colors['highlight']
            else:
                color = colors['text']
            if isinstance(v, int) or isinstance(v, str):
                disp_v = v
            else:
                disp_v = myround(v)
            values.append(f'<td cellpadding="1" align="right" bgcolor="white">'
                          f'<font face="Helvetica" color="{color}" point-size="{instance_fontsize}">{disp_v}</font>'
                          '</td>')

        if instance_orientation == "TD":
            html_output = """<table border="0" cellspacing="0" cellpadding="0">"""
            for header, value in zip(headers, values):
                html_output += f"<tr> {header} {value} </tr>"
            html_output += "</table>"
            return html_output
        else:
            return f"""
                <table border="0" cellspacing="0" cellpadding="0">
                <tr>
                    {''.join(headers)}
                </tr>
                <tr>
                    {''.join(values)}
                </tr>
                </table>
                """

    def instance_gr():
        if X is None:
            return ""
        path = shadow_tree.predict_path(X)
        # print(f"path {[node.feature_name() for node in path]}")
        # print(f"path id {[node.id() for node in path]}")
        # print(f"path prediction {[node.prediction() for node in path]}")

        leaf = f"leaf{path[-1].id}"
        return f"""
            subgraph cluster_instance {{
                style=invis;
                X_y [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {instance_html(path)}
                >]
            }}
            {leaf} -> X_y [dir=back; penwidth="1.2" color="{colors['highlight']}"]
            """

    def instance_validation_node():
        if X is None:
            return ""

        return f"""
            subgraph validation_result_graph {{
                style=invis;
                validation_result_table [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                <table border="0">
                <tr><td CELLPADDING="0" CELLSPACING="0"><font face="Helvetica" color="{colors["node_label"]}" point-size="14"><i>Validation Result</i></font></td></tr>
                <tr>
                    <td><img src="{tmp}/validation_result_{os.getpid()}.svg"/></td>
                </tr>
                </table>
                >]
            }}
            X_y -> validation_result_table [dir=back; penwidth="1.2" color="{colors['highlight']}"]
            """

    def get_internal_nodes():
        if depth_range_to_display is not None or (show_just_path and X is not None):
            _nodes = []
            for _node in shadow_tree.internal:
                if depth_range_to_display is not None:
                    if _node.level not in range(depth_range_to_display[0], depth_range_to_display[1] + 1):
                        continue
                
                if show_just_path and X is not None:
                    if _node.id not in highlight_path:
                        continue
                
                _nodes.append(_node)
            return _nodes
        else:
            return shadow_tree.internal

    def get_leaves():
        if depth_range_to_display is not None or (show_just_path and X is not None):
            _nodes = []
            for _node in shadow_tree.leaves:
                if depth_range_to_display is not None:
                    if _node.level not in range(depth_range_to_display[0], depth_range_to_display[1] + 1):
                        continue
                
                if show_just_path and X is not None:
                    if _node.id not in highlight_path:
                        continue
                
                _nodes.append(_node)
                break
            return _nodes
        else:
            return shadow_tree.leaves

    list_labels_handles=[]
 
    def append_labels_handles(new_labels_handles):
        nonlocal list_labels_handles
        list_labels_handles = list_labels_handles + new_labels_handles
    
    def callback_append_label_handle(label_handle):
        nonlocal list_labels_handles
        list_labels_handles = list_labels_handles + [label_handle.result()]

    # General Setup
    colors = adjust_colors(colors)

    if orientation == "TD":
        ranksep = ".2"
        nodesep = "0.1"
    else:
        if fancy:
            ranksep = ".22"
            nodesep = "0.1"
        else:
            ranksep = ".05"
            nodesep = "0.09"
    
    show_edge_labels = True


    tmp = tempfile.gettempdir()

    # Setting highlight_path given X and calculating the Validation Result Plot specific to X.
    if X is not None:
        path = shadow_tree.predict_path(X)
        highlight_path = [n.id for n in path]

        # Generate Plot for Validation Node
        idx = list(
            np.where((checker.dataset.x_data() == np.array(X)).all(axis=-1))[0])

        fdt = FrequencyDistributionTable(checker, constraints, idx, leaf_grouping_function,
                                         coverage=coverage, non_applicable_counts=non_applicable_counts, only_cached_results=True)
        plot = fdt.visualize('')
        plot.save(f"{tmp}/validation_result_{os.getpid()}.svg")
        append_labels_handles([extract_labels_handles(plot)])

    node_samples = get_node_samples(shadow_tree)

    # Collect nodes to plot
    internal_nodes = get_internal_nodes()
    leaf_nodes = get_leaves()
    nodes_to_plot = internal_nodes + leaf_nodes if fancy else leaf_nodes
    CHUNKSIZE = 1
    tasks_left = []
    futures = []

    if use_node_predictions:
        checker_per_prediction = {}
        for y in np.unique(checker.dataset.y_data()):
            checker_per_prediction[y] = ConstantModelChecker(y, checker.dataset, use_gt=checker._use_gt)
            checker_per_prediction[y].validate(constraints)

    print(f'Using {mp.cpu_count() if visualize_in_parallel else 1} worker(s)!')
    with ProcessPool(max_workers=mp.cpu_count(),context=mp.get_context('spawn'), initializer=process_stats_initializer, initargs=get_process_stats_initalizer_args()) as pool:
        for i, node in enumerate(tqdm(nodes_to_plot)):
            if use_node_predictions:
                handles_or_args = generate_node_plot(node, visualize_in_parallel, shadow_tree, checker_per_prediction[node.shadow_tree.get_prediction(node.id)], constraints, non_applicable_counts, depth_range_to_display, coverage, use_node_predictions, label_fontsize, leaf_grouping_function, node_samples, tmp, os.getpid())
            else:
                handles_or_args = generate_node_plot(node, visualize_in_parallel, shadow_tree, checker, constraints, non_applicable_counts, depth_range_to_display, coverage, use_node_predictions, label_fontsize, leaf_grouping_function, node_samples, tmp, os.getpid())
            tasks_left.append(handles_or_args)
            if visualize_in_parallel and (i % CHUNKSIZE == 0 or i == len(nodes_to_plot)-1):
                while len(tasks_left) != 0:
                    task = tasks_left.pop()
                    future = pool.schedule(visualize_fdt, task)
                    future.add_done_callback(callback_append_label_handle)
                    futures.append(future)
            else:
                if i == len(nodes_to_plot) -1:
                    append_labels_handles(tasks_left)

    internal = []
    for node in internal_nodes:
        internal.append(split_node(node))

    leaves = []
    for node in leaf_nodes:
        leaves.append(leaf_node(node))

    # Draw the overall legend
    plot = new_draw_legend(list_labels_handles)
    plot.save(f"{tmp}/legend_{os.getpid()}.svg")

    all_llabel = '&le;' if show_edge_labels else ''
    all_rlabel = '&gt;' if show_edge_labels else ''
    root_llabel = shadow_tree.get_root_edge_labels(
    )[0] if show_root_edge_labels else ''
    root_rlabel = shadow_tree.get_root_edge_labels(
    )[1] if show_root_edge_labels else ''

    edges = []
    # non leaf edges with > and <=
    for node in internal_nodes:
        if depth_range_to_display is not None:
            if node.level not in range(depth_range_to_display[0], depth_range_to_display[1]):
                continue
        nname = node_name(node)
        if node.left.isleaf():
            left_node_name = 'leaf%d' % node.left.id
        else:
            left_node_name = node_name(node.left)
        if node.right.isleaf():
            right_node_name = 'leaf%d' % node.right.id
        else:
            right_node_name = node_name(node.right)

        if not fancy or not (len(constraints) == 1 or coverage) or show_just_path:
            try:
                llabel, rlabel = checker.dataset.categorical_split(
                    node.feature_name(), node.split())
            except:
                if node == shadow_tree.root:
                    llabel = root_llabel
                    rlabel = root_rlabel
                else:
                    llabel = all_llabel
                    rlabel = all_rlabel
        else:
            if node == shadow_tree.root:
                llabel = root_llabel
                rlabel = root_rlabel
            else:
                llabel = all_llabel
                rlabel = all_rlabel

        if node.is_categorical_split() and not shadow_tree.is_classifier():
            lcolor, rcolor = colors["categorical_split_left"], colors["categorical_split_right"]
        else:
            lcolor = rcolor = colors['arrow']

        lpw = rpw = "0.3"
        if node.left.id in highlight_path:
            lcolor = colors['highlight']
            lpw = "1.2"
        if node.right.id in highlight_path:
            rcolor = colors['highlight']
            rpw = "1.2"

        def split_label(
            label): return f'<font face="Arial" color="{colors["arrow"]}" point-size="{11}">&nbsp;{label}</font>' if label != '' else ''

        if show_just_path:
            if node.left.id in highlight_path:
                edges.append(
                    f'{nname} -> {left_node_name} [penwidth={lpw} color="{lcolor}" label=<{split_label(llabel)}>]')
            if node.right.id in highlight_path:
                edges.append(
                    f'{nname} -> {right_node_name} [penwidth={rpw} color="{rcolor}" label=<{split_label(rlabel)}>]')
        else:
            edges.append(
                f'{nname} -> {left_node_name} [penwidth={lpw} color="{lcolor}" label=<{split_label(llabel)}>]')
            edges.append(
                f'{nname} -> {right_node_name} [penwidth={rpw} color="{rcolor}" label=<{split_label(rlabel)}>]')
            edges.append(f"""
            {{
                rank=same;
                {left_node_name} -> {right_node_name} [style=invis]
            }}
            """)

    newline = "\n\t"
    if title:
        title_element = f'graph [label="{title}", labelloc=t, fontname="{fontname}" fontsize={title_fontsize} fontcolor="{colors["title"]}"];'
    else:
        title_element = ""
    dot = f"""
digraph G {{
    splines=line;
    nodesep={nodesep};
    ranksep={ranksep};
    rankdir={orientation};
    margin=0.0;
    {title_element}
    node [margin="0.03" penwidth="0.5" width=.1, height=.1];
    edge [arrowsize=.4 penwidth="0.3"]

    {newline.join(internal)}
    {newline.join(edges)}
    {newline.join(leaves)}

    {class_legend_gr()}
    {instance_gr()}
    {instance_validation_node()}
}}
    """

    return DTreeVizConv(dot, scale)
