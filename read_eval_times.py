from json import load
from matplotlib.cbook import Stack
import pandas as pd
from matplotlib import pyplot as plt
from validating_models.drawing_utils import GroupedStackedHistogram, PieChart, Heatmap
import numpy as np
from validating_models.colors import get_cmap
from functools import partial

def load_data(input_path, join_is_separate = False, eval_is_separate = False, random_shacl_results_is_separate = False):
    times = pd.read_csv(input_path)
    time_names = list(times.columns) + ['other']

    hyperparameters = pd.read_csv(input_path + '_hps.csv')
    hp_names = list(hyperparameters.columns)

    data = pd.concat((hyperparameters, times), axis = 1)

    overall_categories = ['feature range', 'io', 'fdt', 'viz_hist', 'viz_pie', 'node_samples', 'picture composition']

    if join_is_separate:
        overall_categories = overall_categories + ['join']
    
    if eval_is_separate:
        overall_categories = overall_categories + ['eval']
    
    if random_shacl_results_is_separate:
        overall_categories = overall_categories + ['random shacl results']

    data['other'] = data['overall']

    for categorie in overall_categories:
        if categorie in data.columns:
            data['other'] = data['other'] - data[categorie]
    
    overall_categories = overall_categories + ['other']

    experiments = data.groupby(hp_names)
    mean = experiments.mean()
    std = experiments.std()

    std.rename(columns={time_name: f'{time_name}_std' for time_name in time_names}, inplace=True)

    mean = mean.reset_index()
    std = std.reset_index()
    #print(std.head())
    data = pd.concat([mean[time_names],std], axis=1)
    return data, time_names, hp_names, overall_categories

# Parallel vs Serial Execution Times
# data, _, _, categories = load_data('evaluation results/parallel_vs_serial_times_cx31.csv', eval_is_separate=True)
# defaults = {'max_depth': 5, 'n_samples': 4**10, 'n_nodes':4**10, 'n_constraints': 5}

# for experiment in defaults:
#     locked_hps = {key: value for key, value in defaults.items() if key != experiment}
#     filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
#     filtered_data = filtered_data.drop(columns = list(locked_hps))
    
#     # StackedHistogram for serial execution times
#     serial_filtered_data = filtered_data.loc[filtered_data['visualize_in_parallel'] == False]
#     serial_filtered_data = serial_filtered_data.sort_values(by=[experiment])
#     #print(serial_filtered_data['other'])
#     data_to_draw = serial_filtered_data[categories].values.T
#     bar_labels = serial_filtered_data[experiment].values
#     #print(experiment)
#     #print(data_to_draw.shape)
#     from validating_models.drawing_utils import StackedHistogram
#     plot = StackedHistogram(data_to_draw, figure_size=(6,5))
#     plot.draw(bar_labels=bar_labels, bar_labels_title=experiment, categorical_labels=categories, categorical_colors=[get_cmap(len(categories) +1,'hsv')(i) for i in range(len(categories)) ])

#     coordinates_range = [-0.5, len(bar_labels) - 1 + 0.5]
#     plot.ax.set_xlim(coordinates_range)

#     # Line Plot for serial execution time
#     x_data = bar_labels
#     x_range = [np.min(x_data), np.max(x_data)]
#     transform = partial(plot.transform, range=x_range)
#     new_x_data = np.vectorize(transform)(x_data)
#     y_data = serial_filtered_data['overall'].values
#     plot.ax.plot(new_x_data, y_data,'bo' ,label='serial execution time')

#     # Line Plot for parallel execution time    
#     parallel_filtered_data = filtered_data.loc[filtered_data['visualize_in_parallel'] == True]
#     parallel_filtered_data = parallel_filtered_data.sort_values(by=[experiment])
#     x_data = parallel_filtered_data[experiment].values# - 1
#     x_range = [np.min(x_data), np.max(x_data)]



#     transform = partial(plot.transform, range=x_range)
#     new_x_data = np.vectorize(transform)(x_data)

#     y_data = parallel_filtered_data['overall'].values
#     plot.ax.plot(new_x_data, y_data, 'ro', label='parallel execution time')
    
#     plot.draw_legend()

#     plot.save(f'experiment_{experiment}.png', transparent=False)  




# Join Ordered by Cardinality vs Not ordered by Cardinality over n_samples
data, _,_,_ = load_data('join_strategie_times.csv')
fig, ax = plt.subplots(figsize=(5,5))
print(data.columns)
data_join_sorted = data[data['order_by_cardinality'] == True]
data_join_unsorted = data[data['order_by_cardinality'] == False]
for i,data in enumerate([data_join_sorted, data_join_unsorted]):
    x = data['n_samples']
    y = data['join']
    y_err = data['join_std']
    ax.errorbar(x, y, yerr=y_err, label=f'{i}')
plt.legend()
plt.show()




# Join Results first vs. direct join with mapping
# data, _, _,_ = load_data('evaluation results/join_strategie_times.csv')
# defaults = {'n_samples': 4**10, 'n_nodes':4**10, 'n_constraints': 5}

# fig, ax = plt.subplots(1,3, sharey='all', figsize=(15,5))

# y_max = 0

# for i,experiment in enumerate(defaults):
#     #fig, ax = plt.subplots(figsize=(6,5))
#     locked_hps = {key: value for key, value in defaults.items() if key != experiment}
#     filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
#     filtered_data = filtered_data.drop(columns = list(locked_hps))

#     # Exclude the default value as this is the mean taken over n_constraints
#     if experiment != 'n_constraints':
#         filtered_data = filtered_data.loc[filtered_data[experiment] != defaults[experiment]]

#     data_use_outer_join = filtered_data.loc[filtered_data['use_outer_join'] == True]
#     data_use_outer_join = data_use_outer_join.sort_values(by=[experiment])
#     x_data = data_use_outer_join[experiment]
#     y_data = data_use_outer_join['join']
#     y_err = data_use_outer_join['join_std']
#     ax[i].errorbar(x_data, y_data, yerr=y_err, label='join shacl results first')

#     data_no_outer_join = filtered_data.loc[filtered_data['use_outer_join'] == False]
#     data_no_outer_join = data_no_outer_join.sort_values(by=[experiment])

#     x_data = data_no_outer_join[experiment]
#     y_data = data_no_outer_join['join']
#     y_err = data_no_outer_join['join_std']
#     ax[i].errorbar(x_data, y_data, yerr=y_err, label='directly join with samples-to-node-mapping')
#     ax[i].set_ylabel('time [s]')
#     ax[i].set_xlabel(experiment.replace('n_','#'))

# min_y, max_y = plt.ylim()

# for i,experiment in enumerate(defaults):
#     ax[i].vlines(defaults[experiment],min_y,max_y,'r', alpha=0.3)

# plt.title(f'')
# plt.legend()
# plt.savefig(f'join_exp')
# plt.close()

# NODE Samples over n_samples
# data, _, _,_ = load_data('evaluation results/samples_to_node_times.csv')

# fig, ax = plt.subplots(figsize=(6,5))
# for node_to_samples_non_optimized in data['node_to_samples_non_optimized'].unique():
#     for node_to_samples_dont_convert_to_csc in data['node_to_samples_dont_convert_to_csc'].unique():
#         if node_to_samples_non_optimized and not node_to_samples_dont_convert_to_csc:
#             continue
#         elif not node_to_samples_non_optimized and node_to_samples_dont_convert_to_csc:
#             continue
#         elif node_to_samples_non_optimized:
#            max_depths = [2]
#            label = 'dtreeviz (depth independent)'
#         else:
#             max_depths = data['max_depth'].unique()
#             label = None

#         for max_depth in max_depths:
#             sel_filter = (data['node_to_samples_non_optimized'] == node_to_samples_non_optimized) & (data['node_to_samples_dont_convert_to_csc'] == node_to_samples_dont_convert_to_csc) & (data['max_depth'] == max_depth) 
#             mean_sel = data[sel_filter]
#             x = mean_sel['n_samples'].values
#             y = mean_sel['node_samples'].values
#             y_err = data[sel_filter]['node_samples_std'].values

#             idx = np.argsort(x)
#             x = x[idx]
#             y = y[idx]
#             y_err = y_err[idx]

#             ax.errorbar(x, y,fmt='--o', yerr=y_err, label=f'max depth = {max_depth}' if not label else label)
# ax.set_title(f'{"optimized" if not node_to_samples_non_optimized else ""} {"converted to csc matrix " if not node_to_samples_dont_convert_to_csc else ""}')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel('node to samples mapping creation time [s]')
# ax.set_xlabel('#samples')
# plt.legend()
# plt.savefig('node_samples_exp.png')
# plt.close()

# # Stacked Histogram and Line Charts
# from validating_models.drawing_utils import GroupedStackedHistogram
# file_name = 'times_no_csc_serial'
# #file_name = 'times_csc_parallel'
# data, time_names, hp_names = load_data(f'{file_name}.csv', eval_join_in_fdt = True)
# print(data.columns)
# data = data.loc[(data['n_samples'] != 16777216)]

# defaults = {'max_depth': 4, 'n_constraints': 5, 'n_samples': 65536, 'n_nodes':1048576}
# categories = ['io','join','eval','random shacl results','group_by_node_split_feature', 'viz_hist', 'viz_pie', 'node_samples', 'other']

# for group_hp in defaults.keys():
#     for bar_hp in defaults.keys():
#         if bar_hp == group_hp:
#             continue
#         print(f'Starting {group_hp}-{bar_hp}')
#         locked_hps = {key: value for key,value in defaults.items() if key != group_hp and key != bar_hp}

        
#         filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]

#         filtered_data = filtered_data.drop(columns = list(locked_hps))

#         filtered_data = filtered_data.sort_values(by=[group_hp,bar_hp])

#         bar_labels = sorted(filtered_data[bar_hp].unique())
#         group_labels = sorted(filtered_data[group_hp].unique())

#         data_to_plot = []
#         for category in categories:
#             matrix = filtered_data.loc[:,category].values.reshape((len(group_labels), len(bar_labels)))
#             data_to_plot.append(matrix)

#         # Drawing GroupedStackedHistogram
#         width = GroupedStackedHistogram.get_width(len(group_labels),len(bar_labels))
#         plot = GroupedStackedHistogram(data_to_plot,(width, 5))
#         plot.draw(bar_labels=bar_labels,bar_labels_title=bar_hp,group_labels=group_labels, group_labels_title=group_hp, categorical_labels=categories, categorical_colors=[get_cmap(len(categories) +1,'hsv')(i) for i in range(len(categories)) ])
#         plot.draw_legend()
#         import os
#         path = os.path.join('stacked_histogram_plots',file_name)
#         os.makedirs(path,exist_ok=True)
#         print(f'Saving {group_hp}-{bar_hp}')
#         plot.save(os.path.join(path,f'{group_hp}-{bar_hp}.png'), transparent=False)

#         # Drawing LineCharts
#         line_path = os.path.join(path,f'{group_hp}-{bar_hp}')
#         os.makedirs(line_path, exist_ok=True)

#         for category in categories:
#             fig, ax = plt.subplots(figsize=(6,5))
#             for i,group_label in enumerate(group_labels):
#                 data_to_plot = filtered_data[filtered_data[group_hp] == group_label]
#                 x_data = data_to_plot.loc[:,bar_hp].values
#                 y_data = data_to_plot.loc[:,category].values
#                 y_err = data_to_plot.loc[:,category + '_std']
#                 ax.errorbar(x_data,y_data,yerr=y_err, label=group_label)
#             ax.set_ylabel(category)
#             ax.set_xlabel(bar_hp)
#             if bar_hp in ['n_nodes','n_samples']:
#                 ax.set_xscale('log')
#                 ax.set_yscale('log')
#             plt.legend(title=group_hp)
#             plt.savefig(os.path.join(line_path,f'{category}.png'))
#             plt.close()


# # JOIN - TIME over n_constraints * n_samples --> join_time \in O(n_constraints * n_samples)
# data, time_names, hp_names,_ = load_data('evaluation results/times_no_csc_serial.csv')
# fig, ax = plt.subplots(figsize=(6,5))

# for max_depth in data['max_depth'].unique():
#     data_sel = data[data['max_depth'] == max_depth]
#     x = data_sel['n_constraints'].values * data_sel['n_samples'].values# * np.log(mean_sel['n_nodes']).values
#     y = data_sel['join'].values
#     y_err = data[data['max_depth'] == max_depth]['join_std'].values
    
#     idx = np.argsort(x)
#     x = x[idx]
#     y = y[idx]
#     y_err = y_err[idx]

#     ax.errorbar(x, y, yerr=y_err, label=f'max depth = {max_depth}')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('accumulated join time [s]')
# plt.xlabel('#constraints * #samples * log(#seedNodes)')
# plt.legend()
# plt.show()
# plt.close()

# # EVAL - TIME over n_constraints * n_samples
# data, time_names, hp_names,_ = load_data('evaluation results/times_no_csc_serial.csv')

# fig, ax = plt.subplots(figsize=(6,5))

# for max_depth in data['max_depth'].unique():
#     for n_nodes in data['n_nodes'].unique():
#         mean_sel = data[(data['max_depth'] == max_depth) & (data['n_nodes'] == n_nodes)]
#         x = mean_sel['n_constraints'].values * mean_sel['n_samples'].values
#         y = mean_sel['eval'].values
#         y_err = data[data['max_depth'] == max_depth]['join_std'].values
        
#         idx = np.argsort(x)
#         x = x[idx]
#         y = y[idx]
#         y_err = y_err[idx]

#         ax.errorbar(x, y, yerr=y_err, label=f'max depth = {max_depth}, #SeedNodes = {n_nodes}')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('accumulated constraint evaluation time [s]')
# plt.xlabel('#constraints * #samples')
# plt.legend()
# plt.show()
# plt.close()