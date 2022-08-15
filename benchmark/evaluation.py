from argparse import ArgumentParser
import pandas as pd
from matplotlib import pyplot as plt
from validating_models.drawing_utils import GroupedStackedHistogram, PieChart, Heatmap
import numpy as np
from validating_models.colors import get_cmap

def load_data(input_path, join_is_separate = False, eval_is_separate = False, random_shacl_results_is_separate = False, subtract_overall_constraint_evaluation = False, remove_node_samples = False):
    times = pd.read_csv(input_path)
    time_names = list(times.columns)

    hyperparameters = pd.read_csv(input_path + '_hps.csv')
    hp_names = list(hyperparameters.columns)

    data = pd.concat((hyperparameters, times), axis = 1)
    if 'overall' in data.columns:
        overall_categories = ['io', 'summarization', 'viz_hist', 'viz_pie','viz_grouped_hist', 'overall_constraint_evaluation', 'node_samples', 'picture_composition']

        if join_is_separate:
            overall_categories = overall_categories + ['join']
        
        if eval_is_separate:
            overall_categories = overall_categories + ['eval']
        
        if remove_node_samples:
            overall_categories.remove('node_samples')
        
        if subtract_overall_constraint_evaluation:
            overall_categories.remove('overall_constraint_evaluation')
            data['overall'] = data['overall'] - data['overall_constraint_evaluation']
            data.drop(['overall_constraint_evaluation', 'join', 'shacl_schema_validation','evaluation', 'model_inference'], axis=1)

        data['other'] = data['overall']
        time_names = time_names + ['other']
        for categorie in overall_categories:
            if categorie in data.columns:
                data['other'] = data['other'] - data[categorie]
        
        overall_categories = overall_categories + ['other']
    else:
        overall_categories = []
    experiments = data.groupby(hp_names)
    mean = experiments.mean()
    std = experiments.std()

    std.rename(columns={time_name: f'{time_name}_std' for time_name in time_names}, inplace=True)

    mean = mean.reset_index()
    std = std.reset_index()
    #print(std.head())
    data = pd.concat([mean[time_names],std], axis=1)
    return data, time_names, hp_names, overall_categories


def main():
    parser = ArgumentParser(description='Transforms data collected from speed_test.py into visualizations')
    parser.add_argument('type', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    if args.type == 'dtreevizComparison':
        data_dtree, _,_,_ = load_data('dtreeviz_times.csv')
        data_viz, _,_,_ = load_data('visualization_time.csv', subtract_overall_constraint_evaluation = True)
        defaults = {'max_depth': 5, 'n_samples': 4**10}

        fig, ax = plt.subplots(1,2, figsize=(10,5))

        for i, experiment in enumerate(defaults):
            ax[i].set_xlabel(experiment.replace('n_','#'))
            for fancy in [True, False]:
                for data,label in zip([data_dtree, data_viz],['dtreeviz', 'visualization algorithm']):
                    locked_hps = {key: value for key, value in defaults.items() if key != experiment}
                    filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
                    filtered_data = filtered_data.drop(columns = list(locked_hps))
                    filtered_data = filtered_data[filtered_data['fancy'] == fancy]
                    filtered_data = filtered_data[[experiment, 'overall', 'overall_std']].sort_values(by=[experiment])

                    x_data = filtered_data[experiment].values
                    y_data = filtered_data['overall'].values
                    y_data_std = filtered_data['overall_std'].values
                    ax[i].errorbar(x_data, y_data, yerr=y_data_std, label=f'{label} {"(leaves only)" if not fancy else ""}', fmt='x-')
                    if experiment == 'n_samples':
                        from scipy.stats import linregress
                        print(label, fancy)
                        print(linregress(x_data, y_data))
                    #print(filtered_data.head())

        min_y, max_y = plt.ylim()

        for i,experiment in enumerate(defaults):
            ax[i].vlines(defaults[experiment],min_y,max_y,'r', alpha=0.3)
        plt.legend(loc = 'upper right')
        plt.savefig('dtreevizComparison.png')
        plt.close()
    
    elif args.type == 'interpretME':
        times = pd.read_csv('times_long_seed_query.csv')
        time_names = list(times.columns)
        hyperparameters = pd.read_csv('times_long_seed_query.csv_hps.csv')
        hp_names = list(hyperparameters.columns)
        data = pd.concat((hyperparameters, times), axis = 1)

        times_all_constraints = pd.read_csv('times_short_seed_query.csv')
        time_names_all_constraints = list(times_all_constraints.columns)
        hyperparameters_all_constraints = pd.read_csv('times_short_seed_query.csv_hps.csv')
        hp_names_all_constraints = list(hyperparameters_all_constraints.columns)
        data_all_constraints = pd.concat((hyperparameters_all_constraints, times_all_constraints), axis = 1)

        times_single_constraints = pd.read_csv('times_single_constraint.csv')
        time_names_single_constraints = list(times_single_constraints.columns)
        hyperparameters_single_constraints = pd.read_csv('times_single_constraint.csv_hps.csv')
        hp_names_single_constraints = list(hyperparameters_single_constraints.columns)
        data_single_constraints = pd.concat((hyperparameters_single_constraints, times_single_constraints), axis = 1)

        extraction_times = data.loc[:,['experiment_config', 'PIPE_DATASET_EXTRACTION']].groupby('experiment_config')
        print(extraction_times.mean(), extraction_times.std())

        data['overall_model_training'] = data['PIPE_PREPROCESSING'] + data['PIPE_SAMPLING'] + data['PIPE_IMPORTANT_FEATURES'] + data['PIPE_TRAIN_MODEL']
        
        training_times = data.loc[:,['experiment_config', 'overall_model_training']].groupby('experiment_config')
        print(training_times.mean(), training_times.std())

        dtreeviz_time = data.loc[:,['experiment_config', 'PIPE_DTREEVIZ']].groupby('experiment_config')
        print(dtreeviz_time.mean(), dtreeviz_time.std())

        decision_tree_nodes = data.loc[:,['experiment_config', 'n_dnodes']].groupby('experiment_config')
        print(decision_tree_nodes.mean(), decision_tree_nodes.std())

        n_samples = data.loc[:,['experiment_config', 'n_samples']].groupby('experiment_config')
        print(n_samples.mean(), n_samples.std())
        n_nodes = data.loc[:,['experiment_config', 'n_nodes']].groupby('experiment_config')
        print(n_nodes.mean(), n_nodes.std())

        n_constraints = data.loc[:,['experiment_config', 'n_constraints']].groupby('experiment_config')
        print(n_constraints.mean(), n_constraints.std())

        data.drop(columns = ['PIPE_PREPROCESSING', 'PIPE_SAMPLING', 'PIPE_IMPORTANT_FEATURES', 'PIPE_TRAIN_MODEL', 'PIPE_DATASET_EXTRACTION', 'PIPE_LIME','PIPE_DTREEVIZ', 'n_dnodes', 'n_samples', 'n_constraints'], inplace=True)

        fig, ax = plt.subplots(1,2, figsize = (5,5))

        viz_times = data_all_constraints.loc[:,['experiment_config', 'summarization','io', 'viz_hist', 'viz_pie', 'picture_composition']]
        avg_viz_times = viz_times.mean()
        row = avg_viz_times
        plot = PieChart(row.values, figure_size=(3,3), ax = ax[0])
        plot.draw(colors = [get_cmap(len(row) +1,'hsv')(i) for i in range(len(row)) ], labels = [rename_category_viz(s) for s in list(row.index)], text='all constraints')

        viz_times = data_single_constraints.loc[:,['experiment_config', 'summarization','io', 'viz_hist', 'viz_pie', 'picture_composition']]
        avg_viz_times = viz_times.mean()
        row = avg_viz_times
        plot = PieChart(row.values, figure_size=(3,3), ax = ax[1])
        plot.draw(colors = [get_cmap(len(row) +1,'hsv')(i) for i in range(len(row)) ], labels = [rename_category_viz(s) for s in list(row.index)], text='single constraint')
        
        plot.draw_legend()
        plt.savefig(f'overall_viz_distribution_separate.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, ax = plt.subplots(2,2, figsize=(10,5))

        for k, group_labels in enumerate([['The Lung Cancer KG'],['The French Royalty KG']]):
            # SHACL
            data,_,_,_ = load_data('times_long_seed_query.csv')
            data_short,_,_,_ = load_data('times_short_seed_query.csv')
            
            def filter_data(data):
                filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False) & (data['run_in_parallel'] == False)].loc[:,['shacl_schema_validation','shacl_schema_validation_std', 'experiment_config', 'api_config']]
                filtered_data['experiment_config'] = filtered_data['experiment_config'].apply(lambda s: s.replace('example/example_clarify.json', 'The Lung Cancer KG').replace('example/example_french_royalty.json','The French Royalty KG'))
                filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
                return filtered_data
            
            filtered_data = filter_data(data)
            filtered_data_short = filter_data(data_short)
            
            configs = ['all_heuristics', 'no_heuristics', 'only_prune_shape_network', 'only_replace_target_query']
            data_to_plot = []
            data_to_plot_std = []
            matrix_overhead = np.zeros((len(group_labels), len(configs)))
            matrix_overhead_std = np.zeros((len(group_labels), len(configs)))
            for i, config in enumerate(configs):
                matrix = np.zeros((len(group_labels), len(configs)))
                matrix_std = np.zeros((len(group_labels), len(configs)))
                for j,group_label in enumerate(group_labels):
                    result = filtered_data_short[(filtered_data_short['api_config'] == config) & (filtered_data_short['experiment_config'] == group_label)]
                    result_long = filtered_data[(filtered_data['api_config'] == config) & (filtered_data['experiment_config'] == group_label)]
                    print('SHACL', group_label, result.loc[:,'shacl_schema_validation'])
                    overhead = (result_long.loc[:,'shacl_schema_validation'].values[0] - result.loc[:,'shacl_schema_validation'].values[0])
                    overhead = overhead if overhead > 0.0 else 0.0
                    matrix[j,i] = result.loc[:,'shacl_schema_validation'].values
                    matrix_std[j,i] = result.loc[:,'shacl_schema_validation_std'].values
                    matrix_overhead[j,i] = overhead
                    matrix_overhead_std[j,i] = result_long.loc[:,'shacl_schema_validation_std'].values
                    print(matrix_std[j,i])
                data_to_plot.append(matrix)
                data_to_plot_std.append(matrix_std)
            data_to_plot.append(matrix_overhead)
            data_to_plot_std.append(matrix_overhead_std)

            # Drawing GroupedStackedHistogram
            width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
            plot = GroupedStackedHistogram(data_to_plot,(width, 3), data_std=data_to_plot_std, ax=ax[0][k])
            plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='horizontal', group_labels_title="", categorical_labels=[rename_configs_single(config) for config in configs] + ['overhead expensive seed query'], categorical_colors=[get_cmap(len(configs) +2,'hsv')(i) for i in range(len(configs) + 1) ])
            if k == 1:
                #pass
                plot.draw_legend(inside=False)
            #plot.save(f'shacl_{group_labels[0]}.png')


            # Join
            data,_,_,_ = load_data('times_long_seed_query.csv')
            
            filtered_data = data[(data['api_config'] == 'example/speed_test_shacl_api_configs/all_heuristics.json') & (data['run_in_parallel'] == False)].loc[:,['join','join_std', 'experiment_config', 'use_outer_join', 'optimize_intermediate_results']]
            filtered_data['experiment_config'] = filtered_data['experiment_config'].apply(lambda s: s.replace('example/example_clarify.json', 'The Lung Cancer KG').replace('example/example_french_royalty.json','The French Royalty KG'))
            #print(filtered_data)
            
            configs = ['Join T at the end', 'Join T at the end + Optimizations', 'Join T at the beginning']
            setups = [(True, False),(True, True),(False,False)]
            
            data_to_plot = []
            data_to_plot_std = []

            for i, setup in enumerate(setups):
                matrix = np.zeros((len(group_labels), len(configs)))
                matrix_std = np.zeros((len(group_labels), len(configs)))
                for j,group_label in enumerate(group_labels):
                    result = filtered_data[(filtered_data['use_outer_join'] == setup[0]) & (filtered_data['optimize_intermediate_results'] == setup[1]) & (filtered_data['experiment_config'] == group_label)]
                    matrix[j,i] = result.loc[:,'join'].values
                    print('JOIN', group_label, matrix[j,i])
                    matrix_std[j,i] = result.loc[:,'join_std'].values
                    #print(configs[i], group_label, result.loc[:,['join','join_std']].values)
                data_to_plot.append(matrix)
                data_to_plot_std.append(matrix_std)

            # Drawing GroupedStackedHistogram
            width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
            plot = GroupedStackedHistogram(data_to_plot,(width, 3), data_std=data_to_plot_std, ax=ax[1][k])
            plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=[rename_configs_single(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
            if k == 1:
                pass
                #plot.draw_legend(inside=False)
            #plot.save(f'join_{group_labels[0]}.png')

        plt.savefig(f'join_shacl.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # LIME
        data,_,_,_ = load_data('times_lime.csv')
        print(data.loc[:,['PIPE_LIME', 'PIPE_LIME_std']])

        data,_,_,_ = load_data('times_single_constraint.csv')
        grouped = data.groupby('experiment_config').mean()
        print(grouped.loc[:,'PIPE_SINGLE_CONSTRAINT_VIZ']/grouped.loc[:,'n_constraints'])
        print(grouped.loc[:,'PIPE_SINGLE_CONSTRAINT_VIZ_std']/grouped.loc[:,'n_constraints'])

        data,_,_,_ = load_data('times_short_seed_query.csv')
        grouped = data.groupby('experiment_config').mean()
        print(grouped.loc[:,'PIPE_MULTI_CONSTRAINT_VIZ'])
        print(grouped.loc[:,'PIPE_MULTI_CONSTRAINT_VIZ_std'])
        print(grouped.loc[:,'PIPE_MULTI_CONSTRAINT_VIZ']/grouped.loc[:,'n_constraints'])
        print(grouped.loc[:,'PIPE_MULTI_CONSTRAINT_VIZ_std']/grouped.loc[:,'n_constraints'])

        data,_,_,_ = load_data('times_long_seed_query.csv')
        grouped = data.groupby(['experiment_config','run_in_parallel']).mean()
        print('Single Constraint Parallel', grouped.loc[:,['PIPE_SINGLE_CONSTRAINT_VIZ','PIPE_SINGLE_CONSTRAINT_VIZ_std']])
        print('All Constraints Parallel', grouped.loc[:,['PIPE_MULTI_CONSTRAINT_VIZ','PIPE_MULTI_CONSTRAINT_VIZ_std']])

        # targets_per_experiment = data.loc[:,['experiment_config', 'PIPE_SINGLE_CONSTRAINT_VIZ', 'PIPE_MULTI_CONSTRAINT_VIZ', 'PIPE_CONSTRAINT_VIZ']].groupby('experiment_config')
        # targets_per_experiment.mean().to_csv('targets.csv')

        # targets_per_experiment = data.loc[:,hp_names + ['join', 'shacl_schema_validation', 'number_of_targets']].groupby(hp_names)
        
        # targets_per_experiment.mean().to_csv('targets2.csv')


    elif args.type == 'nodesamples':
# NODE Samples over n_samples
        data, _, _,_ = load_data('benchmark/results/samples_to_node_times.csv')

        fig, ax = plt.subplots(figsize=(6,5))
        for node_to_samples_non_optimized in data['node_to_samples_non_optimized'].unique():
            for node_to_samples_dont_convert_to_csc in data['node_to_samples_dont_convert_to_csc'].unique():
                if node_to_samples_non_optimized and not node_to_samples_dont_convert_to_csc:
                    continue
                elif not node_to_samples_non_optimized and node_to_samples_dont_convert_to_csc:
                    continue
                elif node_to_samples_non_optimized:
                    max_depths = [2]
                    label = 'dtreeviz (depth independent)'
                else:
                    max_depths = data['max_depth'].unique()
                    label = None

                for max_depth in max_depths:
                    sel_filter = (data['node_to_samples_non_optimized'] == node_to_samples_non_optimized) & (data['node_to_samples_dont_convert_to_csc'] == node_to_samples_dont_convert_to_csc) & (data['max_depth'] == max_depth) 
                    mean_sel = data[sel_filter]
                    x = mean_sel['n_samples'].values
                    y = mean_sel['node_samples'].values
                    y_err = data[sel_filter]['node_samples_std'].values

                    idx = np.argsort(x)
                    x = x[idx]
                    y = y[idx]
                    y_err = y_err[idx]

                    print(max_depth, y)

                    ax.errorbar(x, y,fmt='--o', yerr=y_err, label=f'max depth = {max_depth}' if not label else label)
        ax.set_title(f'{"optimized" if not node_to_samples_non_optimized else ""} {"converted to csc matrix " if not node_to_samples_dont_convert_to_csc else ""}')
        ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_xlabel('#samples')
        plt.legend()
        plt.savefig('node_samples_exp.png')
        plt.close()

    elif args.type == 'treevizParallelSerial':
        #data, _, _, categories = load_data('evaluation results/parallel_vs_serial_times_cx31.csv', eval_is_separate=True)
        data, _, _, categories = load_data(args.input, subtract_overall_constraint_evaluation = True, remove_node_samples = True)
        data.to_csv('benchmark/results/output.csv', index=False)
        defaults = {'max_depth': 5, 'n_constraints': 5, 'n_nodes':4**10, 'n_samples': 4**10}
        readable_categories = [rename_category_viz(category) for category in categories]

        fig, ax = plt.subplots(2,2, figsize=(10,10))

        for i, experiment in enumerate(defaults):
            locked_hps = {key: value for key, value in defaults.items() if key != experiment}
            filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
            filtered_data = filtered_data.drop(columns = list(locked_hps))
            
            # StackedHistogram for serial execution times
            serial_filtered_data = filtered_data.loc[filtered_data['visualize_in_parallel'] == False]
            serial_filtered_data = serial_filtered_data.sort_values(by=[experiment])
            data_to_draw = serial_filtered_data[categories].values.T
            data_std_to_draw = serial_filtered_data[[category + '_std' for category in categories]].values.T
            bar_labels = serial_filtered_data[experiment].values

            from validating_models.drawing_utils import StackedHistogram
            plot = StackedHistogram(data_to_draw, figure_size=(6,5), data_std=data_std_to_draw, ax=ax[0 if i < 2 else 1][i % 2])
            plot.draw(bar_labels=bar_labels, bar_labels_title=experiment.replace('n_','#'), categorical_labels=readable_categories, categorical_colors=[get_cmap(len(categories) +1,'hsv')(i) for i in range(len(categories)) ])

            coordinates_range = [-0.5, len(bar_labels) - 1 + 0.5]
            plot.ax.set_xlim(coordinates_range)

            # Line Plot for parallel execution time
            x_data = np.arange(0,len(bar_labels))
            parallel_filtered_data = filtered_data.loc[filtered_data['visualize_in_parallel'] == True]
            parallel_filtered_data = parallel_filtered_data.sort_values(by=[experiment])
            y_data = parallel_filtered_data['overall'].values
            y_data_std = parallel_filtered_data['overall_std'].values
            plot.ax.errorbar(x_data, y_data, yerr=y_data_std, label='parallel execution time', fmt='bo')
            
            min_y, max_y = ax[0 if i < 2 else 1][i % 2].get_ylim()
            
            ax[0 if i < 2 else 1][i % 2].vlines(np.where((defaults[experiment] == bar_labels))[0],min_y, max_y,'r', alpha=0.3)
            
        plot.draw_legend()
        plt.savefig(f'experiment.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close()

    elif args.type == 'join':
        data, _, _,_ = load_data(args.input)
        defaults = {'n_samples': 4**10, 'n_nodes':4**10, 'n_constraints': 5}

        fig, ax = plt.subplots(1,3, sharey='all', figsize=(15,5))

        for i,experiment in enumerate(defaults):
            locked_hps = {key: value for key, value in defaults.items() if key != experiment}
            filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
            filtered_data = filtered_data.drop(columns = list(locked_hps))

            # Exclude the default value as this is the mean taken over n_constraints
            if experiment != 'n_constraints':
                filtered_data = filtered_data.loc[filtered_data[experiment] != defaults[experiment]]

            for use_outer_join, order_by_cardinality,not_pandas_optimized, label in [(False, False,True, 'Join T at the beginning'), (True, True, False, 'Join T at the end + Optimizations'), (True, False, False, 'Join T at the end')]: # (True, True, True, 'Join T at the end + Sort by Cardinality'), (True, False, True, 'Join T at the end')
                data_use_outer_join = filtered_data.loc[(filtered_data['use_outer_join'] == use_outer_join) & (filtered_data['order_by_cardinality'] == order_by_cardinality) & (filtered_data['not_pandas_optimized'] == not_pandas_optimized)]  # use_outer_join,order_by_cardinality --> FF, TT, TF
                data_use_outer_join = data_use_outer_join.sort_values(by=[experiment])
                # if experiment == 'n_nodes':
                #     experiment = 'n_unique_nodes'
                x_data = data_use_outer_join[experiment]
                y_data = data_use_outer_join['join']
                y_err = data_use_outer_join['join_std']
                ax[i].errorbar(x_data, y_data, yerr=y_err, label=label)
                if experiment == 'n_samples':
                    from scipy.stats import linregress
                    print(label)
                    print(linregress(x_data.values, y_data.values))


            ax[i].set_xlabel(experiment.replace('n_','#'))

        min_y, max_y = plt.ylim()

        for i,experiment in enumerate(defaults):
            ax[i].vlines(defaults[experiment],min_y,max_y,'r', alpha=0.3)

        plt.title(f'')
        plt.legend()
        plt.savefig(f'join_exp')
        plt.close()

    elif args.type == 'validation_engine_bars_shacl_times':
        data, _, _,_ = load_data(args.input)
        #data.to_csv('output.csv')
        filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False)].loc[:,['shacl_schema_validation','shacl_schema_validation_std', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        filtered_data.to_csv('output.csv')
        filtered_data_not_separate = filtered_data[filtered_data['constraints_separate'] == False].sort_values(by=['shape_schema_dir','api_config'])
        filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
        configs_not_separate = ['all_heuristics', 'no_heuristics', 'only_prune_shape_network', 'only_replace_target_query']
        configs = configs_not_separate + ['all_heuristics_sep']
        group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
        data_to_plot = []
        data_to_plot_std = []

        for i, config in enumerate(configs_not_separate):
            matrix = np.zeros((len(group_labels), len(configs)))
            matrix_std = np.zeros((len(group_labels), len(configs)))
            for j,group_label in enumerate(group_labels):
                result = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)]
                matrix[j,i] = result.loc[:,'shacl_schema_validation'].values
                matrix_std[j,i] = result.loc[:,'shacl_schema_validation_std'].values
            data_to_plot.append(matrix)
            data_to_plot_std.append(matrix_std)
        
        for i, config in enumerate(['all_heuristics']):
            matrix = np.zeros((len(group_labels), len(configs)))
            matrix_std = np.zeros((len(group_labels), len(configs)))
            i2 = i + len(configs_not_separate)
            for j,group_label in enumerate(group_labels):
                n_constraints = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)].loc[:,'nconstraints'].values
                result = n_constraints * filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)].loc[:,['shacl_schema_validation','shacl_schema_validation_std']].values             
                matrix[j,i2] = result[0][0]
                matrix_std[j,i2] = result[0][1]
            data_to_plot.append(matrix)
            data_to_plot_std.append(matrix_std)

        # Drawing GroupedStackedHistogram
        width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
        plot = GroupedStackedHistogram(data_to_plot,(width, 3), data_std=data_to_plot_std)
        plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='horizontal', group_labels_title="", categorical_labels=[rename_configs_all(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
        plot.draw_legend(inside=True)
        plot.save('ablation_study_shacl_time.png')

    elif args.type == 'validation_engine_bars_targets':
        data, _, _,_ = load_data(args.input)
        #data.to_csv('output.csv')
        filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False)].loc[:,['number_of_targets', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        filtered_data_not_separate = filtered_data[filtered_data['constraints_separate'] == False].sort_values(by=['shape_schema_dir','api_config'])
        filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
        configs_not_separate = ['all_heuristics', 'no_heuristics', 'only_prune_shape_network', 'only_replace_target_query']
        configs = configs_not_separate + ['all_heuristics_sep']
        group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
        data_to_plot = []

        for i, config in enumerate(configs_not_separate):
            matrix = np.zeros((len(group_labels), len(configs)))
            for j,group_label in enumerate(group_labels):
                matrix[j,i] = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)].loc[:,'number_of_targets'].values
            data_to_plot.append(matrix)
        
        for i, config in enumerate(['all_heuristics']):
            matrix = np.zeros((len(group_labels), len(configs)))
            i2 = i + len(configs_not_separate)
            for j,group_label in enumerate(group_labels):
                n_constraints = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)].loc[:,'nconstraints'].values
                result = filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)].loc[:,'number_of_targets'].values
                matrix[j,i2] = n_constraints * result
            data_to_plot.append(matrix)

        # Drawing GroupedStackedHistogram
        width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
        plot = GroupedStackedHistogram(data_to_plot,(width, 3))
        plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='horizontal', group_labels_title="", categorical_labels=[rename_configs_all(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
        plot.draw_legend(inside=True)
        plot.save('ablation_study_targets.png')

    elif args.type == 'validation_engine_bars_shacl_times_single':
        data, _, _,_ = load_data(args.input)
        #data.to_csv('output.csv')
        filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False)].loc[:,['shacl_schema_validation','shacl_schema_validation_std', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
        configs = ['all_heuristics', 'no_heuristics', 'only_prune_shape_network', 'only_replace_target_query']
        group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
        data_to_plot = []
        data_to_plot_std = []
        for i, config in enumerate(configs):
            matrix = np.zeros((len(group_labels), len(configs)))
            matrix_std = np.zeros((len(group_labels), len(configs)))
            for j,group_label in enumerate(group_labels):
                result = filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)]
                matrix[j,i] = result.loc[:,'shacl_schema_validation'].values
                matrix_std[j,i] = result.loc[:,'shacl_schema_validation_std'].values
            data_to_plot.append(matrix)
            data_to_plot_std.append(matrix_std)

        # Drawing GroupedStackedHistogram
        width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
        plot = GroupedStackedHistogram(data_to_plot,(width, 3), data_std=data_to_plot_std)
        plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=[rename_configs_single(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
        plot.draw_legend(inside=True)
        plot.save('ablation_study_shacl_time_single.png')

    elif args.type == 'validation_engine_bars_targets_single':
        data, _, _,_ = load_data(args.input)
        #data.to_csv('output.csv')
        filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False)].loc[:,['number_of_targets', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
        configs = ['all_heuristics', 'no_heuristics', 'only_prune_shape_network', 'only_replace_target_query']
        group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
        data_to_plot = []
        for i, config in enumerate(configs):
            matrix = np.zeros((len(group_labels), len(configs)))
            for j,group_label in enumerate(group_labels):
                matrix[j,i] = filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)].loc[:,'number_of_targets'].values
            data_to_plot.append(matrix)

        # Drawing GroupedStackedHistogram
        width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
        plot = GroupedStackedHistogram(data_to_plot,(width, 3))
        plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=[rename_configs_single(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
        plot.draw_legend(inside=True)
        plot.save('ablation_study_targets_single.png')

    elif args.type == 'validation_engine_factors':
        data, _, _,_ = load_data(args.input)
        filtered_data = data[(data['use_outer_join'] == True) & (data['optimize_intermediate_results'] == False)].loc[:,['shacl_schema_validation', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        filtered_data_not_separate = filtered_data[filtered_data['constraints_separate'] == False].sort_values(by=['shape_schema_dir','api_config'])
        configs = ['no_heuristics', 'only_replace_target_query', 'only_prune_shape_network', 'all_heuristics']

        sums = {}
        for config in configs:
            sums[config] = 0.0
            for ss in ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']:
                base_time = filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == ss)]['shacl_schema_validation'].values
                n_constraints = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == ss)].loc[:,'nconstraints'].values
                sums[config] += base_time * n_constraints
        
        sums_not_sep = {}
        for config in configs:
            sums_not_sep[config] = 0.0
            for ss in ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']:
                base_time = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == ss)]['shacl_schema_validation'].values
                sums_not_sep[config] += base_time
        print(sums)
        print(sums_not_sep)
        # pruning
        print("Prunning Factor:", 1 - ((sums_not_sep['only_prune_shape_network'] + sums['only_prune_shape_network'])/(sums_not_sep['no_heuristics'] + sums['no_heuristics'])))
        print("Target Factor:", 1 - ((sums_not_sep['only_replace_target_query'] + sums['only_replace_target_query'])/(sums_not_sep['no_heuristics'] + sums['no_heuristics'])))
        print("Simultaneous Factor:", 1 - (np.sum(list(sums_not_sep.values()))/np.sum(list(sums.values()))))
        print("Overall:", 1 - (sums_not_sep['all_heuristics']/sums['no_heuristics']))

    elif args.type == "validation_engine_join":
        data, _, _,_ = load_data(args.input)
        print(data.head())
        filtered_data = data.loc[:,['join','join_std', 'shape_schema_dir', 'api_config', 'constraints_separate', 'use_outer_join', 'optimize_intermediate_results']]
        filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d').replace('multiple_shapes_per_class_d', 'd_ms').replace('multiple_shapes_per_class_o', 'o_ms'))
        filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
        #filtered_data = filtered_data[(filtered_data['constraints_separate'] == False) & (filtered_data['api_config'] == 'all_heuristics')].sort_values(by=['shape_schema_dir','api_config'])
        filtered_data = filtered_data.sort_values(by=['shape_schema_dir','api_config'])

        filtered_data.to_csv('output.csv', index=False)
        
        configs = ['Join T at the end', 'Join T at the end + Optimizations', 'Join T at the beginning']
        setups = [(True, False),(True, True),(False,False)]

        group_labels = list(np.unique(filtered_data['shape_schema_dir'].values))#['bt_15_d', 'star_15_d', 'star_5_n', 'bt_15_o','star_26_d_ms']
        #group_labels_large = ['bt_31_n','star_15_n', 'bt_15_n','bt_31_d', 'star_15_d','star_25_d', 'star_26_d_ms']
        #group_labels_small = ['star_25_o', 'bt_15_d','star_5_n','bt_7_n','star_26_o_ms', 'bt_15_o', 'bt_31_o', 'bt_7_d', 'bt_7_o', 'star_15_o', 'star_5_d', 'star_5_o']
        #group_labels = group_labels_small

        data_to_plot = []
        data_to_plot_std = []

        sum = {True: 0.0, False: 0.0}
        for i, setup in enumerate(setups):
            matrix = np.zeros((len(group_labels), len(configs)))
            matrix_std = np.zeros((len(group_labels), len(configs)))
            for j,group_label in enumerate(group_labels):
                result = filtered_data[(filtered_data['use_outer_join'] == setup[0]) & (filtered_data['optimize_intermediate_results'] == setup[1]) & (filtered_data['shape_schema_dir'] == group_label)]
                matrix[j,i] = result.loc[:,'join'].values
                matrix_std[j,i] = result.loc[:,'join_std'].values
                print(configs[i], group_label, result.loc[:,['join','join_std']].values)
                if "_n" in group_label:
                    if i == 0:
                        sum[False] += matrix[j,i]
                        # No Opt
                    elif i == 1:
                        sum[True] += matrix[j,i]
                        # Opt
            data_to_plot.append(matrix)
            data_to_plot_std.append(matrix_std)
        print(sum[True]/sum[False])
        # Drawing GroupedStackedHistogram
        width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
        plot = GroupedStackedHistogram(data_to_plot,(width, 3), data_std=data_to_plot_std)
        plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=[rename_configs_single(config) for config in configs], categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
        plot.draw_legend(inside=True)
        plot.save('validation_engine_join.png')

    else:
        pass

def rename_configs_all(st):
    return st.replace('all_heuristics', 'all').replace('no_heuristics','simult.').replace('only_prune_shape_network','simult. + shapes').replace('only_replace_target_query', 'simult. + targets').replace('all_sep', 'shapes + targets')

def rename_configs_single(st):
    return st.replace('all_heuristics', 'all').replace('no_heuristics','none').replace('only_prune_shape_network','shapes').replace('only_replace_target_query', 'targets')

def rename_category_viz(st):
    if st == 'io':
        return 'writing to disk'
    elif st == 'viz_hist':
        return 'histogram creation'
    elif st == 'viz_pie':
        return 'piechart creation'
    elif st == 'picture_composition':
        return 'composition'
    elif st == 'summarization':
        return 'summarizing'
    else:
        return st

if __name__ == '__main__':
    main()




# Join Results first vs. direct join with mapping




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

    # elif args.type == 'validation_engine_bars_shacl_times_rest':
    #     data, _, _,_ = load_data(args.input)
    #     #data.to_csv('output.csv')
    #     filtered_data = data[data['use_outer_join'] == True].loc[:,['shacl_schema_validation','join', 'join_std','shacl_schema_validation_std', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
    #     filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
    #     filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
    #     filtered_data_not_separate = filtered_data[filtered_data['constraints_separate'] == False].sort_values(by=['shape_schema_dir','api_config'])
    #     filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
    #     configs_not_separate = ['no_heuristics', 'only_replace_target_query', 'only_prune_shape_network', 'all_heuristics']
    #     configs = [config + '_sep' for config in configs_not_separate]
    #     group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
    #     data_to_plot = []
    #     data_to_plot_std = []

        
    #     for i, config in enumerate(configs_not_separate):
    #         matrix = np.zeros((len(group_labels), len(configs)))
    #         for j,group_label in enumerate(group_labels):
    #             n_constraints = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)].loc[:,'nconstraints'].values
    #             matrix[j,i] = n_constraints * filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)].loc[:,'shacl_schema_validation'].values
    #             if config == 'no_heuristics':
    #                 print(group_label, matrix[j,i])
    #         data_to_plot.append(matrix)

    #     # Drawing GroupedStackedHistogram
    #     width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
    #     plot = GroupedStackedHistogram(data_to_plot,(width, 3))
    #     plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=configs, categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
    #     plot.draw_legend()
    #     plot.save('ablation_study_shacl_time_rest.png')

    # elif args.type == 'validation_engine_bars_targets_rest':
    #     data, _, _,_ = load_data(args.input)
    #     #data.to_csv('output.csv')
    #     filtered_data = data[data['use_outer_join'] == True].loc[:,['number_of_targets', 'nconstraints', 'shape_schema_dir', 'api_config', 'constraints_separate']]
    #     filtered_data['shape_schema_dir'] = filtered_data['shape_schema_dir'].apply(lambda s: s.split('/')[-1].replace('full_binary_tree','bt').replace('single_overlap','o').replace('star_graph','star').replace('nested','n').replace('distinct','d'))
    #     filtered_data['api_config'] = filtered_data['api_config'].apply(lambda s: s.split('/')[-1]).apply(lambda s: s.split('.')[0])
    #     filtered_data_not_separate = filtered_data[filtered_data['constraints_separate'] == False].sort_values(by=['shape_schema_dir','api_config'])
    #     filtered_data_separate = filtered_data[filtered_data['constraints_separate'] == True].sort_values(by=['shape_schema_dir','api_config'])
        
    #     configs_not_separate = ['no_heuristics', 'only_replace_target_query', 'only_prune_shape_network', 'all_heuristics']
    #     configs = [config + '_sep' for config in configs_not_separate]
    #     group_labels = ['bt_7_o', 'bt_15_o', 'bt_31_o', 'star_5_o', 'star_15_o', 'star_25_o', 'star_15_d', 'bt_15_d', 'star_5_n']
    #     data_to_plot = []
    #     data_to_plot_std = []
        
    #     for i, config in enumerate(configs_not_separate):
    #         matrix = np.zeros((len(group_labels), len(configs)))
    #         for j,group_label in enumerate(group_labels):
    #             n_constraints = filtered_data_not_separate[(filtered_data_not_separate['api_config'] == config) & (filtered_data_not_separate['shape_schema_dir'] == group_label)].loc[:,'nconstraints'].values
    #             base_time = filtered_data_separate[(filtered_data_separate['api_config'] == config) & (filtered_data_separate['shape_schema_dir'] == group_label)].loc[:,'number_of_targets'].values
    #             matrix[j,i] = n_constraints * base_time
    #         data_to_plot.append(matrix)

    #     # Drawing GroupedStackedHistogram
    #     width = GroupedStackedHistogram.get_width(len(group_labels),len(configs))
    #     plot = GroupedStackedHistogram(data_to_plot,(width, 3))
    #     plot.draw(bar_labels=["" for config in configs],bar_labels_title="",group_labels=group_labels,group_labels_orientation='vertical', group_labels_title="", categorical_labels=configs, categorical_colors=[get_cmap(len(configs) +1,'hsv')(i) for i in range(len(configs)) ])
    #     plot.draw_legend()
    #     plot.save('ablation_study_targets_rest.png')


        # elif args.type == 'join_new':
    #     data, _, _,_ = load_data(args.input)
    #     defaults = {'n_samples': 4**10, 'n_nodes':4**10, 'n_constraints': 5}

    #     fig, ax = plt.subplots(1,3, sharey='all', figsize=(15,5))

    #     for i,experiment in enumerate(defaults):
    #         #fig, ax = plt.subplots(figsize=(6,5))
    #         locked_hps = {key: value for key, value in defaults.items() if key != experiment}
    #         filtered_data = data.loc[(data[list(locked_hps)] == pd.Series(locked_hps)).all(axis=1)]
    #         filtered_data = filtered_data.drop(columns = list(locked_hps))

    #         # Exclude the default value as this is the mean taken over n_constraints
    #         if experiment != 'n_constraints':
    #             filtered_data = filtered_data.loc[filtered_data[experiment] != defaults[experiment]]

    #         for use_outer_join, order_by_cardinality,not_pandas_optimized, label in [(False, False, False, 'Join T at the beginning'), (True, True, False, 'Join T at the end + Optimizations'), (True, False, False, 'Join T at the end')]: # (True, True, True, 'Join T at the end + Sort by Cardinality'), (True, False, True, 'Join T at the end')
    #             data_use_outer_join = filtered_data.loc[(filtered_data['use_outer_join'] == use_outer_join) & (filtered_data['optimize_intermediate_results'] == order_by_cardinality) & (filtered_data['not_pandas_optimized'] == not_pandas_optimized)]  # use_outer_join,order_by_cardinality --> FF, TT, TF
    #             data_use_outer_join = data_use_outer_join.sort_values(by=[experiment])

    #             x_data = data_use_outer_join[experiment]
    #             y_data = data_use_outer_join['join']
    #             y_err = data_use_outer_join['join_std']
    #             ax[i].errorbar(x_data, y_data, yerr=y_err, label=label)



    #         ax[i].set_ylabel('time [s]')
    #         ax[i].set_xlabel(experiment.replace('n_','#'))

    #     min_y, max_y = plt.ylim()

    #     for i,experiment in enumerate(defaults):
    #         if experiment == 'n_nodes':
    #             ax[i].vlines(defaults[experiment],min_y,max_y,'r', alpha=0.3)
    #         else:
    #             ax[i].vlines(defaults[experiment],min_y,max_y,'r', alpha=0.3)

    #     plt.title(f'')
    #     plt.legend()
    #     plt.savefig(f'join_exp_new')
    #     plt.close()



        # elif args.type == 'join_samples_nodes':
        # data, _,_,_ = load_data(args.input)
        # overall_list = np.linspace(4**4, 4**11, num = 5, dtype=np.int_)
        
        # fig, ax = plt.subplots(figsize=(10,10))

        # for join_outer in [False, True]:
        #     data_use_outer_join = data.loc[(data['use_outer_join'] == join_outer)]
        #     for n_samples_multiplier in [0.8, 1.0, 1.5]:
        #         if join_outer:
        #             n_samples = (overall_list * 0.95).astype(int)
        #         else:
        #             n_samples = overall_list
        #         n_nodes = n_samples_multiplier * n_samples
        #         #Collect values
        #         to_plot = []
        #         for n_s, n_n in zip(n_samples, n_nodes):
        #             result = data_use_outer_join.loc[(data_use_outer_join['n_samples'] == int(n_s)) & (data_use_outer_join['n_nodes'] == int(n_n))]
        #             print(result)
        #             to_plot.append(result['join'].values)
        #         #Plot values
        #         ax.plot(n_samples, to_plot, label=f'{("Join T at the end" if join_outer else "Join T at the beginning")}_{n_samples_multiplier}')

        # plt.title(f'')
        # plt.legend()
        # plt.savefig(f'join_samples_nodes_exp.png')
        # plt.close()