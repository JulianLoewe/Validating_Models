from types import FunctionType
import pandas as pd
import numpy as np
import seaborn as sn

from typing import List
from dtreeviz.utils import myround

from .checker import Checker
from .constraint import Constraint
from .drawing_utils import PieChart, StackedHistogram, GroupedStackedHistogram, Heatmap
from .colors import get_cmap
from functools import lru_cache

from validating_models.stats import get_decorator
time_generate_histogram = get_decorator('viz_hist')
time_generate_piechart = get_decorator('viz_pie')
time_generate_grouped_histogram = get_decorator('viz_grouped_hist')

class FrequencyDistributionTable:
    ''' Frequency distribution tables are used to summarize constraint validation results and visualize them accordingly. 

    A table shows the different measurement categories and the number of observations per category. 
    The measurement categories are constraints, the possible validation results and different groupings.
    The columns always have multiple hierachies, the first one refers to the different constraints and the second one to the different validation results (valid, invalid and not applicable).
    The index represents the division of the instances counted into the groups, when multiple group functions are given.
    
    Abstract definition of the function :math:`\mathcal{F}_{G,C}` used to create the frequency distribution table (without the coverage option):
    The function :math:`\mathcal{F}_{G,C}: \mathcal{P}([1,...,N]) \\times \mathbf{\Theta} \\times \mathbf{\Gamma} \\to \mathbb{N}^{|G| \\times |\{-1,0,1\}|}`, where

    * :math:`G` a set of arbitrary group identifiers
    * :math:`C` a constraint
    * :math:`N` the number of samples in the given dataset
    * :math:`\mathbf{\Gamma}` is the endless space of grouping function :math:`\Gamma: G \\to \mathcal{P}([1,...,N])`

    maps a subset of the indices :math:`D_{idx}` of a dataset :math:`D`, a model validation result function :math:`\Theta` and a grouping function :math:`\Gamma` to a frequency distribution table.
    :math:`\mathcal{F}_{G,C}(D_{idx},\Theta, \Gamma) \mapsto F_{G,\{-1,0,1\}}^{C,D_{idx}}`
    where
    
    * :math:`F_{G,\{-1,0,1\}}^{C,D_{idx}} = (|f_{g,v}^{C,D_{idx}}|)_{\substack{g \in G\\ v \in \{-1,0,1\}}}`
    * :math:`f_{g,v}^{C,D_{idx}} = \{ i | i \in (D_{idx} \cap \Gamma(g)) \land \Theta(C,i) = v\}`

    Parameters
    ----------
    checker : validating_models.checker.Checker
        The checker instance to be used to get the constraint validation results.
    constraints : list of validating_models.constraint.Constraint
        The constraints to be checked.
    indices : list of int, optional
        A list of indices referring to samples in the dataset, which should be used while creating the frequency distribution table.
    group_functions : list of group function (see validating_models.groupings)
        The list of group functions defining the groups based on the checker and the given indices.
    coverage : bool
        Whether to use "Coverage" to only count the most important validation results. That is given that the grouping is non overlapping the entries of the frequency distribution table will sum to the number of indices given.
    non_applicable_counts : bool
        Whether to use non applicable counts. The semantics depends on the type of the constraint.
    only_cached_results : bool, optional
        When set to true constraints are assumed to be already validated once, with this checker instance. There the validation step can be skipped. Defaults to False.
    '''

    def __init__(self, checker: Checker, constraints: List[Constraint], indices: List[int], group_functions: List[FunctionType], all_indices_groups=None, coverage=False, non_applicable_counts=False, only_cached_results=False, **args) -> None:
        self._constraints = constraints

        if indices == None:
            self._indices = list(range(len(checker.dataset)))
        else:
            self._indices = indices

        if all_indices_groups != None:
            self.max_samples_overall = 0
            self.min_samples_overall = np.inf

            for samples in all_indices_groups.values():
                self.max_samples_overall = len(samples) if len(
                    samples) > self.max_samples_overall else self.max_samples_overall
                self.min_samples_overall = len(samples) if len(
                    samples) < self.min_samples_overall else self.min_samples_overall
        else:
            self.max_samples_overall = len(list(set(range(len(checker.dataset))) - set(self._indices)))
            self.min_samples_overall = len(self._indices)


        if type(group_functions) != list:
            group_functions = [group_functions]

        self.fdt = checker.get_fdt(constraints=constraints, indices=self._indices, group_functions=group_functions,
                                   coverage=coverage, non_applicable_counts=non_applicable_counts,only_cached_results=only_cached_results, **args)

        self._full_fdt = self.fdt
        self.is_coverage = coverage
        self.is_normalized = False
        self.non_applicable_counts = non_applicable_counts

    @property
    def constraints_descriptor(self):
        if self.is_coverage:
            return 'Coverage'
        else:
            return 'Constraints'

    @property
    def constraint_names(self):
        if self.is_coverage:
            # In case of coverage the results are summarized as if there is a single constraint
            return ['']
        else:
            return list(self.fdt.columns.get_level_values(0).unique())

    def group_names(self, level=0):
        return list(self.fdt.index.get_level_values(level))

    def group_descriptor(self, level=0):
        return self.fdt.index.names[level]

    @property
    def categories_descriptor(self):
        if self.is_coverage:
            return 'Categories'
        else:
            return 'Validation Results'

    @property
    def category_names(self):
        if self.is_coverage:
            return [(f'{constraint} - {valres}' if constraint != '' else f'{valres}') for constraint, valres in list(self.fdt.columns)]
        else:
            return self.fdt.columns.get_level_values(1).unique()

    @lru_cache
    def get_colors(self, n, cmap_name='hsv'):
        if n == 4:
            return lambda i: 'g' if i == 0 else ('r' if i == 1 else ('grey' if i == 2 else get_cmap(n, name=cmap_name)(i - 3)))
        else:
            return get_cmap(n, name=cmap_name)

    def remove_only_zeros_columns(self):
        self.fdt = self.fdt.loc[:, (self.fdt != 0).any(axis=0)]
        return self

    def normalize(self, level=None):
        if not self.is_normalized:
            self.fdt = self.fdt / len(self._indices)
            self.is_normalized = True
        return self

    def denormalize(self, level=None):
        if self.is_normalized:
            self.fdt = self.fdt * len(self._indices)
            self.is_normalized = False
        return self

    # TODO: Generalize for multiple Constraints and normalized data
    def get_validation_result_summary(self, level, number_of_samples=True, percentage_per_validation_result=True):

        def _overall_summary():
            if not self.is_normalized:
                nsamples = len(self._indices)
            else:
                nsamples = 1.0

            invalid_count = np.sum(self.fdt.filter(like='invalid').values)
            not_applicable_count = np.sum(self.fdt.filter(
                like='not applicable').values) + np.sum(self.fdt.filter(like='not covered').values)
            valid_count = nsamples - invalid_count - not_applicable_count

            summary = pd.DataFrame([valid_count, invalid_count, not_applicable_count], columns=[
                                   'Overall'], index=pd.Index(['valid', 'invalid', 'not applicable'], name='Validation Summary'))
            summary = (summary / nsamples) * 100
            return summary

        def _per_group_summary():
            return None

        def _per_constraint_summary():
            return None

        # # Identify what are the options for normalization: Constraint-Groups Combination
        # if self.is_coverage:
        #     column_agg = self.fdt.sum(axis=1)
        # else:
        #     column_agg = self.fdt.groupby(axis=1,level=0).sum().iloc[:,0] # Counting the same indices per Constraint therefore taking the first is enough

        # return column_agg

        if not self.is_coverage:
            # Check is per Group
            if (self.fdt.groupby(axis=1, level=0).sum().values == len(self._indices)).all():
                if len(self.group_names(level)) == 1:
                    return _overall_summary()
                else:
                    return _per_group_summary()
            else:
                if len(self._constraints) == 1:
                    return _overall_summary()
                else:
                    return _per_constraint_summary()
        else:
            # Check is per Group
            if (np.sum(self.fdt.values, axis=1) == len(self._indices)).all():
                if len(self.group_names(level)) == 1:
                    return _overall_summary()
                else:
                    return _per_group_summary()
            else:
                return _overall_summary()

    @time_generate_piechart
    def _visualize_with_piechart(self, title, figure_size, scale, fontname, fontsize, summary, additional_text, level, ax):
        ncategories = len(self.category_names)
        nconstraints = len(self.constraint_names)

        if figure_size == None:
            height = PieChart.get_height(
                self._indices, self.max_samples_overall, self.min_samples_overall) * scale
            figure_size = (height, height)

        ngroups = len(self.group_names(level))

        counts = self.fdt.values.squeeze()
        plot = PieChart(counts, figure_size=figure_size,
                        fontname=fontname, fontsize=fontsize, ax=ax)

        summary = self.get_validation_result_summary(level)
        nsamples = int(np.sum(counts))
        text = f"n = {nsamples}\nValid: {myround(summary.loc['valid'].iat[0])}%\nInvalid: {myround(summary.loc['invalid'].iat[0])}%\nNot applicable: {myround(summary.loc['not applicable'].iat[0])}%"
        ncounts = len(counts)
        cm = self.get_colors(ncounts + 1)
        colors = [cm(i) for i in range(ncounts)]

        if ncategories == 1 and nconstraints == 1:
            labels = self.group_names(level)
        elif ngroups == 1 and ncategories == 1:
            labels = self._constraints
        elif ngroups == 1 and nconstraints == 1:
            labels = self.category_names
        else:
            raise Exception(
                f'Frequency Distribution cannot be visualized with a piechart')

        plot.draw(colors=colors, text=text, labels=labels)
        plot.title(title)
        return plot
    
    @time_generate_histogram
    def _visualize_with_stacked_histogram(self, title, figure_size, scale, fontname, fontsize, summary, additional_text, level, ax):
        ncategories = len(self.category_names)
        nconstraints = len(self.constraint_names)
        ngroups = len(self.group_names(level))

        if ncategories == 1 or nconstraints == 1:
            data = self.fdt.values.T
            bar_labels = self.group_names(level)
            bar_labels_title = self.group_descriptor(level)
        elif ngroups == 1:
            data = self.fdt.values.reshape((-1, ncategories)).T
            bar_labels = self.constraint_names
            bar_labels_title = self.constraints_descriptor
        else:
            raise Exception(
                f'Frequency Distribution cannot be visualized with a stacked histogram')

        if figure_size == None:
            # StackedHistogram builds on GroupedStackedHistogram but without spaces between groups and 1 bar per group
            width = StackedHistogram.get_width(len(bar_labels), 1) * scale
            height = StackedHistogram.get_height(
                self._indices, self.max_samples_overall, self.min_samples_overall) * scale
            figure_size = (width, height)

        if nconstraints == 1 or ngroups == 1:
            categorical_labels = self.category_names
            cm = self.get_colors(ncategories + 1)
            colors = [cm(i) for i in range(ncategories)]
        elif ncategories == 1:
            categorical_labels = self.constraint_names
            cm = self.get_colors(nconstraints + 1)
            colors = [cm(i) for i in range(nconstraints)]
        else:
            raise Exception(
                f'Frequency Distribution cannot be visualized with a stacked histogram')

        plot = StackedHistogram(
            data, figure_size=figure_size, fontname=fontname, fontsize=fontsize, ax=ax)
        plot.draw(bar_labels=bar_labels, bar_labels_title=bar_labels_title,
                  categorical_labels=categorical_labels, categorical_colors=colors)
        plot.set_ylabel('#samples')
        plot.title(title)
        return plot

    @time_generate_grouped_histogram
    def _visualize_with_grouped_stacked_histogram(self, title, figure_size, scale, fontname, fontsize, summary, additional_text, level, ax):

        if figure_size == None:
            width = GroupedStackedHistogram.get_width(
                len(self.group_names(level)), len(self.constraint_names)) * scale
            height = GroupedStackedHistogram.get_height(
                self._indices, self.max_samples_overall, self.min_samples_overall) * scale
            figure_size = (width, height)

        data = self.fdt.values.reshape((len(self.group_names(level)), len(self.constraint_names), len(
            self.category_names)))  # shape (#groups, #constraints, #categories)

        # shape (#categories, #groups, #constraints)
        data = np.transpose(data, axes=(2, 0, 1))
        data = np.split(data, len(self.category_names), axis=0)
        data = [array.reshape((len(self.group_names(level)), len(
            self.constraint_names))) for array in data]
        plot = GroupedStackedHistogram(
            data, figure_size=figure_size, fontname=fontname, fontsize=fontsize, ax=ax)

        ncategories = len(self.category_names)
        cm = self.get_colors(ncategories + 1)
        colors = [cm(i) for i in range(ncategories)]
        plot.draw(bar_labels=self.constraint_names, bar_labels_title=self.constraints_descriptor, group_labels=self.group_names(level),
                  group_labels_title=self.group_descriptor(level), categorical_labels=self.category_names, categorical_colors=colors)
        plot.set_ylabel('#samples')
        plot.title(title)
        return plot

    def matrix(self, category='invalid', fancy=True, range=[None, None]):

        if self.fdt.index.nlevels != 2:
            raise Exception(
                'This function can only be applied if exactly 2 group functions were used during creation.'
            )
        
        if len(self.constraint_names) != 1:
            raise Exception(
                'This function can only be applied, when visualizing exactly 1 constraint.'
            )
        
        if category == 'all':
            select_category = ['valid','invalid','not applicable']
        else:
            select_category = [category]
        
        if self.is_coverage == False:
            selected_columns = [[self.constraint_names[0]], select_category]
        else:
            selected_columns = [slice(None),select_category]
        
        col_loc = self.fdt.columns.get_locs(selected_columns)
        summarized = self.fdt.iloc[:,col_loc]

        if self.is_coverage:
            summarized = summarized.groupby(level=1, axis=1).sum()

        if category == 'all':
            summarized = summarized.sum(axis=1)

        index1 = self.fdt.index.get_level_values(0).unique()
        index2 = summarized.index.get_level_values(1).unique()
        matrix_array = summarized.values.reshape(len(index1), len(index2))
        df = pd.DataFrame(matrix_array, columns=index2, index=index1)
        
        if range == [None, None]:
            max = df.max().max()
            if max == 0:
                max = self.fdt.max().max()
            range = [0,max]

        if not fancy:
            return df
        else:
            plot = Heatmap(df, figure_size=(len(index1) + 1, len(index2) + 1))
            if category == 'valid':
                plot.draw(cmap=sn.light_palette("seagreen"), range=range)
            elif category == 'invalid':
                plot.draw(cmap=sn.light_palette("red"), range=range)
            elif category == 'not applicable':
                plot.draw(cmap=sn.light_palette("grey"), range=range)
            else:
                plot.draw(cmap=None, range=range)
            return plot

    def visualize(self, title='', selected_groups=None, level_to_visualize=0, aggregation_methode='sum()', scale=1.0, summary=False, additional_text='', figure_size=None, type=None, fontname='DejaVu Sans', fontsize=9, ax=None):
        '''Automated visualiztion of the frequency distribution table for one grouping.

        Parameters
        ----------
        title : str
            The title to use for the visualization.
        selected_groups : label, slice, list, mask or a sequence of such
            A selection of groups to be used. See pandas.MultiIndex.get_locs for examples.
        level_to_visualize : int
            The level (group) to be visualized. A pandas groupby is performed on the given level.
        aggregation_methode : string
            If there are multiple groupings the rest has to be aggregated.
        scale : float
            Scales the visualization accordingly, however it's recommended to scale the visualiation while saving to file.
        summary : bool
            Ignored (to be implemented in the feature)
        additional_text : str
            Ignored (to be implemented in the feature)
        figure_size : (int, int), optional
            The size of the visualisation to be created. Normally the figure size can be determined automatically, use in case it fails.
        type : A type, optional
            Force a specific type of visualization: Can be one of GroupedStackedHistogram, StackedHistogram or PieChart. See validating_models.drawing_utils.
        fontname : str, optional
            The font to be used
        fontsize : int, optional
            The size of the font to be used
        ax : matplotlib.axis, optional
            The axis to draw the visualization on.
        '''

        if selected_groups == None:
            selected_groups = [slice(None)
                               for i in range(self.fdt.index.nlevels)]

        loc = self.fdt.index.get_locs(selected_groups)

        fontsize = fontsize * scale

        try:
            self.fdt = self.fdt.iloc[loc].groupby(
                level=level_to_visualize, sort=False)
            self.fdt = eval('self.fdt.' + aggregation_methode)
            if type != None:
                if type.__name__ == GroupedStackedHistogram.__name__:
                    plot = self._visualize_with_grouped_stacked_histogram(
                        title, figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                elif type.__name__ == StackedHistogram.__name__:
                    plot = self._visualize_with_stacked_histogram(
                        title, figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                elif type.__name__ == PieChart.__name__:
                    plot = self._visualize_with_piechart(title, figure_size, scale=scale, fontname=fontname,
                                                         fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                else:
                    raise Exception(
                        f'The specified type of visualization ({type.__name__}) is not implemented.')
            else:
                if len(self.group_names(level_to_visualize)) == 1:
                    if len(self.category_names) == 1:
                        if len(self.constraint_names) == 1:
                            # Only One Entry
                            plot = self._visualize_with_piechart(title=title, figure_size=figure_size, scale=scale, fontname=fontname,
                                                                 fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                        else:
                            # Multiple Constraints
                            plot = self._visualize_with_piechart(title=title, figure_size=figure_size, scale=scale, fontname=fontname,
                                                                 fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                    else:
                        if len(self.constraint_names) == 1:
                            # Multiple Categories
                            plot = self._visualize_with_piechart(title=title, figure_size=figure_size, scale=scale, fontname=fontname,
                                                                 fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                        else:
                            # Multiple constraints, multiple categories
                            plot = self._visualize_with_stacked_histogram(
                                title=title, figure_size=figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                else:
                    if len(self.category_names) == 1:
                        if len(self.constraint_names) == 1:
                            # Multiple Groups
                            plot = self._visualize_with_piechart(title=title, figure_size=figure_size, scale=scale, fontname=fontname,
                                                                 fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                        else:
                            # Multiple Groups, multiple constraints
                            plot = self._visualize_with_stacked_histogram(
                                title=title, figure_size=figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                    else:
                        if len(self.constraint_names) == 1:
                            # Multiple groups, multiple categories
                            plot = self._visualize_with_stacked_histogram(
                                title=title, figure_size=figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
                        else:
                            # Multiple groups, multiple categories, multiple constraints
                            plot = self._visualize_with_grouped_stacked_histogram(
                                title=title, figure_size=figure_size, scale=scale, fontname=fontname, fontsize=fontsize, summary=summary, additional_text=additional_text, level=level_to_visualize, ax=ax)
        except Exception as e:
            raise e
        finally:
            self.fdt = self._full_fdt
        return plot

    def _repr_html_(self):
        return self.fdt.style._repr_html_()
