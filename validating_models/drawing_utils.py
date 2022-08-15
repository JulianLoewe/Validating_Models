import numpy as np
from typing import List, Tuple
import pandas as pd
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dtreeviz.utils import myround
from matplotlib import transforms
import seaborn as sn

from .colors import adjust_colors
from abc import ABC, abstractmethod

from validating_models.stats import get_decorator

time_io = get_decorator('io')
time_io_histogram = get_decorator('io_histogram')
time_io_piechart = get_decorator('io_piechart')

class Visualization(ABC):
    '''Abstract Class for representing a Visualization. 

    This class is used to have a uniform design e.g. font and colors. 
    It's further used to store so called proxy artists, which are not drawn directly, but will be visible in legends.

    Attributes
    ----------
        fig : matplotlib.figure.Figure
            The base figure of the visualization.
        ax : matplotlib.axes.Axes
            The base axes of the visualization, containing the main drawings.
        proxy_artists : list of (matplotlib.artist.Artist, str)
            Components not recognized by matplotlib, which however should be drawn into the legend.

    Parameters
    ----------
        figure_size : (int,int)
            The size of the figure.
        ax : matplotlib.axes.Axes, optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 
    '''

    def __init__(self, figure_size,
                 ax = None,
                 proxy_artists=None,
                 fontname='DejaVu Sans',
                 fontsize=14,
                 graph_colors=None) -> None:
        self._fontname = fontname
        self._fontsize = fontsize
        self._graph_colors = adjust_colors(graph_colors)
        if ax == None:
            self.fig, self.ax = plt.subplots(figsize=figure_size)
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self._width = 1

        # list of (handle, label) of additional Components to include in the legend when generated
        self.proxy_artists = [] if proxy_artists == None else proxy_artists

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_linewidth(.3)
        self.ax.spines['bottom'].set_linewidth(.3)
        super().__init__()

    def title(self, title):
        '''Sets the title of the visualization.

        Parameters
        ----------
            title : str
                The title of the visualization.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        self.ax.set_title(title, fontsize=self._fontsize + 5,
                          fontname=self._fontname, color=self._graph_colors['title'])
        #        self.ax.text(1.0,1.0, title, transform=transforms.IdentityTransform(), fontsize=30,
        #                  fontname=self._fontname, color=self._graph_colors['title'], ha='center', va='top') TODO: Fix for PieChart
        return self

    def set_xlabel(self, label):
        '''Sets the label of the x-Axis of the visualization.

        Parameters
        ----------
            label : str
                The label to be used for the x axis.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        self.ax.set_xlabel(label, fontsize=self._fontsize,
                           fontname=self._fontname, color=self._graph_colors['axis_label'])
        return self

    def set_ylabel(self, label):
        '''Sets the label of the y-Axis of the visualization.

        Parameters
        ----------
            label : str
                The label to be used for the y-Axis.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        self.ax.set_ylabel(label, fontsize=self._fontsize,
                           fontname=self._fontname, color=self._graph_colors['axis_label'])
        return self

    def set_tick_params(self, axis='both', which='major', length=None):
        '''Sets the tick parameters using ax.tick_params.

        Parameters
        ----------
            axis : {'x','y','both'}
                The axis to which the tick parameters of the visualization are set.
            which : {'major','minor','both'}
                The group of ticks to which the tick parameters of the visualization are set.
            length : int
                The length of the ticks in points

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        if length != None:
            self.ax.tick_params(axis=axis, which=which, length=length, width=.3,
                                labelcolor=self._graph_colors['tick_label'], labelsize=self._fontsize)
        else:
            self.ax.tick_params(axis=axis, which=which, width=.3,
                                labelcolor=self._graph_colors['tick_label'], labelsize=self._fontsize)
        return self

    def set_label_orientation(self, orientation, major=True):
        '''Sets the orientation of the tick labels on the x-Axis.

        Parameters
        ----------
            orientation : {horizontal, vertical}
                The orientation of the labels to be set.
            major : bool
                The group of ticks to which the orientation should be set

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        for t in self.ax.get_xticklabels(minor=not major):
            t.set_rotation(orientation)
        return self

    def draw_legend(self, inside=False):
        '''Draws the legend next to the visualization.

        Parameters
        ----------
            inside : boolean
                Whether the legend should be drawn inside the figure or next to it.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        hl = list(zip(*self.ax.get_legend_handles_labels()))
        hl = hl + self.proxy_artists
        h = zip(*hl)
        if not inside:
            self.ax.legend(*h, bbox_to_anchor=(1.04, 1), loc="upper left",
                        fontsize=self._fontsize, title_fontsize=self._fontsize, title='', bbox_transform=self.fig.transFigure)
        else:
            self.ax.legend(*h, loc="upper right",
                        fontsize=self._fontsize, title_fontsize=self._fontsize, bbox_transform=self.fig.transFigure)
        return self

    @time_io
    def save(self, filename, transparent=True, dpi=1000):
        '''Saves the visualization to a file with path filename.

        Parameters
        ----------
            filename : str
                The path of the file, to which the visualization should be saved.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        plt.figure(self.fig.number)
        plt.savefig(filename, bbox_inches='tight',
                    pad_inches=0, transparent=transparent, dpi=dpi)
        plt.close()
        return self

    def draw_wedge(self, x, x_range=None, minor=True):
        '''
        Modifies the ticks on the x-Axis such that the ticks of the given group are longer, drawn in red and a new tick at position x is added.

        Parameters
        ----------
            x : double
                The position at which a new tick should be drawn.
            x_range : (double, double)
                The range of the x axis, which is assumed by the given x.
            minor : bool
                The group of ticks to which the new tick is added and which are now longer and red.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        if x_range != None:
            x_pos = self.transform(x, x_range)
        else:
            x_pos = x

        # Add Text as Tick
        x_ticks = list(self.ax.get_xticks(minor=minor))
        x_tick_labels = list(self.ax.get_xticklabels(minor=minor))

        x_ticks.append(x_pos)
        x_tick_labels.append(str(myround(x, 2)))
        self.ax.set_xticks(x_ticks, minor=minor)
        self.ax.set_xticklabels(x_tick_labels, minor=minor)
        self.ax.tick_params(
            which=('minor' if minor else 'major'), length=12, color='r')
        return self

    def transform(self, x, range):
        '''Transforms x given the range onto the range of the x-axis of the visualization.

        Parameters
        ----------
            x : double
                The value to be transformed.
            range : (double, double)
                The range of x, which is assumed.

        Returns
        -------
            double
                The result
        '''
        old_range = range  # x coordinates
        new_range = self.ax.get_xlim()  # data coordinates
        return (x - old_range[0]) * (new_range[1] - new_range[0])/(old_range[1] - old_range[0]) + new_range[0]

    @abstractmethod
    def draw(self, **args):
        '''Draws the visualization.

        Parameters
        ----------
            args
                The additional parameters needed to draw the visualization

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        pass

    @staticmethod
    @abstractmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):  # Groups of indices
        pass

    @staticmethod
    @abstractmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass


class GroupedStackedHistogram(Visualization):
    '''A visualization consisting of groups of stacked bars. Each group has the same number of bars. A stacked bar consists of data of different categories.

    Parameters
    ----------
        data: list of numpy.ndarray with shape (#groups, #bars_per_group)
            The data to be visualized
        figure_size : (int,int), optional
            The size of the figure.
        ax : matplotlib.axes.Axes), optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 
        data_std : list of numpy.ndarray with shape (#groups, #bars_per_group)
            The standard deviations to visualize matching to the data parameter, defaults to None

    Example
    -------
    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from validating_models.drawing_utils import GroupedStackedHistogram
        >>> categorical_labels = ['A','B','C','D']
        >>> categorical_colors = ['r','b','g','c']
        >>> num_groups = 3
        >>> num_bars_per_group = 2
        >>> data = np.random.rand(len(categorical_labels), num_groups, num_bars_per_group)
        >>> figure_size = (GroupedStackedHistogram.get_width(num_groups, num_bars_per_group), 3)
        >>> plot = GroupedStackedHistogram([data[0,:,:],data[1,:,:],data[2,:,:],data[3,:,:]],figure_size)
        >>> bar_labels = [f'bar_{i}' for i in np.arange(num_bars_per_group)]
        >>> group_labels = [f'group_{j}' for j in np.arange(num_groups)]
        >>> plot.draw(bar_labels, 'Bar Label', group_labels=group_labels, categorical_labels=categorical_labels, categorical_colors=categorical_colors)
        >>> plot.draw_legend()

    '''

    def __init__(self,
                 # List of categories. Per category a array of shape (#groups, #bars_per_group) e.g. [not_applicable_array, invalid_array, valid_array]
                 data,
                 figure_size,
                 ax = None,
                 proxy_artists=None,
                 fontname='DejaVu Sans',
                 fontsize=14,
                 graph_colors=None,
                 data_std=None) -> None:
        self._data = data
        if isinstance(data_std, list):
            print('std given')
            self._data_std = data_std
        else:
            self._data_std = [np.zeros_like(self._data[0]) for i in range(len(self._data))]
        self._num_groups = self._data[0].shape[0]
        self._num_bars_per_group = self._data[0].shape[1]
        super().__init__(figure_size, ax, proxy_artists, fontname, fontsize, graph_colors)

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        '''Approximates the needed size for the figure given the number of groups and bars per group
        '''
        result = ((number_of_groups - 1) +
                  number_of_groups * num_bars_per_group)/4
        return result

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, logscaling_base: float = 10.0):
        '''Function to calculate the height of a histogram. 

        Given that there are different groups with different amount of samples, and in a histogram a selected group of groups are visualized.
        This methode calculates the height of the histogram with the selected groups, such that the height of the histogram is scaled relativ to the maximum number of samples in all groups.

        Parameters
        ----------
            selected_groups : list of str
                The selected groups visualized in the histogram, whose height is to be determined. Has to be a list of keys of all_groups.
            all_groups : dict of (str -> list of indices) 
                A dictionary containing the indices of the samples per group.
            logscaling_base : float, optional
                When given the height is scaled logithmically with the given base.
        '''
        # Adding + 1 to every value to avoid the zero.
        samples_nodes = len(indices) + 1
        max_samples_overall += 1
        min_samples_overall += 1

        if logscaling_base:
            max_samples_overall = np.log(
                max_samples_overall)/np.log(logscaling_base)
            samples_nodes = np.log(samples_nodes)/np.log(logscaling_base)
            min_samples_overall = np.log(
                min_samples_overall)/np.log(logscaling_base)



        height_range = (1, 2)
        if (max_samples_overall - min_samples_overall) == 0:
            h = height_range[1]
        else:
            h = (samples_nodes - min_samples_overall) * (height_range[1] - height_range[0])/(
                max_samples_overall - min_samples_overall) + height_range[0]
        return h
    
    @time_io_histogram
    def save(self, filename, transparent=True, dpi=1000):
        return super().save(filename, transparent, dpi)

    def draw(self,
             # major -> len(bar_labels) = #bars_per_group
             bar_labels,
             bar_labels_title,
             bar_labels_orientation='vertical',
             # minor -> len(group_labels) = #groups
             group_labels=None,
             group_labels_title="",
             group_labels_orientation='vertical',
             categorical_labels=None,
             categorical_colors=None,
             space_between_groups=True):
        '''Draws the GroupedStackedHistogram.

        Parameters
        ----------
            bar_labels : list of str
                The labels of the bars in each group.
            bar_labels_title : str
                A title for the bars.
            bar_labels_orientation : {'horizontal','vertical'}, optional
                The orientation of of the bar labels.
            group_labels : list of str, optional
                The labels of the groups
            group_labels_title : title, optional
                The title for the groups
            group_labels_orientation : {'horizontal','vertical'}, optional
                The orientation of the group labels.
            categorical_labels : list of str
                The labels for the different categories.
            categorical_colors : list of colors
                The colors of the different categories
            space_between_groups: bool, optional
                Whether there should be space between the groups of bars.
        '''

        num_bars = self._num_groups * self._num_bars_per_group

        # Preprocess data
        for i in range(len(self._data)):
            self._data[i] = self._data[i].reshape((-1,))
            self._data_std[i] = self._data_std[i].reshape((-1,))                

        if not space_between_groups:
            bar_locations = np.arange(start=0, stop=num_bars, step=1)
            leaf_locations = np.arange(
                start=self._num_bars_per_group/2, stop=self._num_groups, step=self._num_bars_per_group) + 0.01  # bar_locations cannot be equal to leaf_locations therefore 0.01 is added
        else:
            bar_locations = np.setdiff1d(np.arange(start=0, stop=self._num_groups + num_bars, step=1), np.arange(
                start=self._num_bars_per_group, stop=self._num_groups + num_bars, step=self._num_bars_per_group + 1))
            leaf_locations = np.arange(start=(self._num_bars_per_group - self._width)/2,
                                       stop=self._num_groups + num_bars, step=self._num_bars_per_group + self._width) + 0.01  # bar_locations cannot be equal to leaf_locations therefore 0.01 is added

        bottom = np.zeros_like(self._data[0])
        for i, array in enumerate(zip(self._data, self._data_std)):
            if np.count_nonzero(array[0]) != 0:
                self.ax.bar(bar_locations, array[0], self._width, bottom=bottom,edgecolor=self._graph_colors['pie'],
                            color=categorical_colors[i], lw=.3, align='center', label=categorical_labels[i], yerr=array[1])
                bottom = bottom + array[0]

        # Bars (Major Labels)
        max_height = 0
        max_width = 0

        self.ax.set_xticks(bar_locations, minor=False)

        xticklabels = [label for i, label in enumerate(repmat(np.array(
            bar_labels), 1, self._num_groups).reshape((-1,))) if i < len(bar_locations)]

        self.ax.set_xticklabels(xticklabels, minor=False)
        for t in self.ax.get_xticklabels(minor=False):
            t.set_rotation(bar_labels_orientation)

        self.set_tick_params(axis='x')

        # self.ax.draw(self.fig.canvas.get_renderer())
        if (np.array(bar_labels, dtype=str) != "").any():

            for t in self.ax.get_xticklabels(minor=False):
                bbox = t.get_window_extent(renderer=self.fig.canvas.get_renderer()).transformed(
                    self.ax.transAxes.inverted())  # gets the bounding box of the bar labels in Axes coordinates
                max_height = bbox.height if bbox.height > max_height else max_height
                max_width
        else:
            max_height = -0.1

        # Groups (Minor Labels)
        if group_labels != None and len(group_labels) > 1:
            self.ax.set_xticks(leaf_locations, minor=True)
            self.ax.set_xticklabels(group_labels, minor=True)
            self.set_tick_params(axis='x', which='minor', length=0)
            for t in self.ax.get_xticklabels(minor=True):
                t.set_rotation(group_labels_orientation)
                t.set_y(-max_height-0.1)

        self.set_tick_params(axis='y')

        if group_labels_title != '':
            self.set_xlabel(bar_labels_title + ' per ' + group_labels_title)
        else:
            self.set_xlabel(bar_labels_title)


class StackedHistogram(GroupedStackedHistogram):
    '''A visualization consisting of stacked bars. A stacked bar consists of data of different categories.

    Parameters
    ----------
        data: numpy.ndarray with shape (#categories, #bars)
            The data to be visualized
        figure_size : (int,int), optional
            The size of the figure.
        ax : matplotlib.axes.Axes), optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 

    Example
    -------
    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from validating_models.drawing_utils import StackedHistogram
        >>> categorical_labels = ['A','B','C','D']
        >>> categorical_colors = ['r','b','g','c']
        >>> num_bars = 10
        >>> data = np.random.rand(len(categorical_labels), num_bars)
        >>> plot = StackedHistogram(data,(StackedHistogram.get_width(1, num_bars),3))
        >>> bar_labels = [f'bar_{i}' for i in np.arange(num_bars)]
        >>> plot.draw(bar_labels, 'Bar Label', categorical_labels=categorical_labels, categorical_colors=categorical_colors)
        >>> plot.draw_legend()
    '''

    def __init__(self, data,  # e.g. shape (#categories, #bars)
                 figure_size,
                 ax = None,
                 proxy_artists=None,
                 fontname='DejaVu Sans',
                 fontsize=14,
                 graph_colors=None,
                 data_std=None) -> None:

        self._mydata = [data[i, :].reshape((-1, 1))
                        for i in range(data.shape[0])]
        if isinstance(data_std, np.ndarray):
            self._mydata_std = [data_std[i, :].reshape((-1, 1))
                            for i in range(data_std.shape[0])]
        else:
            self._mydata_std = None
        super().__init__(self._mydata, figure_size, ax,
                         proxy_artists, fontname, fontsize, graph_colors, data_std=self._mydata_std)

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        '''Approximates the needed size for the figure given the number of groups and bars per group
        '''
        result = (number_of_groups * num_bars_per_group)/4
        return result

    def draw(self, bar_labels,
             bar_labels_title,
             bar_labels_orientation='vertical',
             categorical_labels: List[str] = None,
             categorical_colors: List[str] = None):
        '''Draws the StackedHistogramm.

        Parameters
        ----------
            bar_labels : list of str
                The labels of the bars in each group.
            bar_labels_title : str
                A title for the bars.
            bar_labels_orientation : {'horizontal','vertical'}, optional
                The orientation of of the bar labels.
            categorical_labels : list of str
                The labels for the different categories.
            categorical_colors : list of colors
                The colors of the different categories
        '''
        super().draw(bar_labels=bar_labels,
                     bar_labels_title=bar_labels_title, bar_labels_orientation=bar_labels_orientation, categorical_labels=categorical_labels, categorical_colors=categorical_colors, space_between_groups=False)

    def add_wedge(self, x, x_range, categorical_labels=None):
        '''Adds a highlighted tick at position x given the range of x. When categorical_labels is given the x-Axis is labeled accordingly.

        Parameters
        ----------
            x : double
                The position at which a new tick should be drawn.
            x_range : (double, double)
                The range of the x axis, which is assumed by the given x.
            categorical_labels : dict of (double -> str) assignments
                If set the labels are shown as sets on the limits of the x-Axis. The left side of the x-Axis contains the labels, which have a smaller value then x and the right side the bigger ones.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        coordinates_range = [-0.5,
                             self._num_bars_per_group * self._num_groups - 1 + 0.5]
        self.ax.set_xticks(coordinates_range)
        if categorical_labels != None:
            labels = categorical_labels
        else:
            labels = [myround(value) for value in x_range]
        self.ax.set_xticklabels(labels)
        self.ax.set_yticks([0, np.max(np.sum(self._mydata, axis=0))])
        self.ax.set_xlim(coordinates_range)
        self.draw_wedge(x, x_range)
        return self


class PieChart(Visualization):
    '''A PieChart showing the fractions of the given counts. Only counts != 0 will be added to the legend.

    Parameters
    ----------
        counts : numpy.ndarray of shape (#counts,) with dtype int
            The data to be visualized.
        figure_size : (int,int)
            The size of the figure.
        ax : matplotlib.axes.Axes), optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 

    Example
    -------
    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from validating_models.drawing_utils import PieChart
        >>> categorical_labels = ['A','B','C','D']
        >>> categorical_colors = ['r','b','g','c']
        >>> data = np.array([5,4,3,2])
        >>> plot = PieChart(data, (3,3))
        >>> plot.draw(colors=categorical_colors, text='Some text', labels=categorical_labels)
        >>> plot.draw_legend()
    '''

    def __init__(self, counts,  # counts of shape (#counts,)
                 figure_size=None,
                 ax = None,
                 proxy_artists=None,
                 fontname='DejaVu Sans',
                 fontsize=9,
                 graph_colors=None) -> None:

        self._counts = counts
        if figure_size == None:
            size = self.get_size(np.sum(self._counts))
            figure_size = (size, size)

        super().__init__(figure_size, ax=ax, proxy_artists=proxy_artists, fontname=fontname,
                         fontsize=fontsize, graph_colors=graph_colors)
        self._draw_mask = np.where(np.array(self._counts) != 0)[0]

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        nsamples = len(indices)
        return PieChart.get_size(nsamples)

    def get_width(number_of_groups, num_bars_per_group, **args):
        pass
    
    @time_io_piechart
    def save(self, filename, transparent=True, dpi=1000):
        return super().save(filename, transparent, dpi)

    @staticmethod
    def get_size(nsamples):
        '''Calculates the size of a visualization given the number of samples.

        Parameters
        ----------
            nsamples : int
                The number of samples

        Returns
        -------
            double
                The recommended size.
        '''
        minsize = .15
        maxsize = 1.3
        slope = 0.02
        size = nsamples * slope + minsize
        size = min(size, maxsize)
        return size

    def draw(self, colors=None, text=None, labels=None):
        '''Draws the PieChart.

        Parameters
        ----------
            colors : list of str
                The color for each count
            text : str
                The text shown below the PieChart
            labels : list of str
                A label for each count to be used in legends.
        '''
        if colors == None:
            colors = self._graph_colors['classes'][10]
        counts = np.array(self._counts).squeeze()
        size = self.fig.get_figheight()
        tweak = size * .01
        self.ax.axis('equal')
        self.ax.set_xlim(0, self.fig.get_figwidth() - 10 * tweak)
        self.ax.set_ylim(0, self.fig.get_figheight() - 10 * tweak)
        # frame=True needed for some reason to fit pie properly (ugh)
        # had to tweak the crap out of this to get tight box around piechart :(
        wedges, _ = self.ax.pie(counts[self._draw_mask], center=(size / 2 - 6 * tweak, size / 2 - 6 * tweak), radius=size / 2, colors=np.array(colors)[self._draw_mask],
                                shadow=False, frame=True, labels=np.array(labels)[self._draw_mask], labeldistance=None)
        for w in wedges:
            w.set_linewidth(.5)
            w.set_edgecolor(self._graph_colors['pie'])

        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        if text is not None:
            self.ax.text(size / 2 - 6 * tweak, -10 * tweak, text,
                         horizontalalignment='center',
                         verticalalignment='top',
                         fontsize=self._fontsize, color=self._graph_colors['text'], fontname=self._fontname)


class Heatmap(Visualization):
    def __init__(self, df, figure_size, ax = None, proxy_artists=None, fontname='DejaVu Sans', fontsize=14, graph_colors=None) -> None:
        super().__init__(figure_size, ax, proxy_artists, fontname, fontsize, graph_colors)
        self._df = df

    def draw(self, cmap, range=[None, None], annot=True):
        sn.heatmap(self._df, annot=annot, ax=self.ax, cmap=cmap,
                   fmt=".2f", vmin=range[0], vmax=range[1])

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        pass


class Table(Visualization):
    def __init__(self, df,  # counts of shape (#counts,)
                 figure_size,
                 ax = None,
                 proxy_artists=None,
                 fontname='DejaVu Sans',
                 fontsize=9,
                 graph_colors=None) -> None:
        super().__init__(figure_size, ax=ax, proxy_artists=proxy_artists, fontname=fontname,
                         fontsize=fontsize, graph_colors=graph_colors)
        self._df = df

    def draw(self):
        table = pd.plotting.table(self.ax, self._df)
        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        pass


class Scatter(Visualization):
    '''A scatter plot showing #num_points points each with a specified color.

    Parameters
    ----------
        x : np.ndarray of shape (#num_points,)
            The x coordinates of the visualization
        y : np.ndarray of shape (#num_points,)
            The y coordinates of the visualization
        color_idx : np.ndarray of shape (#num_points,)
            A list of indices of colors, in which the point will be drawn.
        figure_size : (int,int)
            The size of the figure.
        ax : matplotlib.axes.Axes), optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 

    Example
    -------
    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from validating_models.drawing_utils import Scatter
        >>> # Generate the data
        >>> x = np.linspace(0,2*np.pi,100)
        >>> y_ot = np.sin(x)
        >>> y_gt = np.random.normal(y_ot,0.5)
        >>> labels = ['ot','gt']
        >>> colors = ['r','b']
        >>> color_idx = [0 for i in range(len(x))] + [1 for i in range(len(x))]
        >>> plot = Scatter(np.concatenate((x,x)), np.concatenate((y_ot,y_gt)), color_idx, (2,2))
        >>> plot.draw(colors=colors, labels=labels)
        >>> plot.add_wedge(np.pi,[0,2*np.pi])
        >>> plot.draw_legend()
    '''

    def __init__(self, x, y, color_idx, figure_size, ax = None, proxy_artists=None, fontname='DejaVu Sans', fontsize=14, graph_colors=None) -> None:
        super().__init__(figure_size,
                         ax=ax,
                         proxy_artists=proxy_artists,
                         fontname=fontname,
                         fontsize=fontsize, graph_colors=graph_colors)
        self.x = x
        self.y = y
        self.color_idx = color_idx

    def draw(self, colors, labels):
        '''Draws the Scatter Plot.

        Parameters
        ----------
            colors : list of colors
                The colors coresponding to the color_idx array.
            labels: list of str
                A list of labels corresponding to the label for each color.
        '''
        c = [colors[int(i)] for i in self.color_idx]
        self.ax.scatter(self.x, self.y, c=c, s=5)

        mask = np.where(pd.DataFrame(self.color_idx, columns=['idx']).groupby(['idx'])[
                        'idx'].count().reindex(list(range(len(colors))), fill_value=0).values != 0)[0]
        for color, label in zip(np.array(colors)[mask], np.array(labels)[mask]):
            patch = patches.Patch(color=color, label=label)
            self.proxy_artists.append((patch, label))

    def add_wedge(self, x, x_range, categorical_labels=None):
        '''Adds a highlighted tick at position x given the range of x. When categorical_labels is given the x-Axis is labeled accordingly.

        Parameters
        ----------
            x : double
                The position at which a new tick should be drawn.
            x_range : (double, double)
                The range of the x axis, which is assumed by the given x.
            categorical_labels : dict of (double -> str) assignments
                If set the labels are shown as sets on the limits of the x-Axis. The left side of the x-Axis contains the labels, which have a smaller value then x and the right side the bigger ones.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        coordinates_range = x_range
        self.ax.set_xticks(coordinates_range)
        if categorical_labels != None:
            labels = categorical_labels
        else:
            labels = [myround(value) for value in x_range]
        self.ax.set_xticklabels(labels)
        self.ax.set_xlim(coordinates_range)
        self.draw_wedge(x, x_range)
        return self

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        pass


class Function(Visualization):
    '''A visualization showing the given datasets as graphs. Each graph has a color assigned.

    Parameters
    ----------
        figure_size : (int,int)
            The size of the figure.
        ax : matplotlib.axes.Axes), optional
            Figure and Axes onto which the visualization should be drawn.
        proxy_artists : list of (matplotlib.artist.Artist, str), optional
            Already known components not recognized by matplotlib, which however should be drawn into the legend.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 

    Example
    -------
    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from validating_models.drawing_utils import Function
        >>> # Generate the data
        >>> x = np.linspace(0,2*np.pi,100)
        >>> y_sin = np.sin(x)
        >>> y_cos = np.cos(x)
        >>> labels = ['sin','cos']
        >>> color_idx = [0 for i in range(len(x))] + [1 for i in range(len(x))]
        >>> plot = Function((2,2))
        >>> plot.add_dataset(x, y_sin, 'r', 'sin')
        >>> plot.add_dataset(x, y_cos, 'b', 'cos')
        >>> plot.draw()
        >>> plot.add_wedge(np.pi,[0,2*np.pi])
        >>> plot.draw_legend()
    '''

    def __init__(self, figure_size, ax = None, proxy_artists=None, fontname='DejaVu Sans', fontsize=14, graph_colors=None) -> None:
        super().__init__(figure_size, ax=ax, proxy_artists=proxy_artists, fontname=fontname,
                         fontsize=fontsize, graph_colors=graph_colors)
        self._xs = []
        self._ys = []
        self._colors = []
        self._labels = []

    def add_dataset(self, x, y, color, label):
        '''Add a new dataset to be drawn as a graph with the specified color.

        Parameters
        ----------
            x : numpy.ndarray of shape (#points,)
                The x coordinates of the dataset
            y : numpy.ndarray of shape (#points,)
                The y coordinates of the dataset
            color : str
                The color to be used to draw the graph.
        '''
        self._xs.append(x)
        self._ys.append(y)
        self._colors.append(color)
        self._labels.append(label)

    def draw(self):
        '''Draws the Functions added before with add_dataset.
        '''
        for x, y, color, label in zip(self._xs, self._ys, self._colors, self._labels):
            self.ax.plot(x, y, c=color, label=label)

    def add_wedge(self, x, x_range):
        '''Adds a highlighted tick at position x given the range of x.

        Parameters
        ----------
            x : double
                The position at which a new tick should be drawn.
            x_range : (double, double)
                The range of the x axis, which is assumed by the given x.

        Returns
        -------
            validating_models.drawing_utils.Visualization
                The visualization object.
        '''
        coordinates_range = x_range
        self.ax.set_xticks(coordinates_range)
        labels = [myround(value) for value in x_range]
        self.ax.set_xticklabels(labels)
        self.ax.set_xlim(coordinates_range)
        self.draw_wedge(x, x_range)
        return self

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        pass


class Legend(Visualization):
    '''A visualization to be used when just a legend should be drawn and the handles and labels are given.

    Parameters
    ----------
        handles: list of matplotlib.artists.Artist
            The handles to be included in the legend
        labels : list of str
            The labels of the handles.
        figure_size : (int,int), optional
            The size of the figure.
        fontname : str, optional
            The name of the font to be used.
        fontsize : int, optional
            The size of the font to be used.
        graph_colors : dict, optional
            A dictionary of color options to be set (See dtreeviz documentation). 
    '''

    def __init__(self, handles, labels, figure_size=(1, 1), fontname='DejaVu Sans', fontsize=14, graph_colors=None) -> None:
        super().__init__(figure_size, fontname=fontname,
                         fontsize=fontsize, graph_colors=graph_colors)
        self._entries = {label: handle for label,
                         handle in zip(labels, handles)}

    def draw(self, title=None):
        '''Draws the Legend.

        Parameters
        ----------
            title : str
                The title of the legend
        '''
        labels, handles = zip(*self._entries.items())
        self.ax.legend(handles=handles, labels=labels, title=title,
                       fontsize=self._fontsize, title_fontsize=self._fontsize)
        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

    @staticmethod
    def get_width(number_of_groups, num_bars_per_group, **args):
        pass

    @staticmethod
    def get_height(indices, max_samples_overall, min_samples_overall, **args):
        pass


def extract_labels_handles(plot):
    viz_handles, viz_labels = plot.ax.get_legend_handles_labels()
    result = list(zip(*plot.proxy_artists)) if len(plot.proxy_artists) > 0 else [[], []]
    labels = viz_labels + list(result[1])
    handles = viz_handles + list(result[0])
    return labels, handles

def new_draw_legend(list_labels_handles, fontsize=9):
    labels = []
    handles = []
    for labels_handles in list_labels_handles:
        plot_labels, plot_handles = labels_handles
        labels = labels + plot_labels
        handles = handles + plot_handles
    plot = Legend(handles, labels, fontsize=fontsize)
    plot.draw(title="Constraint Satisfaction")
    return plot

def draw_legend(visualizations, fontsize=9):
    labels = []
    handles = []
    for viz in visualizations:
        viz_handles, viz_labels = viz.ax.get_legend_handles_labels()
        result = list(zip(*viz.proxy_artists)
                      ) if len(viz.proxy_artists) > 0 else [[], []]
        proxy_labels = list(result[1])
        proxy_handles = list(result[0])
        labels = labels + viz_labels + list(proxy_labels)
        handles = handles + viz_handles + list(proxy_handles)
    plot = Legend(handles, labels, fontsize=fontsize)
    plot.draw(title="Constraint Satisfaction")
    return plot