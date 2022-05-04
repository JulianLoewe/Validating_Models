import numpy as np
import matplotlib.pyplot as plt
from validating_models.drawing_utils import GroupedStackedHistogram
categorical_labels = ['A','B','C','D']
categorical_colors = ['r','b','g','c']
num_groups = 3
num_bars_per_group = 2
data = np.random.rand(len(categorical_labels), num_groups, num_bars_per_group)
figure_size = (GroupedStackedHistogram.get_width(num_groups, num_bars_per_group), 3)
plot = GroupedStackedHistogram([data[0,:,:],data[1,:,:],data[2,:,:],data[3,:,:]],figure_size)
bar_labels = [f'bar_{i}' for i in np.arange(num_bars_per_group)]
group_labels = [f'group_{j}' for j in np.arange(num_groups)]
plot.draw(bar_labels, 'Bar Label', group_labels=group_labels, categorical_labels=categorical_labels, categorical_colors=categorical_colors)
plot.draw_legend()
