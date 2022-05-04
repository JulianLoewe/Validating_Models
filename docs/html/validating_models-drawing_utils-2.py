import numpy as np
import matplotlib.pyplot as plt
from validating_models.drawing_utils import StackedHistogram
categorical_labels = ['A','B','C','D']
categorical_colors = ['r','b','g','c']
num_bars = 10
data = np.random.rand(len(categorical_labels), num_bars)
plot = StackedHistogram(data,(StackedHistogram.get_width(1, num_bars),3))
bar_labels = [f'bar_{i}' for i in np.arange(num_bars)]
plot.draw(bar_labels, 'Bar Label', categorical_labels=categorical_labels, categorical_colors=categorical_colors)
plot.draw_legend()
