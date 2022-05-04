import numpy as np
import matplotlib.pyplot as plt
from validating_models.drawing_utils import PieChart
categorical_labels = ['A','B','C','D']
categorical_colors = ['r','b','g','c']
data = np.array([5,4,3,2])
plot = PieChart(data, (3,3))
plot.draw(colors=categorical_colors, text='Some text', labels=categorical_labels)
plot.draw_legend()
