import numpy as np
import matplotlib.pyplot as plt
from validating_models.drawing_utils import Function
# Generate the data
x = np.linspace(0,2*np.pi,100)
y_sin = np.sin(x)
y_cos = np.cos(x)
labels = ['sin','cos']
color_idx = [0 for i in range(len(x))] + [1 for i in range(len(x))]
plot = Function((2,2))
plot.add_dataset(x, y_sin, 'r', 'sin')
plot.add_dataset(x, y_cos, 'b', 'cos')
plot.draw()
plot.add_wedge(np.pi,[0,2*np.pi])
plot.draw_legend()
