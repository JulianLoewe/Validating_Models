import numpy as np
import matplotlib.pyplot as plt
from validating_models.drawing_utils import Scatter
# Generate the data
x = np.linspace(0,2*np.pi,100)
y_ot = np.sin(x)
y_gt = np.random.normal(y_ot,0.5)
labels = ['ot','gt']
colors = ['r','b']
color_idx = [0 for i in range(len(x))] + [1 for i in range(len(x))]
plot = Scatter(np.concatenate((x,x)), np.concatenate((y_ot,y_gt)), color_idx, (2,2))
plot.draw(colors=colors, labels=labels)
plot.add_wedge(np.pi,[0,2*np.pi])
plot.draw_legend()
