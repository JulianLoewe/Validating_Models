from dtreeviz.colors import COLORS
import matplotlib.pyplot as plt

VAL_COLORS = {
    'valid': 'g',
    'invalid': 'r',
    'not applicable': 'grey'
}

MOD_COLORS = dict(COLORS, **VAL_COLORS)


def adjust_colors(colors):
    if colors is None:
        return MOD_COLORS
    return dict(MOD_COLORS, **colors)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
