from ..drawing_utils import Scatter
from ..constraint import Constraint, TRUTH_LABELS
from ..checker import Checker
from ..colors import VAL_COLORS, adjust_colors
import numpy as np


def _decode_feature(checker, feature):
    if isinstance(feature, str):
        feature_name = feature
        feature_index = np.where(
            np.array(checker.dataset.feature_names) == feature_name)[0][0]
    else:
        feature_index = int(feature)
        feature_name = checker.dataset.feature_names[feature_index]
    return feature_index, feature_name


def get_feature_target_for_indices_plot(feature,  # str or index
                                        indices,
                                        checker: Checker,
                                        constraint: Constraint,
                                        non_applicable_counts: bool = False,
                                        figsize: tuple = None,
                                        fontsize: int = 14,
                                        fontname: str = "Arial",
                                        title: str = None,
                                        colors: dict = None):  # Only applies when figsize is None
    colors = adjust_colors(colors)
    figsize = (2.5, 1.1) if figsize == None else figsize

    if indices == None:
        indices = list(range(len(checker.dataset)))
    X_data = checker.dataset.x_data()
    y_data = checker.dataset.y_data()

    feature_index, feature_name = _decode_feature(checker, feature)

    # Get X, y data for all samples associated with this node.
    X_feature = X_data[:, feature_index]
    X_indices_feature, y_train = X_feature[indices], y_data[indices]

    constraint_validation_results = checker.get_constraint_validation_result(
        [constraint], non_applicable_counts=non_applicable_counts)[indices, :]
    color_idx = constraint_validation_results + 1
    plot = Scatter(X_indices_feature, y_train, color_idx, figsize)
    plot.draw(colors=[VAL_COLORS[truth_value]
              for truth_value in TRUTH_LABELS], labels=TRUTH_LABELS)

    plot.set_xlabel(feature_name)
    plot.set_ylabel(checker.dataset.target_name)
    return plot


def get_feature_target_plot(feature,
                            checker: Checker,
                            constraint: Constraint,
                            non_applicable_counts: bool = False,
                            figsize: tuple = None,
                            fontsize: int = 14,
                            fontname: str = "Arial",
                            title: str = None,
                            colors: dict = None):  # Only applies when figsize is None

    all_indices = list(range(len(checker.dataset)))
    plot = get_feature_target_for_indices_plot(feature, all_indices, checker=checker, constraint=constraint,
                                               non_applicable_counts=non_applicable_counts, figsize=figsize, fontsize=fontsize, fontname=fontname, title=title, colors=colors)
    preds = checker.predictions
    feature_index, feature_name = _decode_feature(checker, feature)

    x_feature = checker.dataset.x_data()[:, feature_index]

    ind = np.argsort(x_feature)

    plot.ax.plot(x_feature[ind], preds[ind], 'b--')

    return plot

    # PCA: pca: X --> X_transformed
    # Given Regression Function r: pca^-1 o X --> y                   X_transformed --> y
