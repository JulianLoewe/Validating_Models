from dtreeviz.utils import myround
import numpy as np

####################################################################
# Grouping Functions to be used by frequency_distribution_table.py #
# Can be used in case of Regression and Classification             #
####################################################################


def group_by_complete_dataset(checker, indices, **args):
    return {'Complete Dataset': indices}, ''


def group_by_feature(checker, indices, feature, bins=10, f_range=None, simple_groups = True, **args):
    example_selection = checker.indices_to_datasetexamples(
        indices).loc[:, feature].values
    bins = np.histogram_bin_edges(example_selection, bins=bins, range=f_range)
    nbins = len(bins)-1
    example_selection = example_selection.astype(float)
    bins = bins.astype(float)
    bin_indices = np.digitize(example_selection, bins) - 1
    bin_indices[bin_indices == nbins] = nbins - 1

    if simple_groups:
        bin_labels = [str(myround(bin)) for bin in bins[:nbins]]
    else:
        bin_labels = [f'[{myround(i)},{myround(j)})' if j < bins[-1]
                    else f'[{myround(i)},{myround(j)}]' for i, j in zip(bins[:-1], bins[1:])]

    groups = {}
    for label in bin_labels:
        groups[label] = []

    bins_indices = np.column_stack((bin_indices,indices))
    bins_indices = bins_indices[bins_indices[:,0].argsort()]
    unique_bins,unique_indices = np.unique(bins_indices[:, 0], return_index=True)
    indices_grouped = np.split(bins_indices[:,1], unique_indices[1:])
    for bin,i in zip(unique_bins,range(len(indices_grouped))):
        groups[bin_labels[bin]] = indices_grouped[i]

    # for i, bin in enumerate(bin_indices):
    #     groups[bin_labels[bin]].append(indices[i])

    return groups, feature
