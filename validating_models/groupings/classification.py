import numpy as np
####################################################################
# Grouping Functions to be used by frequency_distribution_table.py #
# Can be used in case of Classification                            #
####################################################################

def group_by_gt_class(checker, indices, **args):
    dataset = checker.dataset
    example_selection = checker.indices_to_datasetexamples(indices)
    example_selection.set_index(np.array(indices),drop=True, inplace=True)
    groups = {}
    for class_id, class_name in dataset.class_names.items():
        groups[class_name] = list(
            example_selection.loc[example_selection[dataset.target_name] == class_id].index)
    return groups, 'gt class'


def group_by_predicted_class(checker, indices, **args):
    dataset = checker.dataset
    example_selection = checker.indices_to_datasetexamples(indices)
    example_selection['predictions'] = checker.predictions[indices]
    example_selection.set_index(np.array(indices),drop=True, inplace=True)
    groups = {}
    for class_id, class_name in dataset.class_names.items():
        groups[class_name] = list(
            example_selection.loc[example_selection['predictions'] == class_id].index)
    return groups, 'predicted class'