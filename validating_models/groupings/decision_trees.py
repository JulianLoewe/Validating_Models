from ..groupings.general import group_by_feature
from ..models.decision_tree import get_shadow_tree_from_checker, get_node_samples

####################################################################
# Grouping Functions to be used by frequency_distribution_table.py #
# Can be used in case of Regression and Classification             #
####################################################################

def group_by_decision_tree_nodes(checker, indices, model, node_ids=None, **args):
    # if not 'model' in args:
    #    raise Exception('The decision tree model is required to be provided with the parameter "model".')
    # model = args['model']

    indices = set(indices)
    shadow_tree = get_shadow_tree_from_checker(model, checker)

    all_groups = {str(node_id): [sample for sample in samples if sample in indices] for node_id,
                  samples in get_node_samples(shadow_tree).items()}

    if node_ids != None:
        selected_groups = {
            str(node_id): all_groups[str(node_id)] for node_id in node_ids}
    else:
        selected_groups = all_groups

    return selected_groups, 'Nodes'


def group_by_decision_tree_leaves(checker, indices, model, **args):

    shadow_tree = get_shadow_tree_from_checker(model, checker)

    leaf_node_ids = [node.id for node in shadow_tree.leaves]

    groups, _ = group_by_decision_tree_nodes(
        checker, indices, model, node_ids=leaf_node_ids)

    return groups, 'Leaves'

def group_by_node_split_feature(checker, indices, node, **args):
    feature_name = node.feature_name()
    groups, _ = group_by_feature(checker, indices, feature_name, f_range=checker.dataset.feature_range(feature_name))
    return groups, feature_name