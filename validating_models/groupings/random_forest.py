from validating_models.models.random_forest import get_shadow_forest_from_checker


def group_by_bootstraped_decision_trees(checker, indices, model, tree_ids=None, **args):

    shadow_forest = get_shadow_forest_from_checker(model, checker)
    n_estimators = len(shadow_forest.estimators)

    indices = set(indices)
    all_groups = {idx: [sample for sample in shadow_forest.get_bootstrap_indices(idx) if sample in indices] for idx in range(n_estimators)}

    if tree_ids != None:
        selected_groups = {idx: all_groups[idx] for idx in tree_ids}
    else:
        selected_groups = all_groups

    return selected_groups, 'Decision Trees'