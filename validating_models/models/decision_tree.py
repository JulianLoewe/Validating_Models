from dtreeviz.models.shadow_decision_tree import ShadowDecTree, ShadowDecTreeNode
from validating_models.stats import get_decorator, get_hyperparameter_value, new_entry
from validating_models.logger import get_logger

logger = get_logger('validating_models.models.decision_tree')
time_node_samples = get_decorator('node_samples')

def get_shadow_tree_from_checker(model, checker, tree_index=None) -> ShadowDecTree:
    dataset = checker.dataset
    return ShadowDecTree.get_shadow_tree(model, X_train=dataset.x_data(), y_train=dataset.y_data(), feature_names=dataset.feature_names, target_name=dataset.target_name, class_names=dataset.class_names, tree_index=tree_index)


def _get_single_node_samples(node: ShadowDecTreeNode):
    '''
    Fast methode to get the samples of a single node.
    '''
    model = node.shadow_tree.tree_model
    x_data = node.shadow_tree.X_train
    paths = model.decision_path(x_data)
    return list(paths[:,node.id].nonzero()[0])

def get_single_node_samples(node: ShadowDecTreeNode, only_calculate_single_node=False):
    if only_calculate_single_node:
        return _get_single_node_samples(node)
    else:
        tree = node.shadow_tree
        return get_node_samples(tree)[node.id]


@time_node_samples
def get_node_samples(tree: ShadowDecTree):
    result = None
    if tree.node_to_samples is not None:
        logger.info('Reusing Node_samples!')
        result = tree.node_to_samples
    else:
        if get_hyperparameter_value('node_to_samples_non_optimized'):
            logger.info('Using non optimized node_samples on purpose!')
            result = tree.get_node_samples()
        else:
            try:
                logger.info('Calculating node samples!')
                dec_paths = tree.tree_model.decision_path(tree.X_train)
                dec_paths = dec_paths.tocsc()

                n_nodes = dec_paths.shape[1]
                node_to_samples = {}
                for node_id in range(n_nodes):
                    node_to_samples[node_id] = list(dec_paths[:,node_id].nonzero()[0])
                tree.node_to_samples = node_to_samples
                result = tree.node_to_samples
            except:
                result = tree.get_node_samples()
    new_entry('n_dnodes',len(result))
    new_entry('n_leaves',len(tree.leaves))
    new_entry('n_splitnodes',len(tree.internal))
    return result
