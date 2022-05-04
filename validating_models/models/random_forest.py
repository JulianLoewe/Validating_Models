from sklearn import ensemble
import numpy as np
import pandas as pd
from ..checker import Checker
from ..models.decision_tree import get_shadow_tree_from_checker
from functools import cached_property

def get_shadow_forest_from_checker(model, checker: Checker):
    return ShadowForest(model, checker)

class ShadowForest():
    def __init__(self, model, checker) -> None:
        if not isinstance(model, (ensemble.RandomForestClassifier, ensemble.RandomForestRegressor)):
            raise Exception(f'Model of type {type(model)} not supported!')
        
        self.model = model

        self.checker = checker
        self.X = self.checker.dataset.x_data()

        self.max_samples = model.max_samples
        self.estimators = np.array(model.estimators_)
        if self.max_samples == None:
            self.max_samples = self.X.shape[0]
    
    def predict(self, X):
        return self.model.predict(X)

    @cached_property
    def trees(self):
        return np.array([get_shadow_tree_from_checker(self.estimators[i],self.checker) for i in range(len(self.estimators))])

    def get_bootstrap_indices(self, tree_index):
        estimator = self.estimators[tree_index]
        random_instance = np.random.RandomState(estimator.random_state)
        sample_indices = random_instance.randint(0, self.X.shape[0],self.max_samples)
        return list(sample_indices)

    def get_equal_structured_tree_indizes(self):        
        structures = [np.stack((tree.get_children_left(), tree.get_children_right())) for tree in self.trees]

        # Group by shape of the structures
        shapes = np.array([structure.shape for structure in structures])
        _, inv = np.unique(shapes,axis=0,return_inverse=True)
        size_groups = pd.DataFrame(inv, columns=['labels']).groupby(['labels']).indices
        
        final_groups = {}
        count = 0
        for g, indices in size_groups.items():
            indices = list(indices)
            
            # Finally group same size structures by their structure
            u, new_inv = np.unique(np.stack([structures[i] for i in indices]), axis=0,return_inverse=True)
            child_structure_groups = pd.DataFrame(new_inv, columns=['labels']).groupby(['labels']).indices
            for g2,indices2 in child_structure_groups.items():
                
                # Adding a group for each new occuring structure
                final_groups[count] = [indices[i] for i in indices2]
                count += 1
        return final_groups
        


        

    #     # Check same attributes


        
    #     def get_nodes_from_indices(indices):
    #         return map(lambda x: x.root, self.trees[indices])
    

    #     def get_equal_node_groups(nodes):
    #         return pd.DataFrame([{'split': node.split(), 'feature': node.feature()} for node in nodes]).groupby(['split','feature']).indices

    #     def equal_nodes(indices, nodes):
    #         # Check left
    #         pass

    #         # Check right

