from types import FunctionType
from typing import List
import itertools

from .dataset import Dataset
from .constraint import Constraint, TRUTH_VALUE_TO_STRING, PredictionConstraint
import numpy as np
import pandas as pd
from validating_models.stats import get_decorator, measure_time
from functools import cached_property

time_generate_fdt = get_decorator('summarization')
time_overall_constraint_evaluation = get_decorator('overall_constraint_evaluation')

class Checker:
    '''Class used to validate constraints against prediction results given a dataset or against the targets (ground truth) in the dataset given the dataset.

    Attributes
    ----------
        predict : FunctionType
            Function representing the Model: :math:`M_{\\theta}: \mathbb{I} \\to \mathbb{T}`
        dataset : validating_models.dataset.Dataset
            The dataset :math:`D = ((\mathbf{x}_i, t_i) \mid i \in [1,...,N])` with the samples :math:`(\mathbf{x}_i, t_i)`            

    Parameters
    ----------
        predict : FunctionType, optional
            Function representing the Model: :math:`M_\\theta: \mathbb{I} \\to \mathbb{T}`. Can be None if use_gt is set to True.
        dataset : validating_models.dataset.Dataset
            The dataset with the samples (:math:`((\mathbf{x}_i, t_i) \mid i \in [1,...,N])`)
        use_gt : bool, optional
            Whether to validate the constraints against the ground truth in the dataset.
    '''

    def __init__(self, predict: FunctionType, dataset: Dataset, use_gt=False) -> None:
        self.predict = predict
        self.dataset: Dataset = dataset

        self.constraint_validation_results = pd.DataFrame(columns=[])
        self._coverage_results_cache = {}
        self._use_gt = use_gt
        self._pre_evaluated_expressions = pd.DataFrame(columns=[])

    @cached_property
    def predictions(self):
        '''The predictions :math:`M_\\theta(\mathbf{x}_i)` made by predict over the samples of the dataset.

        Returns
        -------
        numpy.ndarray
        '''
        return self.predict(self.dataset.x_data())

    def indices_to_datasetexamples(self, indices):
        return self.dataset.df.iloc[indices, :]
    
    def get_target(self, df=False):
        return self.predictions if not self._use_gt else self.dataset.y_data(df=True)
    
    def explain_validation_results(self, constraint, task_example_indices = None, nodes = None, indices = None):
        self.validate([constraint])

        explaination = pd.DataFrame()
        
        # SHACL Schema used
        if constraint.uses_shacl_constraint:
            shacl_validation_result = self.dataset.get_shacl_schema_validation_results([constraint], checker=self)[constraint.shacl_identifier]
            explaination['SHACL Validation Result'] = shacl_validation_result # Dataset Index Included
            mapping = self.dataset.get_sample_to_node_mapping()
            explaination['seednode'] = mapping

        # Features used
        for column_name in self.dataset.df.columns:
            if constraint.uses_feature(column_name):
                explaination[column_name] = self.dataset.df[column_name] # Dataset Index Included

        # If target is used
        if constraint.uses_target:
            if not self._use_gt and len(explaination) == 0:
                explaination['prediction'] = self.dataset.y_data(df=True) # Last Chance to recover Dataset Index

            target = self.get_target(df=True)
            explaination['predictions' if not self._use_gt else 'gt'] = target # Dataset Index NOT Included
        
        explaination['Constraint Validation Result'] = self.constraint_validation_results[constraint.identifier].values


        if task_example_indices != None:
            explaination = explaination.loc[task_example_indices,:]
        
        if nodes != None:
            explaination = explaination.query("seednode in @nodes")


        if indices != None:
            explaination = explaination.iloc[indices,:]
        
        return explaination
    
    def pre_evaluate_expression(self, constraint: Constraint, target, problem_instances) -> np.ndarray:
        constraint_identifier = constraint.identifier
        if not isinstance(constraint, PredictionConstraint):
            return None
        if constraint_identifier not in self._pre_evaluated_expressions:
            self._pre_evaluated_expressions.loc[:,constraint_identifier] = constraint.eval_expr(target, problem_instances)
        return self._pre_evaluated_expressions[constraint_identifier].values


    @time_overall_constraint_evaluation
    def validate(self, constraints: List[Constraint]):
        # Perform the necessary SHACL Shape Schema Validation + Join to get the shacl validation results per sample in the dataset
        constraint_need_shacl_results = [constraint for constraint in constraints if constraint.uses_shacl_constraint]

        shacl_schema_validation_results = self.dataset.get_shacl_schema_validation_results(constraint_need_shacl_results, checker = self)

        # Model Inference or getting the ground truth values
        with measure_time('model_inference'):
            target = self.get_target()

        # Get the problem instances to evaluate constraint expression making use of feature values
        problem_instances = self.dataset.x_data(df=True)

        for constraint in constraints:
            if constraint.identifier not in self.constraint_validation_results:
                pre_evaluated_expression = self.pre_evaluate_expression(constraint, target, problem_instances)
                self.constraint_validation_results.loc[:, constraint.identifier] = constraint.eval(target, shacl_schema_validation_results, problem_instances, pre_evaluated_expr=pre_evaluated_expression )

    def get_constraint_validation_result(self, constraints, non_applicable_counts=False, df=False, only_cached_results=False):
        '''Gives the constraint validation results for a given set of constraints

        Parameters
        ----------
            constraints : list of :py:class:`validating_models.constraint.Constraint`
                The constraints to be validated.
            non_applicable_counts : bool, optional
                If set to true, a three-valued logic is used. Task-examples not satisfing the shape schema of the constraint can be differentiated from the ones satisfing the shape schema and the logical expression defined in the constraint. The first ones are marked as non_applicable and only the later ones are marked as valid. If set to false there is only valid or invalid. The constraints are treated as implication with the normal two-valued logic.
            only_cached_results : bool, optional
                When set to true constraints are assumed to be already validated once, with this checker instance. There the validation step can be skipped. Defaults to False.

        Returns
        -------
            np.ndarry of shape (#instances, #constraints)
                The validation results.
        '''
        if not only_cached_results:
            self.validate(constraints)

        constraint_identifiers = [constraint.identifier for constraint in constraints]
        validation_result_df = self.constraint_validation_results[constraint_identifiers].copy(
            deep=True)

        if not non_applicable_counts:
            validation_result_df = validation_result_df.abs()

        if df:
            return validation_result_df
        else:
            return validation_result_df.values

    def get_valid_invalid_counts_for_indices(self, constraints: list, indices=None, non_applicable_counts=False, only_cached_results = False):
        '''Returns three counts for each constraint: the count of valid instances, the count of invalid instances and optionally the count of non applicable instances. The instances are choosen according to the given indices of the dataset.

        Parameters
        ----------
            constraints : list of :py:class:`validating_models.constraint.Constraint`
                The list of constraints
            indices : list of int, optional
                The list of indices to select the instances of the dataset. None will select all instances in the dataset.
            non_applicable_counts : bool, optional
                If set to true, a three-valued logic is used. Task-examples not satisfing the shape schema of the constraint can be differentiated from the ones satisfing the shape schema and the logical expression defined in the constraint. The first ones are marked as non_applicable and only the later ones are marked as valid. If set to false there is only valid or invalid. The constraints are treated as implication with the normal two-valued logic.
            only_cached_results : bool, optional
                When set to true constraints are assumed to be already validated once, with this checker instance. There the validation step can be skipped. Defaults to False.

        Returns
        -------
            tuple (valid, invalid, non_applicable) - counts. Each count of shape (#constraints,).
                the counts
        '''
        if indices != None:
            constraint_validation_result = self.get_constraint_validation_result(
                constraints, non_applicable_counts=non_applicable_counts, only_cached_results=only_cached_results)[indices, :]  # shape: (len(indices), len(constraints))
        else:
            constraint_validation_result = self.get_constraint_validation_result(
                constraints, non_applicable_counts=non_applicable_counts, only_cached_results=only_cached_results)

        valid_count = np.count_nonzero(
            constraint_validation_result == 1, axis=0)
        invalid_count = np.count_nonzero(
            constraint_validation_result == 0, axis=0)
        non_applicable_count = np.count_nonzero(
            constraint_validation_result == -1, axis=0)

        return valid_count, invalid_count, non_applicable_count

    def get_valid_invalid_counts_for_array_of_indices(self, constraints: list, array_of_indices: list, non_applicable_counts=False, only_cached_results = False):
        '''Returns three counts for each constraint and for each array of indices: the count of valid instances, the count of invalid instances and optionally the count of non applicable instances. 

        Parameters
        ----------
            constraints : list of :py:class:`validating_models.constraint.Constraint`
                The list of constraints
            array of indices : list of lists of int
                A list of lists of indices to select the instances of the dataset. Per list of indices the counts are returned.
            non_applicable_counts : bool, optional
                If set to true, a three-valued logic is used. Task-examples not satisfing the shape schema of the constraint can be differentiated from the ones satisfing the shape schema and the logical expression defined in the constraint. The first ones are marked as non_applicable and only the later ones are marked as valid. If set to false there is only valid or invalid. The constraints are treated as implication with the normal two-valued logic.
            only_cached_results : bool, optional
                When set to true constraints are assumed to be already validated once, with this checker instance. There the validation step can be skipped. Defaults to False.

        Returns
        -------
            tuple (valid, invalid, non_aplicable) - counts. Each count of shape (#lists, #constraints).
                the counts
        '''
        valid_count = np.zeros((len(array_of_indices), len(constraints)))
        invalid_count = np.zeros((len(array_of_indices), len(constraints)))
        non_applicable_count = np.zeros(
            (len(array_of_indices), len(constraints)))

        for i, indices in enumerate(array_of_indices):
            valid_count[i, :], invalid_count[i, :], non_applicable_count[i, :] = self.get_valid_invalid_counts_for_indices(
                constraints, indices, non_applicable_counts=non_applicable_counts, only_cached_results=only_cached_results)

        return valid_count, invalid_count, non_applicable_count

    def get_validation_coverage(self, constraints: list, not_covered=True, only_cached_results=False):
        '''
        coverage: instances models constraints[i] < instances models constraints[i-1] < instances not models constraint[i] < instances not models constraint[i-1]
        '''
        cache_key = str([constraint.identifier for constraint in constraints]) + str(not_covered)
        
        if cache_key in self._coverage_results_cache:
            return self._coverage_results_cache[cache_key]

        if not only_cached_results:
            self.validate(constraints)
        validation_results = self.get_constraint_validation_result(list(reversed(constraints)), non_applicable_counts=not_covered, only_cached_results=only_cached_results)
        valid_mask = (validation_results == 1)
        invalid_mask = (validation_results == 0)

        # for each instance, (validation, constraint_id)
        coverage = -1 * np.ones((len(self.constraint_validation_results), 2))

        if not_covered == True:
            # retrieve valid instances
            for i,constraint in enumerate(reversed(constraints)):
                valid_mask_constraint = valid_mask[:,i].squeeze()
                coverage[valid_mask_constraint, :] = np.array((1,i))

        # retrieve invalid instances
        for i,constraint in enumerate(reversed(constraints)):
            invalid_mask_constraint = invalid_mask[:,i].squeeze()
            coverage[invalid_mask_constraint, :] = np.array((0,i))

        coverage = pd.DataFrame(coverage, columns=['val', 'constraint_name'])
        constraint_id_mapping = {float(i): constraint.name for i,constraint in enumerate(reversed(constraints))}
        self._coverage_results_cache[cache_key] = (coverage, constraint_id_mapping)
        return coverage, constraint_id_mapping

    def get_coverage_counts_for_indices(self, constraints: list, indices: list, not_covered=True, only_cached_results=False):
        coverage, constraint_id_mapping = self.get_validation_coverage(
            constraints, not_covered=not_covered, only_cached_results=only_cached_results)
        coverage = coverage.loc[indices, :]
        index = pd.MultiIndex.from_product(
            [[0, 1], [float(i) for i in range(len(constraints))]])
        counts = coverage.groupby(['val', 'constraint_name'])['val'].count()
        if (-1, -1) in counts:
            result =  counts[-1, -1], counts.reindex(index, fill_value=0).unstack(fill_value=0), constraint_id_mapping
        else:
            result = 0, counts.reindex(index, fill_value=0).unstack(fill_value=0), constraint_id_mapping
        return result

    def get_coverage_counts_for_array_of_indices(self, constraints: list, array_of_indices: list, not_covered=True, only_cached_results=True):
        coverages = []
        for indices in array_of_indices:
            not_cov, counts, constraint_id_mapping = self.get_coverage_counts_for_indices(
                constraints, indices, not_covered=not_covered, only_cached_results=only_cached_results)
            coverages.append((not_cov, counts))
        return coverages, constraint_id_mapping

    @time_generate_fdt
    def get_fdt(self, constraints, indices, group_functions: List[FunctionType], coverage=False, non_applicable_counts=False, only_cached_results=False, **args) -> pd.DataFrame:

        if not only_cached_results:
            self.validate(constraints)

        if indices == None:
            indices = list(range(len(self.dataset)))

        # Apply the group functions
        list_of_array_of_indices = []
        group_descriptors = []
        list_of_group_names = []
        for group_fkt in group_functions:
            # groups: dict (group_name -> list of indices)
            groups, group_descriptor = group_fkt(self, indices, **args)
            group_names, array_of_indices = zip(*groups.items())
            list_of_group_names.append(group_names)
            group_descriptors.append(group_descriptor)
            list_of_array_of_indices.append(array_of_indices)

        # Calculate the cartesian product of the group combinations and calculate the related indices in the dataset
        group_index = pd.MultiIndex.from_product(
            list_of_group_names, names=group_descriptors)

        final_array_of_indices = []
        for list_of_indices in itertools.product(*list_of_array_of_indices):
            final_array_of_indices.append(list(set.intersection(
                *[set(indices) for indices in list_of_indices])))

        if coverage:
            list_of_coverage_results, constraint_id_mapping = self.get_coverage_counts_for_array_of_indices(
                constraints, final_array_of_indices, not_covered=non_applicable_counts, only_cached_results=True)  # list of
            categories, _ = zip(
                *list_of_coverage_results[0][1].stack().items())
            label_constraints = [constraint_id_mapping[label[1]] for label in categories] + ['']
            label_valres = [TRUTH_VALUE_TO_STRING[label[0]]
                            for label in categories] + (['not covered'] if non_applicable_counts else ['valid'])
            counts = np.zeros((len(group_index), len(label_valres)))
            for i, coverage_result in enumerate(list_of_coverage_results):
                counts[i, :] = np.hstack(
                    (coverage_result[1].stack().values, np.array(coverage_result[0])))
            result_df = pd.DataFrame(data=counts, index=group_index, columns=pd.MultiIndex.from_arrays([label_constraints, label_valres], names=[
                                     'Constraints', 'Validation Results'], sortorder=None))  # pd.Index(group_names, name='Groups')

        else:
            counts = np.stack(self.get_valid_invalid_counts_for_array_of_indices(constraints, final_array_of_indices,
                              non_applicable_counts=non_applicable_counts, only_cached_results=True))  # shape: (#categories, #groups, #constraints)
            # shape: (#groups, #constraints, #categories)
            counts = np.transpose(counts, axes=(1, 2, 0))
            counts = counts.reshape((len(group_index), len(constraints) * 3))

            # Create DataFrame with groups as index and multidimensional columns
            result_df = pd.DataFrame(data=counts, index=group_index, columns=pd.MultiIndex.from_product(
                ([constraint.name for constraint in constraints], ['valid', 'invalid', 'not applicable']), names=['Constraints', 'Validation Results']))

        return result_df

class ConstantModelChecker(Checker):
    def __init__(self, constant_prediction, dataset: Dataset, use_gt=False) -> None:
        def predict(array): return np.ones(
                    (array.shape[0],)) * constant_prediction
        super().__init__(predict, dataset, use_gt=use_gt)

class DecisionNodeChecker(ConstantModelChecker):
    def __init__(self, node, dataset: Dataset, use_gt=False) -> None:
        super().__init__(node.shadow_tree.get_prediction(node.id), dataset, use_gt)

