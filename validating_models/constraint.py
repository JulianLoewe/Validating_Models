from abc import ABC, abstractmethod
import json
import numpy as np
import pandas as pd
import warnings
from validating_models.stats import get_decorator

TRUTH_VALUE_TO_STRING = {-1: "not applicable", 0: "invalid", 1: "valid"}

TRUTH_VALUES = [-1, 0, 1]

TRUTH_LABELS = [TRUTH_VALUE_TO_STRING[value] for value in TRUTH_VALUES]

class Constraint(ABC):

    def __init__(self,name, shape_schema_dir, target_shape):
        self.shape_schema_dir = shape_schema_dir
        self.target_shape = target_shape
        self.name = name

    @staticmethod
    def get_shacl_identifier(shape_schema_dir, target_shape):
        return f'{shape_schema_dir}_{target_shape}' 
    
    @property
    def shacl_identifier(self):
        return Constraint.get_shacl_identifier(self.shape_schema_dir, self.target_shape)

    @staticmethod
    def _eval_expr(expr: str, predictions: np.ndarray, problem_instances: pd.DataFrame) -> np.ndarray:
        predictions = predictions.squeeze()
        for column_name in problem_instances.columns:
            expr = expr.replace(
                column_name, f'problem_instances[["{column_name}"]].values.squeeze()')
        expr = expr.replace('target', 'predictions')
        
        if expr != '': 
            try:
                result = np.array(
                    eval(expr), dtype=bool)
            except:
                raise Exception(f'Error during the evaluation of {expr}')
        else:
            result = np.ones_like(predictions, dtype=bool)
        return result

    def check_shacl_condition(self, dataset):
        if not self.uses_shacl_constraint:
            # warnings.warn(
            #     f'Shape Network or Target Shape not given for constraint "{self.name}" only using condition!')
            return pd.Series(np.ones((len(dataset), ), dtype=bool))
        else:
            return dataset.get_shacl_schema_validation_results([self])[self.shacl_identifier]

    @property
    def uses_shacl_constraint(self):
        return self.shape_schema_dir != None and self.target_shape != None

    @abstractmethod
    def eval(self, predictions: np.ndarray, dataset, pre_evaluated_expr: np.ndarray = None) -> pd.DataFrame:
        pass

class PredictionConstraint(Constraint):
    """ A constraint coupels the validation of a knowledge graph with a logical expression about the target of a predictive model.

    A constraint is of the form :math:`S_{ts} \rightsquigarrow \sigma`, where
        * S denotes the Shape Network and ts the target shape.
        * expr is a logical expression formulated in python involving the target variable "target", which should get predicted by a model.

    :param str shape_schema_dir: The directory of the shape network
    :param str expr: A logical expression involving "target".
    """

    def __init__(self, name: str,  expr: str, shape_schema_dir: str = None, target_shape: str = None, condition: str = '') -> None:
        super().__init__(name, shape_schema_dir, target_shape)
        self.condition = condition
        self.expr = expr
    
    @property
    def uses_target(self):
        return 'target' in self.expr or 'target' in self.condition

    def uses_feature(self, column_name):
        return column_name in self.expr or column_name in self.condition

    def eval_expr(self, predictions: np.ndarray, dataset):
        return self._eval_expr(self.expr, predictions, dataset.x_data(df=True))

    def eval(self, predictions: np.ndarray, dataset, pre_evaluated_expr: np.ndarray = None) -> pd.DataFrame:
        """
        Evaluates the Constraint according to the semantics of ^v-> in section 3.2.2.

        Parameters:
            val_results : ndarray
                For each datapoint a validation result
            predictions : ndarray
                For each datapoint x the prediction of the model M_theta(x)
        """
        val_results = self.check_shacl_condition(dataset).fillna(value=True, inplace=False).values
        problem_instances = dataset.x_data(df=True)
        evaluation_result = np.zeros_like(val_results, dtype=int)
        
        # Evaluate left-hand side of the constraint
        evaluated_cond = self._eval_expr(self.condition,predictions,problem_instances)
        val_results = evaluated_cond & val_results
        val_results = val_results.astype(bool)

        # Evaluate right-hand side of the constraint
        if not isinstance(pre_evaluated_expr, np.ndarray):
            evaluated_expr = self._eval_expr(self.expr,predictions,problem_instances)
        else:
            evaluated_expr = pre_evaluated_expr

        evaluation_result[evaluated_expr] = 1
        evaluation_result[~val_results] = -1
        return pd.DataFrame(data=evaluation_result, columns=[self.name])


class ShaclSchemaConstraint(Constraint):
    def __init__(self, name: str, shape_schema_dir: str = None, target_shape: str = None) -> None:
        super().__init__(name, shape_schema_dir, target_shape)
    
    def eval(self, predictions: np.ndarray, dataset, pre_evaluated_expr: np.ndarray = None) -> pd.DataFrame:
        val_results = self.check_shacl_condition(dataset)
        val_results = val_results.map({False: 0, True: 1, np.nan: -1})
        return val_results

    @property
    def uses_target(self):
        return False

    def uses_feature(self, column_name):
        return False
    

    @staticmethod
    def from_dict(input):
        if not input['inverted']:
            return ShaclSchemaConstraint(input['name'], shape_schema_dir=input['shape_schema_dir'], target_shape=input['target_shape'])
        else: 
            return InvertedShaclSchemaConstraint(input['name'], shape_schema_dir=input['shape_schema_dir'], target_shape=input['target_shape'])

class InvertedShaclSchemaConstraint(ShaclSchemaConstraint):
    def __init__(self, name: str, shape_schema_dir: str = None, target_shape: str = None) -> None:
        super().__init__(name, shape_schema_dir, target_shape)
    
    def eval(self, predictions: np.ndarray, dataset, pre_evaluated_expr: np.ndarray = None) -> pd.DataFrame:
        not_inverted_result = super().eval(predictions, dataset, pre_evaluated_expr)
        val_results = not_inverted_result.map({0:1, 1:0, -1:-1})
        return val_results

class InvertedPredictionConstraint(PredictionConstraint):
    def __init__(self, name: str, expr: str, shape_schema_dir: str = None, target_shape: str = None, condition: str = '') -> None:
        super().__init__(name, expr, shape_schema_dir, target_shape, condition)
    
    def eval(self, predictions: np.ndarray, dataset, pre_evaluated_expr: np.ndarray = None) -> pd.DataFrame:
        not_inverted_result = super().eval(predictions, dataset, pre_evaluated_expr)
        val_results = not_inverted_result.map({0:1, 1:0, -1:-1})
        return val_results
