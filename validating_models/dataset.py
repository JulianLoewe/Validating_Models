from abc import ABC, abstractmethod
from functools import lru_cache
import itertools
import json
from pathlib import Path

from .constraint import Constraint, InvertedPredictionConstraint, InvertedShaclSchemaConstraint, PredictionConstraint, ShaclSchemaConstraint
from .shacl_validation_engine import Communicator,ReducedTravshaclCommunicator

import pandas as pd
import numpy as np
from sklearn.utils import Bunch
import re

from typing import Union, List, Mapping

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.plugins import sparql

from validating_models.stats import get_decorator, get_hyperparameter_value, new_entry

time_join = get_decorator('join')
time_shacl_schema_validation = get_decorator('shacl_schema_validation')

MAX_INSTANCES_IN_FILTER = 200
DATATYPE_TO_PANDAS_TYPE = {
    'http://www.w3.org/2001/XMLSchema#int': 'numeric',
    'http://www.w3.org/2001/XMLSchema#boolean': 'bool'
}

TYPE_TO_PANDAS_TYPE = {
    'uri': 'category',
    'literal': 'category'
}

class Dataset(ABC):
    """Abstract representation of a dataset, including the most important parameters about the dataset.

    A dataset used for suppervised learning consists of samples (x,y), where x is a set of features (the problem instance) and y is a label (the target).

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset having the features and the target as columns.
    target_name : str
        The name of the target.
    categorical_mapping : mapping of str to (mapping of int to str)
        Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target_name: str,
                 categorical_mapping: Mapping[str, Mapping[int, str]] = {}):

        super().__init__()
        self.df = df
        self.target_name = target_name
        self.categorical_mapping = categorical_mapping

        # Target is always numerical
        # self.categorical_mapping[self.target_name] = {float(i):class_name for i, class_name in self.class_names.items()}

        self._feature_range_cache = {}
        self.feature_names = list(self.df.columns)
        if self.target_name in self.feature_names:
            self.feature_names = self.feature_names.remove(self.target_name)

    def x_data(self, df=False):
        """Returns the problem instances of the dataset (e.g. the dataset without the target)

        Parameters
        ----------
        df : bool
            Whethere to return a pandas.Dataframe.
        
        Returns
        -------
        numpy.ndarray or pandas.Dataframe
        """
        if df:
            return self.df[self.feature_names]
        else:
            return self.df[self.feature_names].values

    def y_data(self, df=False):
        """Returns the target values of the dataset (e.g. the dataset without the features)

        Parameters
        ----------
        df : bool
            Whethere to return a pandas.Dataframe.
        
        Returns
        -------
        numpy.ndarray or pandas.Dataframe
        """
        if df:
            return self.df[[self.target_name]]
        else:
            return self.df[[self.target_name]].values

    def __len__(self):
        """Returns the number of samples in the dataset

        Returns
        -------
        int
            The number of samples in the dataset
        """
        return len(self.df)

    @property
    def class_names(self):
        """The names of the classes of the target. In case of a regression task, this will return all unique values of the target.

        Returns
        -------
        list of string
            The names of the classes of the target.
        """
        if self.target_name in self.categorical_mapping:
            return self.categorical_mapping[self.target_name]
        else:
            return {i: class_name for i, class_name in enumerate(np.unique(self.y_data()))}

    def feature_range(self, feature):
        """The numerical range of the given feature.

        Parameters
        ----------
        feature : str
            The feature

        Returns
        -------
        [numerical, numerical]
            The numerical range of the given feature.
        """
        if feature not in self._feature_range_cache:
            max = self.df.loc[:, feature].max()
            min = self.df.loc[:, feature].min()
            self._feature_range_cache[feature] = [min, max]
        return self._feature_range_cache[feature]

    def categorical_split(self, feature, x):
        """Given the categorical mapping for the given feature two strings are returned. The first representing the categories assigned to values below x and the other categories above x.

        Parameters
        ----------
        feature : str
            The feature
        x : float
            The split value
        
        Returns
        -------
        str, str
        """
        if not feature in self.categorical_mapping:
            return None
        mapping = self.categorical_mapping[feature]
        left_labels = [str(label)
                       for value, label in mapping.items() if value <= x]
        right_labels = [str(label)
                        for value, label in mapping.items() if value > x]
        if len(left_labels) > 1:
            left = "{" + ",".join(left_labels) + "}"
        else:
            left = left_labels[0]

        if len(right_labels) > 1:
            right = "{" + ",".join([str(label) for value,
                                   label in mapping.items() if value > x]) + "}"
        else:
            right = right_labels[0]
        return left, right

    def is_categorical(self, feature):
        """Whether a feature is included in the categorical mapping.

        Parameters
        ----------
        feature : str
            The feature
        """
        return feature in self.categorical_mapping
    
    @abstractmethod
    def get_sample_to_node_mapping(self):
        pass

    # @staticmethod
    # def from_sklearn_dataset(data: Bunch, target_name, **args):
    #     return Dataset(x_data=data.data, y_data=data.target, feature_names=data.feature_names, target_name=target_name, class_names=data.target_names, **args)

    # @staticmethod
    # def from_openml_dataset(id, target: str = None, **args):
    #     dataset = openml.datasets.get_dataset(id)

    #     if target == None:
    #         target = dataset.default_target_attribute

    #     X, y, categorical_indicator, attribute_names = dataset.get_data(
    #         dataset_format="dataframe", target=dataset.default_target_attribute)
    #     df = pd.merge(X, y, left_index=True, right_index=True, validate='1:1')
    #     for cat, name in zip(categorical_indicator, attribute_names):
    #         if cat:
    #             df[name] = df[name].astype('category')
    #     return Dataset(df=df, target_name=target, **args)
    
    def convert_to_numerical(self, column):
        """Given the name of a column in the dataset, the column is converted inplace to a numerical. This methode also updates the cateogical_mapping attribute.

        Parameters
        ----------
        column : str
            The name of the column to be converted
        """
        if self.df[column].dtype == 'category':
            uniques = self.df[column].cat.categories
        else:
            uniques = self.df[column].unique()

        if np.array([str.isnumeric(entry) for entry in uniques], dtype=bool).all():
            self.df[column] = pd.to_numeric(self.df[column], errors='ignore')
        else:
            self.df[column] = self.df[column].astype(pd.CategoricalDtype(categories=uniques))
        
        mapping = {column: dict(zip(self.df[column].cat.codes, self.df[column]))}
        self.df[column] = self.df[column].apply(lambda x: x.cat.codes)
        self.categorical_mapping.update(mapping)


class BaseDataset(Dataset):
    """
    The BaseDataset extends the Dataset by providing the capabilities to create the dataset based on a knowledge graph and by doing so extract a mapping from each sample to a seed node (IRI / URI) in the knowledge graph.
    As the SHACL validation should be performed over the minimal number of instances, it's useful to have a query only returning the IRIs of the seed nodes in the dataset. That is the seed_query.
    The seed_query refers to the seed nodes with the variable called seed_var. 
    The SHACL validation is performed via a communicator and can be turned to random by setting random_schema_validation to true.

    Attributes
    ----------
    sample_to_node_mapping : pandas.DataFrame
        A DataFrame(*idx*, node_id) having the same index idx as the dataframe df storing the dataset and a column with the seed_var as name including all the seed node identifiers (IRI / URI). When SHACL validation is performed the joined SHACL valiation results, will be stored here.
    shacl_validation_results : dict of pandas.DataFrame
        When validating a shacl schema the results are stored here. Entries are of the form f'{shacl_schema_directory}_{target_shape}'. The results are stored separate to avoid joining the different validation results for each seed node.
        Each entry is a DataFrame(*node_id*, validation_result) 
    seed_var : str
        The variable in seed_query and in the sample_to_node_mapping referring to the seed nodes
    random_schema_validation : bool
        Whether random results are used instead of the SHACL valiation results.
    communicator : validating_models.shacl_validation_engine.Communicator
        The communicator object used to communication with the SHACL validation engine.
    seed_query : str
        The SPARQL query returning the seed nodes used to create the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset having the features and the target as columns.
    target_name : str
        The name of the target.
    categorical_mapping : mapping of str to (mapping of int to str), optional
        Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)
    seed_query : str
        The SPARQL query returning the seed nodes used to create the dataset.
    seed_var : str
        The variable in seed_query and in the sample_to_node_mapping referring to the seed nodes
    sample_to_node_mapping : pandas.DataFrame
        A DataFrame(*idx*, node_id) having the same index idx as the dataframe df storing the dataset and a column with the seed_var as name including all the seed node identifiers (IRI / URI).
    communicator : validating_models.shacl_validation_engine.Communicator
        The communicator object to use for the communication with an SHACL validation engine
    random_schema_validation : bool
        Whether to use random results instead of the SHACL valiation results.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target_name: str,
                 seed_query: str,
                 seed_var: str,
                 sample_to_node_mapping: pd.DataFrame,
                 communicator: Communicator = None,
                 random_schema_validation: bool = False,
                 categorical_mapping: Mapping[str, dict] = {}):

        super().__init__(df=df, target_name=target_name,
                         categorical_mapping=categorical_mapping)

        self.seed_var = seed_var
        self.sample_to_node_mapping = sample_to_node_mapping

        self.shacl_validation_results = {}

        self.unique_seed_nodes = set(sample_to_node_mapping[self.seed_var].unique())
        self.unneeded_seed_nodes = set()

        self.random_schema_validation = random_schema_validation

        if communicator != None:
            self.communicator = communicator
        else:
            self.communicator = Communicator(None, "", "")

        self.seed_query = seed_query
        self.feature_names = list(self.df.columns)
        if self.target_name in self.feature_names:
            self.feature_names.remove(self.target_name)
        if self.seed_var in self.feature_names:
            self.feature_names.remove(self.seed_var)
        print('Number of unique nodes: ' + str(len(self.get_sample_to_node_mapping().unique())))

    @staticmethod
    def from_knowledge_graph(endpoint, validation_engine: Communicator, data_query: str, target_name: str, seed_var:str = 'x', seed_query: str = None, ground_truth: pd.DataFrame = None, raw_data_query_results_to_df_hook=None, **args):
        """To create the dataset, while generating the task_example_to_node mapping, the dataset has to be retrieved from an endpoint by querying. 
        This is the method, which has to be used if one doesn't has the task_example_to_node mapping locally available.

        Parameters
        ----------
        endpoint : str
            The SPARQL endpoint
        validation_engine : validating_models.shacl_validation_engine.Communicator
            The communicator object to use for the communication with an SHACL validation engine
        data_query : str
            The used to extract the dataset, but at most the seed nodes (bound to ?x (!!)) 
        seed_query : str, optional
            The query extracting all the seed nodes. Is inferred automatically from the data_query, but use incase of failure. 
        target_name : 
            The name of the target.
        ground_truth : pandas.DataFrame, optional
            If the ground_truth is not available in the knowledge graph a DataFrame(*seed_var*, target_name) can be provided, which will be joined with the dataset.
        raw_data_query_results_to_df_hook : Functional
            The user may provide a custom methode to transform the raw sparql query results into the final dataframe. The seed nodes can be ignored here, but the order of the results need to be kept and all results have to be transformed into an sample. 
        args : Further Arguments
            Further Arguments to be passed to raw_data_query_results_to_df_hook.
        """
        if seed_query != None:
            pvs = sparql.processor.prepareQuery(seed_query).algebra.get('PV')
            if len(pvs) != 1:
                raise Exception(
                    'Seed query needs to project exactly one variable!')
            seed_var = str(pvs[0])
        else:
            if ('?' + seed_var) not in data_query:
                raise Exception(f'If the seed query is not given the data_query has to retrieve the seeds and assign it to the variable {seed_var}.')
            seed_query = re.sub(
                r'(SELECT\s+(DISTINCT|REDUCED)?).*WHERE',
                f'SELECT ?{seed_var} WHERE',
                data_query, 1)

        # Get the data from the kg
        endpoint = SPARQLWrapper(endpoint)
        endpoint.setReturnFormat(JSON)
        endpoint.setQuery(data_query)
        result = endpoint.query().convert()
        bindings = [{key: value['value'] for key, value in binding.items()}
                    for binding in result['results']['bindings']]
        df = pd.DataFrame.from_dict(bindings)

        # Create DataFrame from Mapping
        if raw_data_query_results_to_df_hook != None:
            sample_to_node_mapping = df[[seed_var]].copy(deep=True)
            del df
            df = raw_data_query_results_to_df_hook(result, **args)

        else:
            # Automatic estimation of column types based on given bindings
            for column in df.columns:
                idx_with_column = None
                for i in range(len(result['results']['bindings'])):
                    if column in result['results']['bindings'][i]:
                        idx_with_column = i
                        break
                if idx_with_column != None:
                    value = result['results']['bindings'][idx_with_column][column]
                    found_type = None
                    if 'datatype' in value:
                        if value['datatype'] in DATATYPE_TO_PANDAS_TYPE:
                            found_type = DATATYPE_TO_PANDAS_TYPE[value['datatype']]
                    if 'type' in value:
                        if value['type'] in TYPE_TO_PANDAS_TYPE:
                            found_type = TYPE_TO_PANDAS_TYPE[value['type']]
                    if found_type != None:
                        print(column, found_type)
                        if found_type == 'numeric':
                            df[column] = pd.to_numeric(
                                df[column], errors='ignore')
                        elif found_type == 'category':
                            categories = list(df[column].unique().astype(str))
                            df[column] = df[column].astype(
                                pd.CategoricalDtype(categories=categories))
                        elif found_type == 'bool':
                            categories = list(df[column].unique().astype(str))
                            df[column] = df[column].astype(
                                pd.CategoricalDtype(categories=categories))
                        else:
                            df[[column]].apply(lambda x: x.astype(found_type))

            sample_to_node_mapping = df[[seed_var]].copy(deep=True)

        if isinstance(ground_truth, pd.DataFrame):
            df = pd.merge(df, ground_truth, on=seed_var, validate='1:1')

        return BaseDataset(df, target_name, seed_query, seed_var, sample_to_node_mapping=sample_to_node_mapping, communicator=validation_engine)
    
    def _generate_random_shacl_schema_validation_results(self, constraint):
        p = np.random.random_sample(size=(3,))
        p = p/p.sum()
        random_val_results = np.random.choice(np.array([True, False, None]), size=(len(self.unique_seed_nodes),),p = list(p))
        return {constraint.target_shape: {node: random_val_results[i] for i,node in enumerate(self.unique_seed_nodes) if random_val_results[i] != None}}

    def _calculate_shacl_schema_validation_result(self, constraint: Constraint):
        print(f"Validating SHACL shape schema {constraint.shape_schema_dir} of single constraint with target {constraint.target_shape}")
        if self.random_schema_validation:
            val_results = self._generate_random_shacl_schema_validation_results(constraint)
        else:
            val_results = self.communicator.request(
                self.seed_query, constraint.shape_schema_dir, [constraint.target_shape], self.seed_var)
        
        self.shacl_validation_results[constraint.shacl_identifier] = pd.DataFrame.from_dict(val_results[constraint.target_shape], orient='index', columns=[constraint.shacl_identifier], dtype=bool) 
    
    @time_shacl_schema_validation
    def calculate_shacl_schema_valiation_results(self, constraints: List[Constraint], checker = None):
        """Calculate the shacl schema validation results, such that each shacl schema is reduced and each shacl schema is only validated once. 
           The join with the sample-to-node mapping is not performed.

        Parameters
        ----------
        constraints : list of Constraint
            The list of constraints referring to the shacl schemas and target schemas needed to be validated over the knowledge graph
        checker : Checker
            If constraints include constraints of type PredictionConstraint the number of SHACL validation results needed can be further reduced by interleaving the constraint checking with the SHACL validation.

        """
        from validating_models.checker import Checker

        if isinstance(checker,Checker):
            dataset = checker.dataset
        else:
            dataset = self

        if isinstance(self.communicator, ReducedTravshaclCommunicator):
            sample_to_node_mapping = dataset.get_sample_to_node_mapping()
            # Collect target shapes for the given shape schema's
            constraints_per_shape_network = {}
            for constraint in constraints:
                if constraint.shape_schema_dir != None and constraint.target_shape != None:
                    if constraint.shacl_identifier not in self.shacl_validation_results:
                        if constraint.shape_schema_dir not in constraints_per_shape_network:
                            constraints_per_shape_network[constraint.shape_schema_dir] = [constraint]
                        else:
                            constraints_per_shape_network[constraint.shape_schema_dir].append(constraint)
            
            target_shapes_per_shape_network = {shape_schema: list(set([constraint.target_shape for constraint in constraints])) for shape_schema, constraints in constraints_per_shape_network.items()}
            for shape_schema_dir, target_shapes in target_shapes_per_shape_network.items():
                constraints_validated_here = constraints_per_shape_network[shape_schema_dir]
                filter_clause_per_target_shape = None

                if isinstance(checker, Checker): # and np.array([isinstance(constraint, PredictionConstraint) for constraint in constraints]).all():
                    filter_clause_per_target_shape = {target_shape: '' for target_shape in target_shapes}

                    #  Only perform the shacl validation for the instances needed (restricted in the case of PredictionConstraints)
                    indices_to_exclude_per_target_shape = {target_shape: [] for target_shape in target_shapes}
                    for constraint in constraints_validated_here:
                        if isinstance(constraint, PredictionConstraint):
                            if constraint.target_shape in indices_to_exclude_per_target_shape:
                                indices_to_exclude_per_target_shape[constraint.target_shape].append(checker.pre_evaluate_expression(constraint))
                        if isinstance(constraint, ShaclSchemaConstraint):
                            del indices_to_exclude_per_target_shape[constraint.target_shape] #TODO: May however reduce the nodes unneeded by the ProcessedDataset

                    for target_shape, indices_to_exclude_list in indices_to_exclude_per_target_shape.items():
                        if len(indices_to_exclude_list) == 0:
                            continue
                        indices_to_exclude = np.concatenate(indices_to_exclude_list, axis=0)
                        
                        nodes_to_exclude = set(sample_to_node_mapping[indices_to_exclude].values.unique()).union(dataset.unneeded_seed_nodes)
                        nodes_to_include = dataset.unique_seed_nodes.difference(nodes_to_exclude)                        
                        num_nodes_to_exclude = len(nodes_to_exclude)
                        num_nodes_to_include = len(nodes_to_include)

                        if num_nodes_to_exclude < num_nodes_to_include:
                            if num_nodes_to_exclude <= MAX_INSTANCES_IN_FILTER:
                                print('Adding FILTER clause to reduce the number of results.')
                                filter_clause_per_target_shape[target_shape] = f'FILTER (?{self.seed_var} NOT IN ({",".join([f"<{node}>" for node in nodes_to_exclude])}))'
                            else:
                                print(f'Skip adding filter clause to {target_shape} because of to many instances ({num_nodes_to_exclude}) to exclude.')
                                continue
                        else:
                            if num_nodes_to_include <= MAX_INSTANCES_IN_FILTER:
                                print('Adding FILTER clause to reduce the number of results.')
                                filter_clause_per_target_shape[target_shape] = f'FILTER (?{self.seed_var} IN ({",".join([f"<{node}>" for node in nodes_to_include])}))'
                            else:
                                print(f'Skip adding filter clause to {target_shape} because of to many instances ({num_nodes_to_include}) to include.')
                                continue  

                else:
                    print('Skip adding filter clause because no checker instance is provided.')

                print(f"Validating {shape_schema_dir} for targets {target_shapes}")
                if filter_clause_per_target_shape:
                    try:
                        val_results = self.communicator.request(self.seed_query, shape_schema_dir, target_shapes, self.seed_var, filter_clause_per_target_shape)
                    except:
                        print('Falling back to not use additional FILTER clauses.')
                        
                else:
                    val_results = self.communicator.request(self.seed_query, shape_schema_dir, target_shapes, self.seed_var)

                for target_shape in target_shapes:
                    shacl_identifier = Constraint.get_shacl_identifier(shape_schema_dir, target_shape)
                    self.shacl_validation_results[shacl_identifier] = pd.DataFrame.from_dict(val_results[target_shape], orient='index', columns=[shacl_identifier], dtype=bool) 

                    print("Number of validated targets: " + str(self.shacl_validation_results[shacl_identifier].sum()))
        else:
            new_entry('n_unique_nodes',len(self.unique_seed_nodes))
            for constraint in constraints:
                if constraint.shacl_identifier not in self.shacl_validation_results:
                    self._calculate_shacl_schema_validation_result(constraint)
    
    def get_shacl_schema_validation_results(self, constraints: List[Constraint], indices= None, rename_columns=False, replace_non_applicable_nans=False, checker = None):
        """Gives the shacl schema validation results for the given constraints. Performs the needed index-join with the sample-to-node-mapping.

        Parameters
        ----------
        constraints : List[Constraint]
            The constraints, for which the shacl schema validation results are needed
        indices : list of int, optional
            The indices for which the validation results are needed, by default all indices in the dataset
        rename_columns : boolean, optional
            Renames the columns of shacl schema validation results to their constraint names, by default no renaming is done
        replace_non_applicable_nans : boolean, optional
            Replaces nan values with true, by default nans are not replaced. Normally entities not included in the target defintion of the target shape are marked with nan.
        checker : Checker
            If constraints include constraints of type PredictionConstraint the number of SHACL validation results needed can be further reduced by interleaving the constraint checking with the SHACL validation.

        Returns
        -------
        pandas.Dataframe
            The dataframe containing the shacl schema validation results.
        """
        not_joined_yet = set()
        not_calculated_yet = []
        class_to_identifier= {}
        for constraint in constraints:
            if not constraint.shacl_identifier in self.sample_to_node_mapping.columns:
                # The Join was not yet performed
                not_joined_yet.add(constraint.shacl_identifier)
                try:
                    with open(Path(constraint.shape_schema_dir, f'{constraint.target_shape}.json'), 'r') as f:
                        shape = json.load(f)
                        target_class = shape['targetDef']['class']
                except:
                    target_class = 'NONE'
                finally:
                    if not target_class in class_to_identifier:
                        class_to_identifier[target_class] = set([constraint.shacl_identifier])   
                    else:
                        class_to_identifier[target_class].add(constraint.shacl_identifier)
                if not constraint.shacl_identifier in self.shacl_validation_results:
                    # The are no shacl schema validation results yet
                    not_calculated_yet.append(constraint)
                
        # Calculate the shacl schema validation results for the given constraints
        self.calculate_shacl_schema_valiation_results(not_calculated_yet, checker = checker)                

        if get_hyperparameter_value('use_outer_join'):
            self.sample_to_node_mapping = self._outer_join_pandas(not_joined_yet, class_to_identifier)
        else:
            self.sample_to_node_mapping = self._index_join_pandas(not_joined_yet, class_to_identifier)
        
        result_identifiers = [constraint.shacl_identifier for constraint in constraints]
        if indices != None:
            result = self.sample_to_node_mapping.loc[indices,result_identifiers]
        else:
            result = self.sample_to_node_mapping[result_identifiers]
        
        if replace_non_applicable_nans: #TODO: This might be the wrong place to do this...
            for constraint in constraints:
                if isinstance(constraint, (InvertedPredictionConstraint, InvertedShaclSchemaConstraint)):
                    result[constraint.shacl_identifier] = result[constraint.shacl_identifier].replace({np.nan: False})
                else:
                    result[constraint.shacl_identifier] = result[constraint.shacl_identifier].replace({np.nan: True})
        if rename_columns:
            result = result.rename(columns={constraint.shacl_identifier:constraint.name for constraint in constraints})
        return result
    
    @time_join
    def _index_join_pandas(self, identifiers, class_to_identifier): # left outer join
        print('Directly joining with the sample to node mapping!')
        for identifier in identifiers:
            print(f"Joining {identifier}")
            self.sample_to_node_mapping = self.sample_to_node_mapping.join(self.shacl_validation_results[identifier], on=self.seed_var)
        return self.sample_to_node_mapping
    
    @time_join
    def _outer_join_pandas(self, identifiers, class_to_identifier):
        print('Using the outer join')
        if isinstance(identifiers, list) and len(identifiers) == 1:
            identifiers = identifiers.pop()
            return self.sample_to_node_mapping.join(self.shacl_validation_results[identifiers], on=self.seed_var)
        elif isinstance(identifiers, str):
            return self.sample_to_node_mapping.join(self.shacl_validation_results[identifiers], on=self.seed_var)
        else:
            if get_hyperparameter_value('optimize_intermediate_results'):
                print('Ordering by cardinality')
                if 'NONE' in class_to_identifier:
                    remaining_identifiers = sorted(list(class_to_identifier['NONE']), key = lambda x: len(self.shacl_validation_results[x]))
                else:
                    remaining_identifiers = []
                class_to_identifier = {classe: list(identif) for classe, identif in class_to_identifier.items() if classe != 'NONE'}
                ordered_classes = sorted(list(class_to_identifier.keys()), key = lambda x: len(self.shacl_validation_results[class_to_identifier[x][0]]))
                identifiers = list(itertools.chain.from_iterable([class_to_identifier[classe] for classe in ordered_classes])) + remaining_identifiers
            
            #print([(identifier, len(self.shacl_validation_results[identifier])) for identifier in identifiers])

            first_key = identifiers.pop() # shacl results are joined first with full outer join and afterwards joined with the mapping with a left outer join
            
            if get_hyperparameter_value('not_pandas_optimized'):
                print('Using not pandas optimized join')
                result = self.shacl_validation_results[first_key]
                for key in identifiers:
                    result = result.join(self.shacl_validation_results[key], how='outer')
                    
                return self.sample_to_node_mapping.join(result, on=self.seed_var)
            
            return self.sample_to_node_mapping.join(self.shacl_validation_results[first_key].join([self.shacl_validation_results[key] for key in identifiers], how='outer'), on=self.seed_var)

    def get_sample_to_node_mapping(self, indices=None):
        """Returns the sample-to-node mapping.

        Parameters
        ----------
        indices : list of int, optional
            the indices of interest, defaults to all indices

        Returns
        -------
        pandas.Dataframe
            The DataFrame(*idx*, node_id) containg the sample-to-node mapping. 
        """
        if indices != None:
            return self.sample_to_node_mapping.loc[indices,self.seed_var]
        else:
            return self.sample_to_node_mapping.loc[:, self.seed_var]

    @staticmethod
    def get_result_iris(endpoint_url, query, var):
        endpoint = SPARQLWrapper(endpoint_url)
        endpoint.setReturnFormat(JSON)
        endpoint.setQuery(query)
        result = endpoint.query().convert()
        iris = set([binding[var]['value']
                   for binding in result['results']['bindings']])
        return iris


class ProcessedDataset(Dataset):
    """
    The user of a BaseDataset ''base'' may choose to extract the dataframe by calling base.df and choose to modify the features and targets (feature engineering). Also samples may be merged, dropped or duplicated.
    In that case the sample-to-node-mapping will be corrupted and need to be repaired. Keeping the mapping intact after performing operations like the one above is the task of this class.

    Parameters
    ----------
    processed_df : pandas.DataFrame
        The processed dataset having the features and the target as columns.
    base : validating_models.dataset.BaseDataset
        The BaseDataset originally used to create the Dataset, and includes the original sample-to-node mapping.
    base_indices : list of int
        The indices of the BaseDataset, which belong to the new processed dataset. (If the index structure is kept intakt it's simply list(processed_df.index))
    target_name : str
        The name of the target.
    categorical_mapping : mapping of str to (mapping of int to str)
        Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)
    """

    def __init__(self, processed_df: pd.DataFrame, base: BaseDataset, base_indices: List[int], target_name: str = None, categorical_mapping=None) -> None:

        categorical_mapping = categorical_mapping if categorical_mapping != None else base.categorical_mapping
        target_name = target_name if target_name != None else base.target_name

        super().__init__(processed_df, target_name, categorical_mapping)

        if not isinstance(base, BaseDataset) or base == None:
            raise Exception(f"base has to be of the type {type(BaseDataset)}!")

        self.base = base
        self.base_indices = base_indices
        self.seed_var = self.base.seed_var
        self.feature_names = list(self.df.columns)
        if self.target_name in self.feature_names:
            self.feature_names.remove(self.target_name)
        if self.seed_var in self.feature_names:
            self.feature_names.remove(self.seed_var)
        self.unique_seed_nodes = set(self.get_sample_to_node_mapping().unique())
        self.unneeded_seed_nodes = self.base.unique_seed_nodes - self.unique_seed_nodes
        print(f'Identified {len(self.unique_seed_nodes)} unique seed nodes!')
        print(f'In comparison to the BaseDataset this makes a total of {len(self.unneeded_seed_nodes)} unneeded seed nodes!')
    
    @staticmethod
    def from_unchanged_index(processed_df: pd.DataFrame, base: BaseDataset, target_name: str = None, categorical_mapping=None):
        """Creates the ProcessedDataset from a Dataframe with the index structure kept intact. For the datasets, this means that a given index refers to the semantically indentical samples (which therefor refer to the same seed node in the knowledge graph).
        E.g. :math:`\\forall i ((\\text{processed\_df}[i] \\rightarrow seednode_1) \land (\\text{base.df}[i] \\rightarrow seednode_2)) \implies seednode_1 = seednode_2`


        Parameters
        ----------
        processed_df : pd.DataFrame
            The modified dataframe representing the dataset.
        base : BaseDataset
            The origin of processed_df is base.df. Therefor base is the dataset directly created from the knowledge graph and contains the sample-to-node mapping.
        target_name : str, optional
            The name of the column in processed_df refering to the target, optional
        categorical_mapping : mapping of str to (mapping of int to str), optional
            Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)

        Returns
        -------
        ProcessedDataset
            The ProcessedDataset object with the correctly set base_indices.
        """
        return ProcessedDataset(processed_df, base, list(processed_df.index), target_name, categorical_mapping)
    
    @staticmethod
    def from_index_column(processed_df: pd.DataFrame, base: BaseDataset, column: str, target_name: str = None, categorical_mapping=None, drop_column_afterwards=True):
        """Works as from_unchanged_index but assumes the index copied to column before processing the dataframe. For example by using reset_index(drop=False) on base.df.

        Parameters
        ----------
        processed_df : pd.DataFrame
            The modified dataframe representing the dataset.
        base : BaseDataset
            The origin of processed_df is base.df. Therefore base is the dataset directly created from the knowledge graph and contains the sample-to-node mapping.
        column: str
            The index column.
        target_name : str, optional
            The name of the column in processed_df refering to the target, optional
        categorical_mapping : mapping of str to (mapping of int to str), optional
            Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)
        drop_column_afterwards, optional
            Whether to drop the specified column afterwards as the column is usually not needed afterwards. Defaults to True.

        Returns
        -------
        ProcessedDataset
            The ProcessedDataset object with the correctly set base_indices.
        """
        base_indices = list(processed_df[column])
        
        if drop_column_afterwards:
            processed_df.drop(columns=[column], inplace=True)

        return ProcessedDataset(processed_df, base, base_indices, target_name, categorical_mapping)

    @staticmethod
    def from_node_unique_columns(processed_df: pd.DataFrame, base: BaseDataset, base_columns: List[str], matching_new_columns: List[str] = None, target_name: str = None, categorical_mapping={}, drop_join_columns_afterwads=True):
        """Creates the ProcessedDataset from a Dataframe by joining with the BaseDataset via base_columns = matching_new_columns and therefore reconstructing the sample-to-node mapping.

        Parameters
        ----------
        processed_df : pd.DataFrame
            The modified dataframe representing the dataset.
        base : BaseDataset
            The origin of processed_df is base.df. Therefore base is the dataset directly created from the knowledge graph and contains the sample-to-node mapping.
        base_columns : List[str]
            The columns to be used for joining in base.df.
        matching_new_columns : List[str], optional
            The columns to be used for joining in processed_df, by default base_columns
        target_name : str, optional
            The name of the column in processed_df refering to the target, optional
        categorical_mapping : mapping of str to (mapping of int to str), optional
            Each feature/target can have an entry, which maps integer values of the feature/target to an meaningful description. (Especially useful for categorical features converted to numerical values.)
        drop_join_columns_afterwads : bool, optional
            Whether to drop the matching_new_columns in processed_df. Defaults to True.

        Returns
        -------
        ProcessedDataset
            The ProcessedDataset object with the correctly set base_indices.
        """
        if matching_new_columns == None:
            matching_new_columns = base_columns

        processed_df = processed_df.copy(deep=True).reset_index(drop=True)

        # Get the indices belonging to a node-unique entry --> Each list of indices should refer to the same seed node.
        column_value_to_indices = base.df.groupby(base_columns).indices
        
        # Check Node is unique per node_unique_column entry
        # for column_value, indices in column_value_to_indices.items():
        #     unique_nodes = np.unique(
        #         base.sample_to_node_mapping.iloc[indices, :].values.reshape(-1,))
        #     if len(unique_nodes) > 1:
        #         print(f'{column_value} has multiple different nodes assigned:')
        #         print(',\n'.join(unique_nodes[:5]))


        # Reverse the mapping created above, but only use the first element in each list (Should not matter as each indice in the list refer to the same seed node)
        # indices[0] is unique as it was already unique before
        index_to_column_value = {
            indices[0]: value for value, indices in column_value_to_indices.items()}

        # Create the according dataframe with the correct index 
        correct_index_with_join_columns = pd.DataFrame.from_dict(
            index_to_column_value, orient='index', columns=base_columns)
        # Copy the index into a column, as the index is reset when performing the merge.
        correct_index_with_join_columns.loc[:,
                                            '__index__'] = correct_index_with_join_columns.index

        processed_df[matching_new_columns] = processed_df[matching_new_columns].astype(
            str)
        correct_index_with_join_columns[base_columns] = correct_index_with_join_columns[base_columns].astype(
            str)

        df = pd.merge(processed_df, correct_index_with_join_columns,
                      how='left', left_on=base_columns, right_on=matching_new_columns)
        
        # Now the base_indices refering to the samples in the base dataset are the ones in the index column, created before.
        base_indices = pd.to_numeric(df.loc[:, '__index__'])

        if drop_join_columns_afterwads:
            processed_df.drop(columns=matching_new_columns, inplace=True)
        return ProcessedDataset(processed_df, base, list(base_indices), target_name=target_name, categorical_mapping=categorical_mapping)
    
    def get_shacl_schema_validation_results(self, constraints: List[Constraint], rename_columns=False, replace_non_applicable_nans=False, checker=None):
        """Gives the shacl schema validation results for the given constraints. Performs the needed index-join with the sample-to-node-mapping.
  

        Parameters
        ----------
        constraints : List[Constraint]
            The constraints, for which the shacl schema validation results are needed
        rename_columns : boolean, optional
            Renames the columns of shacl schema validation results to their constraint names, by default no renaming is done
        replace_non_applicable_nans : boolean, optional
            Replaces nan values with true, by default nans are not replaced. Normally entities not included in the target defintion of the target shape are marked with nan.

        Returns
        -------
        pandas.Dataframe
            The dataframe containing the shacl schema validation results.
        """
        return self.base.get_shacl_schema_validation_results(constraints, indices = self.base_indices, rename_columns=rename_columns, replace_non_applicable_nans=replace_non_applicable_nans, checker=checker)
    
    @lru_cache
    def get_sample_to_node_mapping(self):
        """Returns the sample-to-node mapping.

        Returns
        -------
        pandas.Dataframe
            The DataFrame(*idx*, node_id) containg the sample-to-node mapping. 
        """
        return self.base.get_sample_to_node_mapping(indices=self.base_indices)
    
    def calculate_shacl_schema_valiation_results(self, constraints: List[Constraint], checker = None):
        """Calculate the shacl schema validation results, such that each shacl schema is reduced and each shacl schema is only validated once. 
           The join with the sample-to-node mapping is not performed.

        Parameters
        ----------
        constraints : list of Constraint
            The list of constraints referring to the shacl schemas and target schemas needed to be validated over the knowledge graph
        checker : Checker
            If constraints include constraints of type PredictionConstraint the number of SHACL validation results needed can be further reduced by interleaving the constraint checking with the SHACL validation.
        """
        self.base.calculate_shacl_schema_valiation_results(constraints, checker = checker)


def categories_to_numericals(df):
    """Identifies categorical columns and converts all of them to numerical columns.

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    tuple of pandas.Dataframe, dict
        The processed dataset and a categorical mapping documenting the replacements done, to convert categorical colums to numerical ones.
    """
    df = df.copy(deep=True)
    converted_columns = set()
    cat_columns = set(df.select_dtypes(['category']).columns)
    object_columns = set(df.select_dtypes(['object']).columns)

    uniques_per_column = {column: df[column].unique() for column in object_columns}
    uniques_per_column.update({column: df[column].cat.categories for column in cat_columns})

    # 1.) Identify columns of dtype object or categorical only including numerical values and change the dtype accordingly
    for column in uniques_per_column.keys():
        if np.array([str.isnumeric(entry) for entry in uniques_per_column[column]], dtype=bool).all():
            df[column] = pd.to_numeric(df[column], errors='ignore')
            converted_columns.add(column)

    cat_columns = cat_columns - converted_columns

    # 2.) The rest of the columns of dtype object should be categorical
    for column in object_columns.difference(converted_columns):
        df[column] = df[column].astype(pd.CategoricalDtype(categories=uniques_per_column[column]))        
        cat_columns.add(column)

    # 3.) Convert all categoricals to numericals
    cat_columns = list(cat_columns)
    mappings = {column: dict(zip(df[column].cat.codes, df[column]))
                for column in cat_columns}

    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df, mappings


def get_valid_indices(dataset, constraints):
    """Given the dataset and a list of contraints this method will compute the indices of the dataset, belonging to entries, which are valid w.r.t. all given constraints. 
    If a constraint involves the target the ground truth value in the dataset is used.

    Parameters
    ----------
    dataset : BaseDataset or ProcessedDataset
        The dataset
    constraints : list of constraints
        The list of constraints

    Returns
    -------
    list of int
        the indices of the entries valid w.r.t. the constraints
    """
    from .checker import Checker
    checker = Checker(None, dataset, use_gt=True)
    validation_results = checker.get_constraint_validation_result(
        constraints)
    valid_idx = np.where(
        np.sum(validation_results, axis=1) == len(constraints))[0]
    return valid_idx
