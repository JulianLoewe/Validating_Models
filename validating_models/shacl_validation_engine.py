from abc import ABC
import requests
from multiprocessing import Queue
import logging
from validating_models.stats import new_entry
from pathlib import Path

from shaclapi import logger as shaclapi_logger
shaclapi_logger.setup(level=logging.DEBUG, handler=logging.FileHandler('api.log'))
from shaclapi.reduction import prepare_validation
from shaclapi.config import Config
from shaclapi.reduction.ValidationResultTransmitter import ValidationResultTransmitter
from shaclapi.query import Query


class Communicator(ABC):
    """Abstract base class used to communicate with a shacl valiation engine.

    Parameters
    ----------
    endpoint : str
        The endpoint of the shacl validation engine
    external_endpoint : str
        The SPARQL endpoint representing the knowledge graph to validate the shacl schema against.
    """

    def __init__(self, endpoint, external_endpoint, *args, **kwargs) -> None:
        self.endpoint = endpoint
        self.external_endpoint = external_endpoint

    def request(self, query, shape_schema_dir, target_shapes, seed_var, *args, **kwargs):
        '''Call to forward a validation requrest to the shacl validation engine. As a minimal requirement the shape schema and the target shapes are needed. 
        The target shapes are the shapes of interest, to which the valided and invalided instances will be returned. The query is used to reduce the number of targets to be validated.

        Parameters
        ----------
            query : str
                The query to be used to reduce the number of targets to be validated.
            shape_schema_dir : str
                The directory containing the shape schema definition files as json.
            target_shapes: List[str]
                The names of the shapes of interest.

        Returns
        -------
            Mapping[str,(List[str], List[str])]
                A dictionary mapping the target shape to the matching validated and invalidated nodes (identified by their IRI)
        '''
        pass

class ReducedTravshaclCommunicator(Communicator):
    def __init__(self, endpoint, external_endpoint, api_config) -> None:
        super().__init__(endpoint, external_endpoint)
        self.api_config = api_config
    
    def request(self, query, shape_schema_dir, target_shapes, seed_var, query_extension_per_target_shape = {}):
        config = Config.from_request_form({'config': self.api_config, 'query': query, 'schemaDir': shape_schema_dir, 'external_endpoint': self.external_endpoint, 'targetShape': {f'?{seed_var}': target_shapes}, 'query_extension_per_target_shape': query_extension_per_target_shape, 'test_identifier': str(Path(shape_schema_dir).name) }) #+ str(sorted(target_shapes))[:100] })
        queue = Queue()
        result_transmitter = ValidationResultTransmitter(output_queue=queue)
        shape_schema = prepare_validation(config, Query(query), result_transmitter)
        shape_schema.validate(True)
        queue.put('EOF')

        val_results = {shape: {} for shape in target_shapes}

        number_of_targets = 0
        item = queue.get()
        while item != 'EOF':
            number_of_targets += 1
            instance = item['instance']
            val_shape = item['validation'][0]
            val_res = item['validation'][1]
            if val_shape in target_shapes:
                val_results[val_shape][instance] = val_res
            item = queue.get()
        new_entry('number_of_targets', number_of_targets)
        queue.close()
        queue.cancel_join_thread()
        return val_results


class SHACLAPICommunicator(Communicator):
    '''Implements the Communicator interface for the shaclapi.
    '''
    def __init__(self, endpoint, external_endpoint, api_config) -> None:
        super().__init__(endpoint, external_endpoint)
        self.api_config = api_config

    def request(self, query, shape_schema_dir, target_shapes, seed_var):
        target_shape = target_shapes[0]

        params = {
            "query": query,
            "targetShape": target_shape,
            "external_endpoint": self.external_endpoint,
            "schemaDir": shape_schema_dir,
            "output_format": "test",
            "config": self.api_config,
        }
        response = requests.post(self.endpoint, data=params)
        assert response.status_code == 200, "Engine error, check engine output for details"

        val_results = {result[0]: True for result in response.json()['validTargets']}
        val_results.update({result[0]: False for result in response.json()['invalidTargets']})
        return  {target_shape: val_results}