from pathlib import Path
import json
import random
import os
from pebble import ProcessPool
import multiprocessing as mp
import numpy as np
from pyinstrument import Profiler
from argparse import ArgumentParser
from collections import defaultdict
import re

P_ADDITIONAL_CONSTRAINT_INVALID_ON_INVALID_ENTITY = 0.5 # Number of valid constraints on invalid entity ~ B(#num_constraints - 1, p) 
#P_VALID_CONNECTION_ON_INVALID_CONSTRAINT = 0.5 # Number of valid connections on invalid entity ~ B(#num_required_connections - 1,p)
MAXIMAL_NUMBER_OF_ADDITIONAL_VALID_PARTNERS = 2
MAXIMAL_NUMBER_OF_ADDITIONAL_INVALID_PARTNERS = 2
# The entities to connect to are choosen uniform without replacement if choosen for a valid connection and else with replacement

def writeTriple(f, triple):
    f.write(f'{triple[0]} {triple[1]} {triple[2]} .\n')

def get_class_id(class_name, class_to_id, namespace):
    class_name = str(class_name)
    result = (class_to_id[namespace + class_name] if namespace + class_name in class_to_id else class_name)
    return str(result)

def graph_uri(item, namespace):
    return f'{namespace}:{item}'

def get_class_uri_from_class_id(class_id, id_to_class, namespace):
    if class_id in id_to_class:
        return '<' + id_to_class[class_id] + '>'
    else:
        return graph_uri(class_id, namespace)

def write_namespaces(f,namespaces):
    for key,value in namespaces.items():
        f.write(f'@prefix {key}: <{value}> .\n')


#################################################
# Writing the connections making entities valid #
#################################################

def get_number_of_partners(min_num, max_num, valid):
    if valid:
        num_partners = min_num
    else:
        if min_num > 0 and max_num < np.inf:
            if np.random.choice([True, False]): # Choose to violate min
                num_partners = int(np.random.uniform(0,min_num -1))
            else: # Choose to violate max
                num_partners = int(max_num + 1 + np.random.exponential(1))
        elif min_num > 0:
            num_partners = int(np.random.uniform(0,min_num -1))
        elif max_num < np.inf:
            num_partners = int(max_num + 1 + np.random.exponential(1))
        else:
            raise Exception(f'Constraint is not an Restriction with a minimum of {min_num} and a maximum of {max_num}')
    return num_partners

def make_intershape_constraint(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, entity_id_range, namespace, valid):
    partner_class_id = class_to_id[shapes[partner_shape]['targetDef']['class']] # For now only not overlapping classes are used
    class_combi_id = class_combi_to_class_combi_id[partner_class_id]
    # Get possible valid partner ids
    valid_partner_entity_range = get_entity_id_range_for_class_combi(class_combi_id, entity_borders, valid=True)
    valid_partner_entity_num = valid_partner_entity_range[1] - valid_partner_entity_range[0]
    print(f'Adding intershape constraints for {entity_id_range} valid {valid}')

    for entity_uri in entity_id_range_uri_generator(entity_id_range, namespace):
        for path, valid_outgoing_degree_range in constraint.items():
            min_num = valid_outgoing_degree_range[0]
            max_num = valid_outgoing_degree_range[1]
            if min_num > max_num:
                raise Exception('Either there is an impossible combination of constraints or the case is not supported!')

            num_partners = get_number_of_partners(min_num,max_num,valid)

            # Choose the minimal number of random valid entities to connect 
            partner_entity_ids = list(np.ones((num_partners,)) * valid_partner_entity_range[0] + np.random.choice(valid_partner_entity_num, size=(num_partners,), replace=False))

            # Add further valid entities
            if valid:
                maximal_number_of_additional_partners = max_num - min_num
                maximal_number_of_additional_partners = maximal_number_of_additional_partners if maximal_number_of_additional_partners <= MAXIMAL_NUMBER_OF_ADDITIONAL_VALID_PARTNERS else MAXIMAL_NUMBER_OF_ADDITIONAL_VALID_PARTNERS
                number_of_additional_partners = int(np.random.uniform(0, maximal_number_of_additional_partners))
                addional_valid_partner_entity_ids = list(np.ones((number_of_additional_partners,)) * valid_partner_entity_range[0] + np.random.choice(valid_partner_entity_num, size=(number_of_additional_partners,), replace=True))
                partner_entity_ids = partner_entity_ids + addional_valid_partner_entity_ids

            for partner_entity_id in partner_entity_ids:
                writeTriple(f, (entity_uri, path, get_entity_uri_from_entity_id(partner_entity_id, namespace)))
            
            if not valid:
                writeTriple(f, (entity_uri, graph_uri('violated_by', namespace), path))


def make_intrashape_constraint(f, constraint, valid_entity_id_range, namespace, valid):
    for path, valid_outgoing_degree_range in constraint.items():
        last_match = re.findall(r'()(?<=[/,#])(\w+)(?=[>])', path)[-1]
        for entity_uri in entity_id_range_uri_generator(valid_entity_id_range, namespace):
            min_num = valid_outgoing_degree_range[0]
            max_num = valid_outgoing_degree_range[1]

            num_partners = get_number_of_partners(min_num,max_num,valid)

            for i in range(num_partners):
                writeTriple(f, (entity_uri, path, f'"{last_match[1]}_{i}"^^xsd:string'))
            
            if not valid:
                writeTriple(f, (entity_uri, graph_uri('violated_by', namespace), path))


def add_random_invalid_intershape_constraint_partners(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, entity_id_range, namespace):
    partner_class_id = class_to_id[shapes[partner_shape]['targetDef']['class']] # For now only not overlapping classes are used
    class_combi_id = class_combi_to_class_combi_id[partner_class_id]

    # Get possible invalid partner ids
    invalid_partner_entity_range = get_entity_id_range_for_class_combi(class_combi_id, entity_borders, valid=False)
    invalid_partner_entity_num = invalid_partner_entity_range[1] - invalid_partner_entity_range[0]

    for entity_uri in entity_id_range_uri_generator(entity_id_range, namespace):
        for path, valid_outgoing_degree_range in constraint.items():
            number_of_additional_partners = int(np.random.uniform(0, MAXIMAL_NUMBER_OF_ADDITIONAL_INVALID_PARTNERS))
            addional_invalid_partner_entity_ids = list(np.ones((number_of_additional_partners,)) * invalid_partner_entity_range[0] + np.random.choice(invalid_partner_entity_num, size=(number_of_additional_partners,), replace=True))

            for partner_entity_id in addional_invalid_partner_entity_ids:
                writeTriple(f, (entity_uri, path, get_entity_uri_from_entity_id(partner_entity_id, namespace)))

def add_entity_properties(constraints, shapes, relevant_shape_names, class_to_id, class_combi_to_class_combi_id, i, class_combi, entity_borders, namespace, output_dir):
    output_file = f'data_{class_combi.replace("$","_")}.ttl'
    path = Path(output_dir, output_file).resolve()
    with open(path, 'a') as f:
        # Make these entities valid
        valid_entity_id_range = get_entity_id_range_for_class_combi(i, entity_borders, valid=True)

        for partner_shape, constraint in constraints.items():
            if partner_shape != '' and len(constraint) > 0: # Intershape Constraint
                make_intershape_constraint(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, valid_entity_id_range, namespace, True)
                add_random_invalid_intershape_constraint_partners(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, valid_entity_id_range, namespace)
            elif partner_shape == '' and len(constraint) > 0:
                make_intrashape_constraint(f, constraint, valid_entity_id_range, namespace, True)

        for entity_uri in entity_id_range_uri_generator(valid_entity_id_range, namespace):
            writeTriple(f,(entity_uri, graph_uri('is_valid',namespace),'"true"^^xsd:boolean'))

        # Make these entities invalid
        invalid_entity_id_range = get_entity_id_range_for_class_combi(i, entity_borders, valid=False)

        # if len(relevant_shape_names) == 1:
        #     # In that case the constraints can be split arbitary
        #     all_constraints = [(partner_shape, path, card_range[0:2], card_range[2]) for partner_shape, constraint in constraints.items() for path, card_range in constraint.items()]
        #     print(class_combi, "All Constraints: " + str(all_constraints))


        if len(constraints) > 1:
            if len(relevant_shape_names) > 1:
                shapes_covered_per_partner_shape = []
                for partner_shape, constraint in constraints.items():
                    covered_shapes = [shape for _, outgoing_degree_range in constraint.items() for shape in outgoing_degree_range[2]]
                    shapes_covered_per_partner_shape.append((len(covered_shapes), covered_shapes, partner_shape))            
                shapes_covered_per_partner_shape = sorted(shapes_covered_per_partner_shape, key=lambda x: x[0])

                covered_shapes = set()
                need_to_cover_x_more_shapes = len(relevant_shape_names.difference(covered_shapes))
                choosen_partner_shapes = []
                while need_to_cover_x_more_shapes > 0:
                    _, cov_s, partner_shape = shapes_covered_per_partner_shape.pop()
                    new_covered_shapes = covered_shapes.union(cov_s)
                    new_need_to_cover_x_more_shapes = len(relevant_shape_names.difference(new_covered_shapes))
                    if need_to_cover_x_more_shapes > new_need_to_cover_x_more_shapes:
                        covered_shapes = new_covered_shapes
                        need_to_cover_x_more_shapes = new_need_to_cover_x_more_shapes
                        choosen_partner_shapes.append(partner_shape)
            else:
                num_valid_constraints = 1 + np.random.binomial(len(constraints)-1, P_ADDITIONAL_CONSTRAINT_INVALID_ON_INVALID_ENTITY)
                choosen_partner_shapes = set(np.random.choice(list(constraints.keys()), size=(num_valid_constraints,), replace=False))
            print(f'{class_combi} will violate {choosen_partner_shapes}/{list(constraints.keys())}')

            valid_constraints = {partner_shape: constraint for partner_shape, constraint in constraints.items() if partner_shape not in choosen_partner_shapes}
            invalid_constraints = {partner_shape: constraint for partner_shape, constraint in constraints.items() if partner_shape in choosen_partner_shapes}


        else:
            invalid_constraints = constraints
            valid_constraints = {}
            print(f'{class_combi} will violate all constraints')


        for partner_shape, constraint in valid_constraints.items():
            if partner_shape != '' and len(constraint) > 0: # Intershape Constraint
                make_intershape_constraint(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, invalid_entity_id_range, namespace, True)
            elif partner_shape == '' and len(constraint) > 0:
                make_intrashape_constraint(f, constraint, invalid_entity_id_range, namespace, True)

        for partner_shape, constraint in invalid_constraints.items():
            if partner_shape != '' and len(constraint) > 0: # Intershape Constraint
                make_intershape_constraint(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, invalid_entity_id_range, namespace, False)
                add_random_invalid_intershape_constraint_partners(f, constraint, partner_shape, shapes, class_to_id, class_combi_to_class_combi_id, entity_borders, valid_entity_id_range, namespace)
            elif partner_shape == '' and len(constraint) > 0:
                make_intrashape_constraint(f, constraint, invalid_entity_id_range, namespace, False)

        for entity_uri in entity_id_range_uri_generator(invalid_entity_id_range, namespace):
            writeTriple(f,(entity_uri, graph_uri('is_valid',namespace),'"false"^^xsd:boolean'))
            
    return f'Adding {class_combi} entities properties done!'

# def get_entity_ids_for_class_id(class_id, entity_borders, class_combinations, valid):
#     start_id = 0
#     result = []
#     for i,class_combi in enumerate(class_combinations):
#         if class_id in class_combi.split('$'):
#             lower_bound = entity_borders[i * 2 + int(not valid)] #included
#             upper_bound = entity_borders[(i + 1) * 2 - int(valid)] #excluded
#             number_of_entities = upper_bound - lower_bound
#             result.append((start_id, lower_bound))
#             start_id = start_id + number_of_entities
#     return result

#########################################
# Writing the entities with the classes #
#########################################

def class_combi_to_uris(class_combi, id_to_class, namespace):
    for class_id in class_combi.split('$'):
        yield get_class_uri_from_class_id(class_id, id_to_class, namespace)

def entity_id_range_uri_generator(id_range, namespace):
    for i in range(*id_range):
        yield get_entity_uri_from_entity_id(i, namespace)

def get_entity_id_range_for_class_combi(i, entity_borders, valid = None):
    lower_bound = entity_borders[i * 2] #included
    upper_bound = entity_borders[(i + 1) * 2] #excluded
    if valid == None:
        return [lower_bound, upper_bound]
    else:
        if valid:
            upper_bound = entity_borders[(i + 1) * 2 - 1] #excluded
        else:
            lower_bound = entity_borders[i * 2 + 1] #included
        return [lower_bound, upper_bound]

def get_entity_uri_from_entity_id(i,namespace):
    return graph_uri(f'entity_{int(i):010d}', namespace)

def write_entities_for_class_combi(i, class_combi, id_to_class, entity_borders, namespace, output_dir):
    output_file = f'data_{class_combi.replace("$","_")}.ttl'
    path = Path(output_dir, output_file).resolve()
    with open(path, 'w') as f:
        class_uris = list(class_combi_to_uris(class_combi, id_to_class, namespace))
        for entity_id in range(*get_entity_id_range_for_class_combi(i, entity_borders)):
            for class_uri in class_uris:
                writeTriple(f,(get_entity_uri_from_entity_id(entity_id, namespace),'a', class_uri))
    return f'Creating {class_combi} entities done!'

#########################################
# Main                                  #
#########################################


def main():
    parser = ArgumentParser(description='Creates an artificial knowledge graph serialized with turtle given a SHACL shape schema only containing min constraints.')
    parser.add_argument('shape_schema_dir', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    args = parser.parse_args()

    # Parse Input and Output Directories
    input_shape_schema_dir = Path(args.shape_schema_dir)
    if args.output_dir != None:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(args.shape_schema_dir, 'data').resolve()
    os.makedirs(output_dir,exist_ok=True)

    # Read Shape Schema
    shapes = {}
    for shape_file in input_shape_schema_dir.glob('*.json'):
        with open(shape_file,'r') as f:
            shape = json.load(f)
            shapes[shape['name']] = shape

    with open(args.config, 'r') as f:
        config = json.load(f)  

    namespaces = config["namespaces"]
    namespace = config["graph_namespace"]

    # Read and Setup Classes
    class_to_id = {class_name: re.findall(r'()(?<=[/,#])(\w+)(?=[>])', '<' + class_name + '>')[-1][1] for i, class_name in enumerate(set([shape['targetDef']['class'] for shape in shapes.values()]))}
    id_to_class = {str(i): class_name for class_name, i in class_to_id.items()}
    class_id_to_shapes = {str(i):[shape_inner for shape_inner in shapes.values() if shape_inner['targetDef']['class'] == class_name] for i,class_name in id_to_class.items()}
    print('Shape Target Classes:', id_to_class)

    def read_targets_per_class_dict(dict_input):
        if isinstance(dict_input, dict):
            results = {}
            result_classes = set()
            for child in dict_input:
                result, class_out = read_targets_per_class_dict(dict_input[child])
                result_classes.update(class_out)
                child_id = get_class_id(child, class_to_id, namespaces[namespace])
                result_classes.update([child_id])
                results.update({child_id + '$' + get_class_id(key, class_to_id, namespaces[namespace]) if key != '' else child_id: value for key, value in result.items()})
            return results, result_classes
        else:
            return {"": dict_input}, set()
    
    targets_per_class_combi, class_ids = read_targets_per_class_dict(config['targets_per_class']) # dictionary of class combinations to [valid, invalid]
    print('All Class ids:', class_ids)
    print('[Valid, Invalid] per Class Combination:', targets_per_class_combi)
    class_combinations = list(targets_per_class_combi.keys())
    class_combi_to_class_combi_id = {class_combi: i for i, class_combi in enumerate(class_combinations)}
    class_combi_to_shapes = {class_combi: [class_id_to_shapes[class_id] for class_id in class_combi.split('$') if class_id in class_id_to_shapes] for class_combi in class_combinations}
    print('Class Combi to shapes: ', list(class_combi_to_shapes.keys()))
    entity_borders = [0]
    for combi in class_combinations:
        entity_borders.append(entity_borders[-1] + targets_per_class_combi[combi][0])
        entity_borders.append(entity_borders[-1] + targets_per_class_combi[combi][1])

    # Write Namespace
    path = Path(output_dir, 'data_.ttl').resolve()
    with open(path, 'w') as f:
        write_namespaces(f, namespaces)

    # A process for each class
    with ProcessPool(max_workers=mp.cpu_count(),context=mp.get_context('spawn')) as pool:
        results = []

        # merge overlapping shape constraints
        for i, class_combi in enumerate(class_combinations):
            results.append(pool.schedule(write_entities_for_class_combi, (i, class_combi, id_to_class, entity_borders, namespace, output_dir)))
            # Lower Bound collection phase
            constraints = {} # constraint --> intershape partner --> path --> range
            relevant_shapes = [shape for shapes in class_combi_to_shapes[class_combi] for shape in shapes]
            for shape in relevant_shapes:
                for constraint in shape['constraintDef']['conjunctions'][0]:
                    if not 'shape' in constraint:
                        constraint['shape'] = ''
                    
                    if constraint['shape'] not in constraints:
                        constraints[constraint['shape']] = {}

                    if 'min' in constraint: # Min Constraint
                        if constraint['path'] not in constraints[constraint['shape']]:
                            constraints[constraint['shape']][constraint['path']] = [constraint['min'], np.inf, set([shape['name']])]
                        elif constraints[constraint['shape']][constraint['path']][0] < constraint['min']:
                            constraints[constraint['shape']][constraint['path']][0] = constraint['min']
                            constraints[constraint['shape']][constraint['path']][2].add(shape['name'])
                    else: # Max Constraint
                        if constraint['path'] not in constraints[constraint['shape']]:
                            constraints[constraint['shape']][constraint['path']] = [0, constraint['max'], set([shape['name']])]
                        elif constraints[constraint['shape']][constraint['path']][1] > constraint['max']:
                            constraints[constraint['shape']][constraint['path']][1] = constraint['max']
                            constraints[constraint['shape']][constraint['path']][2].add(shape['name'])
            relevant_shape_names = set([shape['name'] for shape in relevant_shapes])
            results.append(pool.schedule(add_entity_properties, (constraints, shapes, relevant_shape_names, class_to_id, class_combi_to_class_combi_id, i, class_combi, entity_borders, namespace, output_dir)))
            print(class_combi, constraints)
        for result in results:
            print(result.result())

    os.system(f'cd {output_dir}; rm data.ttl; cat data_*.ttl > data.ttl; rm data_*.ttl')

    with open('generation_log.txt', 'w') as f:
        for i,class_combi in enumerate(class_combinations):
            for valid in [True, False]:
                f.write(f"{get_entity_id_range_for_class_combi(i, entity_borders,valid)} - {str(valid)} - {[(get_class_uri_from_class_id(class_id, id_to_class, namespace), set([shape['name'] for shapes in class_combi_to_shapes[class_id] for shape in shapes])) for class_id in class_combi.split('$') if class_id in class_combi_to_shapes]} \n")        

if __name__ == '__main__':
    main()
