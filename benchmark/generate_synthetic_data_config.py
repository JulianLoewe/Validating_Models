from argparse import ArgumentParser
import json
import os
from pathlib import Path
import numpy as np

def generate_single_overlap_config(central_range, seed_range, additional_range, number_of_constraints, nshapes):
    # Creating config from the outside to the inside
    def generate_star_config_rek(seed_range, nconstraint, nshapes):
        if nconstraint == nshapes:
            return seed_range
        else:
            return {f"Class{nconstraint}": generate_star_config_rek(seed_range, nconstraint + 1, nshapes)}
    
    first_constraint_class = nshapes - number_of_constraints

    result_config = {"Qs": generate_star_config_rek(seed_range, first_constraint_class, nshapes)}

    result_config.update({f"Class{i}": additional_range for i in range(nshapes)})
    result_config.update({"Class0": central_range})

    return result_config

def generate_distinct_config(class_range, number_of_constraints, nshapes):
    distinct_config = {f"Class{i}": ({"Qs": class_range} if i >= nshapes - number_of_constraints else class_range) for i in range(nshapes)}
    return distinct_config

def generate_nested_config(central_range, increasing_range, number_of_constraints, nshapes):
    
    def generate_nested_config_rek(central_range, increasing_range, nconstraint, nshapes):
        if nconstraint == nshapes:
            return central_range.astype(int).tolist()
        else:
            return {f"Class{nconstraint}": generate_nested_config_rek(central_range - increasing_range, increasing_range, nconstraint + 1, nshapes), "": central_range.astype(int).tolist()}

    result_config = {"Qs": generate_nested_config_rek(np.array(central_range) + number_of_constraints * np.array(increasing_range), np.array(increasing_range), nshapes - number_of_constraints, nshapes)}
    result_config.update({f"Class{i}": central_range  for i in range(nshapes-number_of_constraints)})
    return result_config


def generate_config(type, constraints, output_dir, graph_name, nshapes):

    if type == 'single_overlap':
        central_range = [2000,2000]
        seed_range = [2000,2000]
        additional_range = [2000,2000]
        config = generate_single_overlap_config(central_range, seed_range, additional_range, constraints, nshapes)
    elif type == 'distinct':
        class_range = [2000,2000]
        config = generate_distinct_config(class_range, constraints, nshapes)
    elif type == 'nested':
        central_range = [2000,2000]
        increasing_range = [1000,1000]
        config = generate_nested_config(central_range, increasing_range, constraints, nshapes)


    final_config = {
        "targets_per_class": config,
        "graph_namespace": graph_name,
        "namespaces":
            {
                graph_name: f"http://example.com/{graph_name}/",
                "example": "http://example.com/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            }
        }
    output_file = Path(output_dir, f'{graph_name}.json').resolve()
    with open(output_file, 'w') as f:
        json.dump(final_config, f)
    return output_file

if __name__ == '__main__':
    parser = ArgumentParser(description='Generates a configuration file.')
    parser.add_argument('type', type=str)
    parser.add_argument('-c', '--constraints', type=int, default=None),
    parser.add_argument('-s', '--nshapes', type=int, default=None)
    parser.add_argument('-g', '--graph_name', type=str, default='test')
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    args = parser.parse_args()
    generate_config(args.type, args.constraints, args.output_dir, args.graph_name, args.nshapes)