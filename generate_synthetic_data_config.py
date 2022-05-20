from argparse import ArgumentParser
import json
import os
from pathlib import Path
import numpy as np

def generate_single_overlap_config(central_range, seed_range, additional_range, number_of_constraints):
    # Creating config from the outside to the inside
    def generate_star_config_rek(seed_range, additional_range, number_of_constraints, rek_number):
        if rek_number == 2:
            return {"Qs": seed_range}
        else:
            return {f"Class{number_of_constraints - (rek_number - 2) }": generate_star_config_rek(seed_range, additional_range, number_of_constraints, rek_number - 1)}
    
    star_config = {"Class0": central_range, "Class1": generate_star_config_rek(seed_range, additional_range, number_of_constraints, number_of_constraints)}
    star_config['Class1'][''] = additional_range
    if number_of_constraints > 1:
        for i in range(1,number_of_constraints-1):
            star_config[f"Class{i+1}"] = additional_range
    return star_config

def generate_distinct_config(class_range, number_of_constraints):
    distinct_config = {f"Class{i}": {"Qs": class_range} for i in range(number_of_constraints)}
    return distinct_config

def generate_nested_config(central_range, increasing_range, number_of_constraints):
    
    def generate_nested_config_rek(central_range, increasing_range, number_of_constraints, rek_number):
        if rek_number == 1:
            return {"Class0": central_range.astype(int).tolist()}
        else:
            return {f"Class{rek_number - 1}": generate_nested_config_rek(central_range, increasing_range, number_of_constraints, rek_number -1), "": (central_range + (rek_number - 1) * increasing_range).astype(int).tolist()}

    return {"Qs": generate_nested_config_rek(np.array(central_range), np.array(increasing_range), number_of_constraints, number_of_constraints)}


def generate_config(type, constraints, output_dir):

    graph_name = f'{type}_{constraints}'

    if type == 'single_overlap':
        central_range = [2000,2000]
        seed_range = [2000,2000]
        additional_range = [2000,2000]
        config = generate_single_overlap_config(central_range, seed_range, additional_range, constraints)
    elif type == 'distinct':
        class_range = [2000,2000]
        config = generate_distinct_config(class_range, constraints)
    elif type == 'nested':
        central_range = [2000,2000]
        increasing_range = [1000,1000]
        config = generate_nested_config(central_range, increasing_range, constraints)


    final_config = {
        "targets_per_class": config,
        "graph_namespace": graph_name,
        "namespaces":
            {
                graph_name: f"http://example.com/{graph_name}/",
                "example": "http://example.com/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#" 
            }
        }
    print(final_config)
    with open(Path(output_dir, f'{type}_{constraints}.json').resolve(), 'w') as f:
        json.dump(final_config, f)

if __name__ == '__main__':
    parser = ArgumentParser(description='Generates a configuration file.')
    parser.add_argument('type', type=str)
    parser.add_argument('-c', '--constraints', type=int, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    args = parser.parse_args()
    generate_config(args.type, args.constraints, args.output_dir)