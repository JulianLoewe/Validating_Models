from argparse import ArgumentParser
import json
import os

def generate_star_config(central_range, seed_range, additional_range, number_of_constraints):
    # Creating config from the outside to the inside
    def generate_star_config_rek(seed_range, additional_range, number_of_constraints, rek_number):
        if rek_number == 1:
            return {"Qs": seed_range}
        else:
            return {f"Class{number_of_constraints - (rek_number -2) }": generate_star_config_rek(seed_range, additional_range,number_of_constraints, rek_number - 1)}
    
    star_config = {"Class0": central_range, "Class1": generate_star_config_rek(seed_range, additional_range, number_of_constraints, number_of_constraints)}
    star_config['Class1'][''] = additional_range
    if number_of_constraints > 1:
        for i in range(1,number_of_constraints):
            star_config[f"Class{i+1}"] = additional_range
    return star_config


def main():
    parser = ArgumentParser(description='Generates a configuration file.')
    parser.add_argument('type', type=str)
    parser.add_argument('-c', '--constraints', type=int, default=None)
    parser.add_argument('-o', '--output_file', type=str, default=None)
    args = parser.parse_args()
    print(args)

    output_file = os.path.basename(args.output_file)
    graph_name = output_file.split('.json')[0]

    if args.type == 'star':
        central_range = [2000,2000]
        seed_range = [2000,2000]
        additional_range = [2000,2000]
        config = generate_star_config(central_range, seed_range, additional_range, args.constraints)
    
    final_config = {
        "targets_per_class": config,
        "graph_namespace": graph_name,
        "namespaces":
            {
                graph_name: f"http://example.com/graph_name/",
                "example": "http://example.com/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#" 
            }
        }
    with open(args.output_file, 'w') as f:
        json.dump(final_config, f)
main()