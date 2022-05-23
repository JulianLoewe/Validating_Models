from argparse import ArgumentParser
from pathlib import Path
from generate_synthetic_data_kg import generate_kg
from generate_synthetic_data_config import generate_config
import os
import shutil
import json

def process_all_combinations():
    collect_nconstraints = {}
    shape_schemes_dir = Path('speed_test_shape_schemes/')
    for shape_schema_dir in shape_schemes_dir.glob("*/**"):
        for type in ['single_overlap', 'distinct', 'nested']:
            collect_nconstraints[f'{shape_schema_dir.name}_{type}'] = process_combination(shape_schema_dir, type)
    with open(Path(shape_schemes_dir.name + '_new', 'nconstraints.json'), 'w') as f:
        json.dump(collect_nconstraints,f)
    
def process_combination(shape_schema_dir, type):
    shape_schema_dir = Path(shape_schema_dir)
    # Rename namespace in shape schemes
    graph_name = f'{shape_schema_dir.name}_{type}'
    new_shape_schema_dir = Path('speed_test_shape_schemes_new/',graph_name)
    os.makedirs(new_shape_schema_dir, exist_ok=True)
    for file in shape_schema_dir.glob('*.json'):
        with open(Path(new_shape_schema_dir, file.name).resolve(), 'w') as f_new:
            with open(file, 'r') as f_old:
                new = f_old.read().replace('http://example.com/',f'http://example.com/{graph_name}/')
                f_new.write(new)
    
    # Generate configs
    nshapes = len(list(new_shape_schema_dir.glob('*.json')))
    if 'star_graph' in str(new_shape_schema_dir.name):
        nconstraints = nshapes - 1
    elif 'full_binary_tree' in str(new_shape_schema_dir.name):
        h = 1
        while True:
            if 2**(h + 1) >= nshapes:
                break 
            h +=1
        nconstraints = 2**h
    else: 
        raise Exception('Number of constraints unspecified!')

    config_file = generate_config(type, nconstraints, new_shape_schema_dir.parent, graph_name, nshapes)
    # Generate data
    generate_kg(new_shape_schema_dir, config_file,f'speed_test_kg/{graph_name}/')
    shutil.copyfile(f'speed_test_kg/{graph_name}/data.ttl', f'speed_test_kg/data/data_{graph_name}.ttl')
    shutil.rmtree(f'speed_test_kg/{graph_name}')
    return (nshapes-nconstraints, nshapes)

if __name__ == '__main__':
    parser = ArgumentParser(description='Setups the given experiment')
    parser.add_argument('-s','--shape_schema_dir', type=str, default='all')
    parser.add_argument('-t', '--type', type=str, default='all')
    args = parser.parse_args()
    if args.type == 'all':
        process_all_combinations()
    else:
        process_combination(args.shape_schema_dir, args.type)

    
            



    
    