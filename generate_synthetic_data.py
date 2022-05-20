from pathlib import Path
from generate_synthetic_data_kg import generate_kg
from generate_synthetic_data_config import generate_config
import os
if __name__ == '__main__':
    shape_schemes_dir = Path('speed_test_shape_schemes/')
    for shape_schema_dir in shape_schemes_dir.glob("*/**"):
        for type in ['single_overlap', 'distinct', 'nested']:
            # Rename namespace in shape schemes
            new_shape_schema_dir = Path('speed_test_shape_schemes_new/', f'{shape_schema_dir.name}_{type}')
            os.makedirs(new_shape_schema_dir, exist_ok=True)
            for file in shape_schema_dir.glob('*.json'):
                with open(Path(new_shape_schema_dir, file.name).resolve(), 'w') as f_new:
                    with open(file, 'r') as f_old:
                        new = f_old.read().replace('http://example.com/',f'http://example.com/{shape_schema_dir.name}_{type}/')
                        f_new.write(new)
            
            # Generate configs
            num_constraints = len(list(new_shape_schema_dir.glob('*.json')))
            generate_config(type, num_constraints,new_shape_schema_dir.parent)



    
    