import yaml
import csv
import random
import os
import copy
import time
import argparse
import sys


def load_config(path):
    """
    Reads and parses the YAML configuration file.
    """
    print(f"INFO: Attempting to load configuration from: {path}")
    # First, check if the file exists at the given path to avoid a crash.
    if not os.path.exists(path):
        print(f"ERROR: Configuration file not found at {path}")
        return None
    
    # 'with open(...)' is the standard way to handle files in Python.
    # It ensures the file is automatically closed even if errors occur.
    with open(path, 'r') as f:
        try:
            # yaml.safe_load is the recommended function to parse YAML.
            # It safely converts the YAML text into a Python dictionary.
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            # This block catches errors if the YAML file has a syntax issue.
            print(f"ERROR: Could not parse YAML file: {e}")
            return None

def validate_config(config):
    """
    Validates the structure and types of the loaded configuration.
    Raises an AssertionError if any check fails.
    """
    print("INFO: Validating configuration...")
    # 'assert' is a statement that checks if a condition is true.
    # If the condition is false, it raises an AssertionError, which we catch in main().
    assert 'simulation' in config, "Config missing 'simulation' section."
    assert 'num_transactions' in config['simulation'], "Config missing 'num_transactions'."
    assert isinstance(config['simulation']['num_transactions'], int), "'num_transactions' must be an integer."
    
    assert 'schema' in config, "Config missing 'schema' section."
    assert 'transaction_fields' in config['schema'], "Schema missing 'transaction_fields'."
    assert isinstance(config['schema']['transaction_fields'], list), "'transaction_fields' must be a list."
    print("INFO: Configuration validation passed.")

def get_csv_headers(config):
    """
    Dynamically generates the list of header names from the nested schema.
    """
    headers = []
    
    def find_headers_recursive(fields):
        for field in fields:
            if 'name' in field:
                headers.append(field['name'])
            # If it's a conditional generator, look inside its conditions
            if field.get('generator') == 'conditional_generator':
                for condition_fields in field.get('conditions', {}).values():
                    find_headers_recursive(condition_fields)

    find_headers_recursive(config['schema']['transaction_fields'])
    
    # Add the processor outcome columns
    for processor in config['processors']:
        headers.append(f"{processor['name']}_outcome")
        
    # Use dict.fromkeys to preserve order and remove duplicates
    unique_headers = list(dict.fromkeys(headers))
    print(f"INFO: Generated CSV headers: {unique_headers}")
    return unique_headers

def get_current_processor_state(config, entry_number, processor_name):
    """
    Calculates the state of a processor for a given transaction by checking for event overrides.
    For now, this just returns the default state.
    """
    # The 'next()' function is a concise way to find the first item in a list that matches a condition.
    # Here, it finds the processor object with the matching name. It returns None if not found.
    processor_defaults = next((p['defaults'] for p in config['processors'] if p['name'] == processor_name), None)
    
    if processor_defaults is None:
        print(f"WARNING: Could not find processor named '{processor_name}' in config.")
        return {}

    # copy.deepcopy creates a completely new copy of the dictionary and all its nested dictionaries.
    # This is crucial to prevent event overrides from one transaction affecting the next.
    current_state = copy.deepcopy(processor_defaults)

    # TODO: Implement event schedule logic here.
    # This will loop through config['event_schedule'] and apply overrides to 'current_state'.
    
    return current_state

def generate_random_number(params):
    """
    Generates a random number within a given range using parameters from the config.
    Raises KeyError if required parameters are missing.
    """
    try:
        # Direct dictionary access (e.g., params['min']) will 'panic' if the key is missing.
        min_val = params['min']
        max_val = params['max']
        decimals = params['decimals']
        
        # random.uniform() returns a floating-point number.
        raw_number = random.uniform(min_val, max_val)
        return round(raw_number, decimals)
    except KeyError as e:
        # We catch the default KeyError and re-raise it with a more helpful message.
        raise KeyError(f"Missing required parameter in 'random_number' params: {e}")

def generate_random_choice(params):
    """
    Generates a weighted random choice from the provided values or combinations.
    """
    try:
        primary_values = params['primary_values']
        primary_weights = params['primary_weights']
        if len(primary_weights) != len(primary_values):
            raise ValueError("Length of 'primary_weights' must match length of 'primary_values'.")

        if 'combinations' in params:
            distribution_weights = params['distribution_weights']
            if len(distribution_weights) != 2:
                raise ValueError("'distribution_weights' must have exactly two values.")

            pools = ['primary', 'combination']
            chosen_pool = random.choices(pools, weights=distribution_weights, k=1)[0]

            if chosen_pool == 'primary':
                chosen = random.choices(primary_values, weights=primary_weights, k=1)[0]
                return chosen if isinstance(chosen, list) else [chosen]
            else: # chosen_pool == 'combination'
                combinations = params['combinations']
                combination_weights = params['combination_weights']
                if len(combination_weights) != len(combinations):
                    raise ValueError("Length of 'combination_weights' must match 'combinations'.")
                return random.choices(combinations, weights=combination_weights, k=1)[0]
        else:
            chosen = random.choices(primary_values, weights=primary_weights, k=1)[0]
            return chosen if isinstance(chosen, list) else [chosen]

    except KeyError as e:
        raise KeyError(f"A required parameter (e.g., 'primary_values' or a weight) is missing: {e}")

def generate_transaction_data(config):
    """
    Acts as an orchestrator, calling the correct generator for each transaction field,
    handling conditional logic.
    """
    # This dictionary will store the generated data for a single transaction.
    transaction_data = {}
    
    # A nested function is used here to allow for recursion.
    # This is useful for handling nested conditional generation logic.
    def process_fields(fields):
        # Iterate over each field defined in the current level of the schema.
        for field_config in fields:
            generator_type = field_config.get('generator')

            # --- Conditional Generation ---
            # If the field is a 'conditional_generator', it means we need to decide
            # which sub-fields to generate based on the value of a previously generated field.
            if generator_type == 'conditional_generator':
                # 'on_field' tells us which previously generated field to check.
                conditional_field_name = field_config['on_field']
                # Get the value of that field from our transaction_data dictionary.
                conditional_value = transaction_data.get(conditional_field_name)
                
                # The 'random_choice' generator returns a list. We need the actual value inside.
                if conditional_value and isinstance(conditional_value, list):
                    conditional_value = conditional_value[0]

                # Check if the value we found matches one of the defined conditions.
                if conditional_value in field_config['conditions']:
                    # If it matches, recursively call this function to process the sub-fields.
                    process_fields(field_config['conditions'][conditional_value])

            # --- Standard Generation ---
            # If it's not a conditional generator, it's a standard data field.
            elif generator_type:
                field_name = field_config['name']
                params = field_config.get('params', {})
                
                value = None
                # Call the appropriate generator function based on the 'generator' type.
                if generator_type == "random_number":
                    value = generate_random_number(params)
                elif generator_type == "random_choice":
                    value = generate_random_choice(params)
                else:
                    # If the generator type is unknown, raise an error to stop the script.
                    raise ValueError(f"Unknown generator type '{generator_type}' for field '{field_name}'.")
                
                # Store the generated value in our dictionary with its field name as the key.
                transaction_data[field_name] = value

    # Start the process by calling the function with the top-level transaction fields.
    process_fields(config['schema']['transaction_fields'])
    # Return the completed dictionary for this transaction.
    return transaction_data

def discover_existing_scenes():
    """
    Discover all existing scene folders in the project directory.
    Returns a list of scene numbers that have folders.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scene_numbers = []
    
    for item in os.listdir(project_root):
        if os.path.isdir(os.path.join(project_root, item)) and item.startswith('scene-'):
            try:
                scene_num = int(item.split('-')[1])
                scene_numbers.append(scene_num)
            except (ValueError, IndexError):
                continue
    
    return sorted(scene_numbers)

def parse_scene_numbers(scenes_input):
    """
    Parse comma-separated scene numbers or 'all' keyword.
    Returns a list of scene numbers.
    """
    if scenes_input.lower() == 'all':
        return discover_existing_scenes()
    
    scene_numbers = []
    for scene_str in scenes_input.split(','):
        scene_str = scene_str.strip()
        try:
            scene_num = int(scene_str)
            scene_numbers.append(scene_num)
        except ValueError:
            print(f"ERROR: Invalid scene number '{scene_str}'. Must be an integer.")
            return None
    
    return sorted(list(set(scene_numbers)))  # Remove duplicates and sort

def validate_scenes(scene_numbers):
    """
    Validate that schema.yaml exists for all specified scenes.
    Returns (valid_scenes, missing_scenes).
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    valid_scenes = []
    missing_scenes = []
    
    for scene_num in scene_numbers:
        scene_folder = f"scene-{scene_num}"
        schema_path = os.path.join(project_root, scene_folder, 'schema.yaml')
        
        if os.path.exists(schema_path):
            valid_scenes.append(scene_num)
        else:
            missing_scenes.append(scene_num)
    
    return valid_scenes, missing_scenes

def generate_transactions_for_scene(scene_number):
    """
    Generate transactions for a single scene using its specific schema.yaml.
    Returns True if successful, False otherwise.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scene_folder = f"scene-{scene_number}"
    schema_path = os.path.join(project_root, scene_folder, 'schema.yaml')
    
    print(f"\n[SCENE {scene_number}] Starting transaction generation...")
    
    # Load scene-specific configuration
    config = load_config(schema_path)
    if config is None:
        print(f"ERROR: Failed to load schema for scene-{scene_number}")
        return False

    try:
        validate_config(config)
    except (AssertionError, KeyError) as e:
        print(f"ERROR: Configuration validation failed for scene-{scene_number}: {e}")
        return False

    # Get output path from config (should be scene-specific)
    output_filename = config['simulation']['output_csv_path']
    if not output_filename.startswith('./'):
        # Ensure the path is relative to project root
        output_path = os.path.join(project_root, output_filename)
    else:
        output_path = os.path.join(project_root, output_filename.lstrip('./'))

    headers = get_csv_headers(config)

    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            num_transactions = config['simulation']['num_transactions']
            print(f"[SCENE {scene_number}] Generating {num_transactions} transactions...")

            for i in range(1, num_transactions + 1):
                if i % 1000 == 0:
                    print(f"[SCENE {scene_number}] Progress: {i}/{num_transactions} transactions")
                
                # Generate transaction data
                row_data = generate_transaction_data(config)

                row_data['paymentId'] = f"PAY_{int(time.time() * 1000)}"

                # Add processor outcomes
                for processor in config['processors']:
                    processor_name = processor['name']
                    state = get_current_processor_state(config, i, processor_name)
                    
                    if random.random() < state.get('success_rate', 0):
                        outcome = "success"
                    else:
                        outcome = "fail"
                    
                    row_data[f"{processor_name}_outcome"] = outcome
                
                writer.writerow(row_data)

        print(f"[SCENE {scene_number}] ✓ Successfully generated transactions: {output_path}")
        return True
        
    except (IOError, KeyError, ValueError) as e:
        print(f"[SCENE {scene_number}] ✗ Error during generation: {e}")
        return False

def parse_arguments():
    """
    Parse command line arguments for multi-scene transaction generation.
    """
    parser = argparse.ArgumentParser(
        description='Generate transaction CSV files for one or more scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/csv_generator.py --scenes 1,2,3
  python3 scripts/csv_generator.py --scenes 1,5,10
  python3 scripts/csv_generator.py --scenes all
  python3 scripts/csv_generator.py  # Legacy mode: generates for scene-1 only
        """
    )
    
    parser.add_argument(
        '--scenes',
        type=str,
        help='Comma-separated list of scene numbers (e.g., "1,2,3") or "all" for all existing scenes'
    )
    
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the CSV generation process.
    Supports both legacy single-scene mode and new multi-scene mode.
    """
    args = parse_arguments()
    
    if args.scenes:
        # Multi-scene mode
        print("="*60)
        print("           MULTI-SCENE TRANSACTION GENERATOR")
        print("="*60)
        
        # Parse scene numbers
        scene_numbers = parse_scene_numbers(args.scenes)
        if scene_numbers is None:
            sys.exit(1)
        
        if not scene_numbers:
            print("ERROR: No valid scene numbers provided")
            sys.exit(1)
        
        print(f"Requested scenes: {', '.join(map(str, scene_numbers))}")
        
        # Validate scenes
        valid_scenes, missing_scenes = validate_scenes(scene_numbers)
        
        if missing_scenes:
            print(f"\n✗ ERROR: Missing schema.yaml for scenes: {', '.join(f'scene-{num}' for num in missing_scenes)}")
            print("Please ensure schema.yaml exists in each scene folder before generating transactions.")
            sys.exit(1)
        
        print(f"✓ All scenes validated successfully")
        
        # Generate transactions for each scene
        successful_scenes = []
        failed_scenes = []
        
        for scene_num in valid_scenes:
            success = generate_transactions_for_scene(scene_num)
            if success:
                successful_scenes.append(scene_num)
            else:
                failed_scenes.append(scene_num)
        
        # Summary
        print(f"\n{'='*60}")
        print("                    GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total scenes processed: {len(valid_scenes)}")
        print(f"Successful: {len(successful_scenes)}")
        print(f"Failed: {len(failed_scenes)}")
        
        if successful_scenes:
            print(f"✓ Successfully generated transactions for: {', '.join(f'scene-{num}' for num in successful_scenes)}")
        
        if failed_scenes:
            print(f"✗ Failed to generate transactions for: {', '.join(f'scene-{num}' for num in failed_scenes)}")
            sys.exit(1)
        
    else:
        # Legacy single-scene mode (backward compatibility)
        print("INFO: Starting CSV generator script (legacy mode for scene-1)...")
        
        config_path = os.path.join(os.path.dirname(__file__), '..', 'scene-1', 'schema.yaml')
        config = load_config(config_path)
        if config is None:
            return

        try:
            validate_config(config)
        except (AssertionError, KeyError) as e:
            print(f"FATAL: Configuration validation failed: {e}")
            return

        print("INFO: Configuration loaded successfully.")

        output_filename = config['simulation']['output_csv_path']
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_path = os.path.join(project_root, output_filename)

        headers = get_csv_headers(config)

        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                print(f"INFO: Successfully created CSV file and wrote headers to: {output_path}")

                num_transactions = config['simulation']['num_transactions']
                print(f"INFO: Starting generation of {num_transactions} transactions...")

                for i in range(1, num_transactions + 1):
                    if i % 1000 == 0:
                        print(f"  ...generating transaction {i} of {num_transactions}")
                    
                    row_data = generate_transaction_data(config)

                    for processor in config['processors']:
                        processor_name = processor['name']
                        state = get_current_processor_state(config, i, processor_name)
                        
                        if random.random() < state.get('success_rate', 0):
                            outcome = "success"
                        else:
                            outcome = "fail"
                        
                        row_data[f"{processor_name}_outcome"] = outcome
                    
                    writer.writerow(row_data)

                # Add a tiny sleep to ensure timestamp is unique for each row
                time.sleep(0.001)

        except (IOError, KeyError, ValueError) as e:
            print(f"FATAL: An error occurred during CSV generation: {e}")
            return

        print(f"\nINFO: Script finished. Successfully generated {num_transactions} transactions in '{output_path}'.")


# This is a standard Python construct. The code inside this 'if' block
# will only run when the script is executed directly (e.g., "python3 scripts/csv_generator.py").
# It won't run if this script is imported as a module into another script.
if __name__ == "__main__":
    main()
