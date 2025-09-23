import yaml
import csv
import random
import os
import copy

# --- Configuration ---
# Define the path to the configuration file.
# os.path.dirname(__file__) gets the directory of the current script (e.g., 'scripts').
# os.path.join then intelligently combines path parts to navigate up ('..') and then
# into the 'scene-1' directory, making the path robust.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'scene-1', 'schema.yaml')


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
    transaction_data = {}
    
    def process_fields(fields):
        for field_config in fields:
            generator_type = field_config.get('generator')

            if generator_type == 'conditional_generator':
                conditional_field_name = field_config['on_field']
                conditional_value = transaction_data.get(conditional_field_name)
                
                if conditional_value and isinstance(conditional_value, list):
                    conditional_value = conditional_value[0]

                if conditional_value in field_config['conditions']:
                    process_fields(field_config['conditions'][conditional_value])

            elif generator_type:
                field_name = field_config['name']
                params = field_config.get('params', {})
                
                value = None
                if generator_type == "random_number":
                    value = generate_random_number(params)
                elif generator_type == "random_choice":
                    value = generate_random_choice(params)
                else:
                    raise ValueError(f"Unknown generator type '{generator_type}' for field '{field_name}'.")
                
                transaction_data[field_name] = value

    process_fields(config['schema']['transaction_fields'])
    return transaction_data

def main():
    """
    Main function to orchestrate the CSV generation process.
    """
    print("INFO: Starting CSV generator script...")
    
    config = load_config(CONFIG_PATH)
    if config is None: return

    try:
        # We wrap the main logic in a try...except block to catch any errors
        # from our validation or generation functions and exit gracefully.
        validate_config(config)
    except (AssertionError, KeyError) as e:
        print(f"FATAL: Configuration validation failed: {e}")
        return

    print("INFO: Configuration loaded successfully.")

    # Construct a full, absolute path to place the output file in the project root.
    output_filename = config['simulation']['output_csv_path']
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_path = os.path.join(project_root, output_filename)

    headers = get_csv_headers(config)

    try:
        # The 'with' statement ensures the file is properly closed after we're done.
        with open(output_path, 'w', newline='') as f:
            # A DictWriter is convenient as it can write a dictionary directly to a CSV row.
            writer = csv.DictWriter(f, fieldnames=headers)
            # Write the header row first.
            writer.writeheader()
            print(f"INFO: Successfully created CSV file and wrote headers to: {output_path}")

            num_transactions = config['simulation']['num_transactions']
            print(f"INFO: Starting generation of {num_transactions} transactions...")

            # This is the main loop where each transaction row is created.
            for i in range(1, num_transactions + 1):
                # A progress indicator for long-running generation.
                if i % 1000 == 0:
                    print(f"  ...generating transaction {i} of {num_transactions}")
                
                # 1. Generate the base transaction data (e.g., amount, available_networks).
                row_data = generate_transaction_data(config)

                # 2. Determine the outcome for each processor for this specific transaction.
                for processor in config['processors']:
                    processor_name = processor['name']
                    
                    # Get the current state (properties) for this processor and this transaction.
                    state = get_current_processor_state(config, i, processor_name)
                    
                    # Perform a random roll against the success_rate to determine the outcome.
                    if random.random() < state.get('success_rate', 0):
                        outcome = "success"
                    else:
                        outcome = "fail"
                    
                    # Add the outcome to our row data dictionary (e.g., row_data['Stripe_outcome'] = 'success').
                    row_data[f"{processor_name}_outcome"] = outcome
                
                # 3. Write the complete dictionary as a new row in the CSV file.
                writer.writerow(row_data)

    except (IOError, KeyError, ValueError) as e:
        print(f"FATAL: An error occurred during CSV generation: {e}")
        return

    print(f"\nINFO: Script finished. Successfully generated {num_transactions} transactions in '{output_path}'.")


# This is a standard Python construct. The code inside this 'if' block
# will only run when the script is executed directly (e.g., "python3 scripts/csv_generator.py").
# It won't run if this script is imported as a module into another script.
if __name__ == "__main__":
    main()