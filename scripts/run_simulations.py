# These are 'import' statements. They bring in pre-written code from Python's
# standard library or other installed packages, so we don't have to write
# everything from scratch.
import csv  # For reading and writing CSV files.
import os  # For interacting with the operating system, like finding file paths.
import requests  # For making HTTP requests to web services (APIs).
import yaml  # For reading and parsing YAML files.
import random  # For generating random numbers.
import time  # For pausing the script.

# --- Configuration ---
# These are global constants. They are variables whose values are set once and
# are not expected to change while the script is running.

# This constructs the full path to the configuration file.
# os.path.dirname(__file__) gets the directory of this script (e.g., 'scripts').
# os.path.join then intelligently combines path parts to navigate up ('..') and
# then into the 'scene-1' directory, making the path robust.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'scene-1', 'schema.yaml')

# The web address (URL) of the Decision Engine API we need to call.
DECISION_ENGINE_URL = "http://localhost:8080/decide-gateway"
# The URL for the API that we send feedback to after a simulated transaction.
FEEDBACK_API_URL = "http://localhost:8080/update-gateway-score"
# A throttle to limit how many requests we make per second, to avoid overwhelming the server.
REQUESTS_PER_SECOND = 10

# This is a Python dictionary that maps card network combinations to a specific
# 'cardIsin' number required by the API.
# The keys are tuples (e.g., ('Visa',)) which are like immutable lists, because
# dictionary keys must be unchangeable. We sort the networks before lookup to
# ensure ('Interlink', 'Visa') and ('Visa', 'Interlink') match the same key.
CARD_ISIN_MAP = {
    ('Visa',): '500251',
    ('Mastercard',): '600123',
    ('Interlink', 'Visa'): '500252',
    ('Maestro', 'Mastercard'): '600124',
}

# 'def' is how you define a function in Python.
def load_config(path):
    """
    Reads and parses the YAML configuration file.
    The text between triple quotes is a "docstring," which explains what the function does.
    """
    # An f-string (formatted string) lets us easily embed variable values inside a string.
    print(f"INFO: Attempting to load configuration from: {path}")
    # First, check if the file exists at the given path to avoid a crash.
    if not os.path.exists(path):
        print(f"ERROR: Configuration file not found at {path}")
        return None  # 'None' is Python's version of null or nothing.
    
    # 'with open(...)' is the standard, safe way to handle files in Python.
    # It ensures the file is automatically closed even if errors occur.
    # 'r' means we are opening the file for reading.
    with open(path, 'r') as f:
        # A 'try...except' block is for error handling. Python will 'try' to run
        # the code in the 'try' block. If an error occurs, it will run the code
        # in the 'except' block instead of crashing.
        try:
            # yaml.safe_load is the recommended function to parse YAML.
            # It safely converts the YAML text into a Python dictionary.
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            # This block catches errors if the YAML file has a syntax issue.
            print(f"ERROR: Could not parse YAML file: {e}")
            return None

def get_card_isin(networks_list):
    """
    Finds the correct cardIsin based on the list of available networks from the CSV.
    """
    # We sort the list of networks and convert it to a tuple so it can be used
    # as a key to look up the value in our CARD_ISIN_MAP dictionary.
    key = tuple(sorted(networks_list))
    # We check if the key exists in our dictionary.
    if key in CARD_ISIN_MAP:
        # If it exists, we return the corresponding value (the cardIsin).
        return CARD_ISIN_MAP[key]
    else:
        # If no mapping is found, we 'raise' an error to stop the script,
        # because we can't proceed without this information.
        raise ValueError(f"No cardIsin mapping found for network combination: {key}")

def prepare_api_payload(csv_row, line_number, config):
    """
    Prepares the JSON payload (as a Python dictionary) for the decision engine API call.
    'csv_row' is a dictionary representing one line from the input CSV file.
    """
    # The 'available_networks' from the CSV is a string like "['Visa']".
    # The 'eval()' function executes this string as Python code, turning it into a real list.
    # Note: eval() can be risky if the data isn't trusted, but here we trust our own generated CSV.
    available_networks = eval(csv_row['available_networks'])
    # We call our helper function to get the correct cardIsin for the networks.
    card_isin = get_card_isin(available_networks)

    # --- DYNAMIC FIELD 1: eligibleGatewayList ---
    # We create a list of all processor names from the configuration file.
    # This uses a list comprehension, a concise way to create lists in Python.
    eligible_gateways = [p['name'] for p in config['processors']]

    # --- DYNAMIC FIELD 2: paymentMethod ---
    # The instruction is to send one network, even for dual-network cards.
    # We'll just pick the first one from the list of available networks.
    payment_method = available_networks[0].upper()
    
    # This is the main payload dictionary. It's structured exactly as the API expects.
    # Values have been updated based on the user's request.
    payload = {
        "merchantId": "m3",
        "eligibleGatewayList": eligible_gateways, # Dynamically populated
        "rankingAlgorithm": "SUPER_ROUTER",
        "eliminationEnabled": True,
        "paymentInfo": {
            "paymentId": f"PAY{line_number}", # A unique ID for this payment attempt.
            "amount": float(csv_row.get('amount', 0)),
            "currency": "USD",
            "customerId": "c1",
            "udfs": None,
            "preferredGateway": None,
            "paymentType": "ORDER_PAYMENT",
            "metadata": "{\"merchant_category_code\":\"merchant_category_code_0001\",\"acquirer_country\":\"US\"}",
            "internalMetadata": None,
            "isEmi": False,
            "emiBank": None,
            "emiTenure": None,
            "paymentMethodType": "CARD",
            "paymentMethod": payment_method, # Dynamically populated
            "paymentSource": None,
            "authType": None,
            "cardIssuerBankName": None,
            "cardIsin": card_isin, # The ISIN we looked up earlier.
            "cardType": "DEBIT",
            "cardSwitchProvider": None
        }
    }
    return payload

def simulate_and_send_feedback(decision, csv_row, payment_id, payment_method):
    """
    Simulates a transaction using only the TOP gateway from the engine's decision,
    sends feedback for that single attempt, and determines the final outcome.
    'decision' is the JSON response we got from the decision engine.
    """
    # This 'try...except' block handles cases where the decision engine gives us
    # a response that doesn't have the structure we expect.
    try:
        # We access the nested data from the decision response dictionary.
        # This is like navigating a JSON object: decision -> super_router -> priority_map
        priority_map = decision['super_router']['priority_map']
        # If the list of gateways is empty, we can't proceed.
        if not priority_map:
            print(f"WARNING: Decision engine returned an empty priority map.")
            return None, "fail" # We return two values: the chosen processor (None) and the outcome.
        
        # We get the first item from the priority list (index 0) and then get its 'gateway' value.
        first_gateway_name = priority_map[0]['gateway']

    except (KeyError, TypeError, IndexError):
        # This will catch several types of errors:
        # - KeyError: A dictionary key (like 'super_router_output') doesn't exist.
        # - TypeError: We tried to index something that isn't a list (e.g., priority_map is None).
        # - IndexError: The priority_map list was empty, so accessing index 0 failed.
        print(f"ERROR: Could not parse first gateway from decision engine response: {decision}")
        return None, "fail"

    # This is the name of the processor the engine chose for us.
    chosen_processor = first_gateway_name
    
    # We construct the name of the column in our CSV that holds the pre-determined outcome
    # for this specific processor (e.g., "Stripe_outcome").
    outcome_column = f"{chosen_processor}_outcome"
    # We check if this column exists in our CSV row data.
    if outcome_column in csv_row:
        # If it exists, we use its value ('success' or 'fail').
        pre_determined_outcome = csv_row[outcome_column]
    else:
        # If not, we print a warning and assume it failed. This might happen if the
        # decision engine suggests a gateway we didn't define in our scenario.
        print(f"WARNING: Outcome for gateway '{chosen_processor}' not found in CSV row. Assuming fail.")
        pre_determined_outcome = "fail"

    # Now, we build the payload for the feedback API call.
    feedback_payload = {
        "merchantId": "m3",
        "gateway": chosen_processor,
        # This is a Python ternary operator. It's a compact if-else statement.
        # It sets 'status' to "SUCCESS" if pre_determined_outcome is "success", otherwise "FAILURE".
        "status": "SUCCESS" if pre_determined_outcome == "success" else "FAILURE",
        "paymentId": payment_id,
        "paymentMethodType": payment_method,
        # We simulate a random network latency for the feedback call.
        "txnLatency": { "gatewayLatency": random.randint(150, 6000) }
    }

    # We send the feedback to the feedback API.
    try:
        print(f"  -> Simulating with TOP choice '{chosen_processor}', outcome: '{pre_determined_outcome}'. Sending feedback.")
        # requests.post() sends an HTTP POST request. We send our payload as JSON
        # and set a timeout to prevent the script from hanging indefinitely.
        requests.post(FEEDBACK_API_URL, json=feedback_payload, timeout=2)
    except requests.exceptions.RequestException as e:
        # This catches any network-related errors during the API call.
        print(f"ERROR: Feedback API call failed for gateway {chosen_processor}: {e}")

    # The final outcome of the transaction is simply the pre-determined one.
    final_outcome = "success" if pre_determined_outcome == "success" else "fail"
    
    # This function returns two values to the main loop.
    return chosen_processor, final_outcome

def main():
    """
    This is the main function that orchestrates the entire simulation process.
    It's called at the very end of the script.
    """
    print("INFO: Starting simulation runner script...")

    # Load the configuration from the YAML file first.
    config = load_config(CONFIG_PATH)
    # If loading fails, config will be None, and we exit the script.
    if config is None: return

    # Construct the full, absolute paths for the input and output files.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_csv_path = os.path.join(project_root, config['simulation']['output_csv_path'])
    output_csv_path = os.path.join(project_root,"scene-1" ,"output_results.csv")

    # Check if the input CSV file from the generator script actually exists.
    if not os.path.exists(input_csv_path):
        print(f"FATAL: The data file was not found at {input_csv_path}")
        print("INFO: Please run the csv_generator.py script first to generate the data.")
        return # Exit the script.

    print(f"INFO: Reading data from {input_csv_path}")

    # This 'try...except' block will catch any errors related to file operations.
    try:
        # We open both the input file for reading ('r') and the output file for
        # writing ('w') at the same time.
        with open(input_csv_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
            
            # csv.DictReader reads the CSV file and treats each row as a dictionary,
            # where the keys are the header names. This is very convenient.
            reader = csv.DictReader(infile)
            
            # We define the headers for our output file. It will be all the original
            # headers plus two new ones for our simulation results.
            output_headers = reader.fieldnames + ['chosen_processor', 'final_outcome']
            # csv.DictWriter will write dictionaries to the CSV file.
            writer = csv.DictWriter(outfile, fieldnames=output_headers)
            # Write the header row first.
            writer.writeheader()

            print(f"INFO: Writing simulation results to: {output_csv_path}")
            print(f"INFO: Starting simulation...")

            # This is the main loop. It will iterate over every row in the input CSV.
            for row in reader:
                # This prints a progress message every 500 transactions.
                # The '%' is the modulo operator (it gives the remainder of a division).
                if reader.line_num % 500 == 0:
                    print(f"  ...processing transaction {reader.line_num}")


                # We set default values for our results. If anything goes wrong in the
                # 'try' block below, these are the values that will be written to the CSV.
                chosen_processor, final_outcome = (None, 'fail')
                try:
                    # Step 1: Prepare the payload for the API call, now passing the config object.
                    api_payload = prepare_api_payload(row, reader.line_num, config)
                    if reader.line_num == 2:
                        print(f"rachit payload {api_payload}")
                    payment_method_used = api_payload['paymentInfo']['paymentMethod']

                    # Step 2: Call the decision engine API.
                    response = requests.post(DECISION_ENGINE_URL, json=api_payload, timeout=5)
                    # This line will automatically raise an error if the API returns a
                    # bad status code (like 404 Not Found or 500 Internal Server Error).
                    response.raise_for_status()
                    # .json() parses the JSON response from the API into a Python dictionary.
                    decision = response.json()

                    # Step 3: Simulate the transaction with the engine's decision and send feedback.
                    payment_id = api_payload['paymentInfo']['paymentId']
                    chosen_processor, final_outcome = simulate_and_send_feedback(decision, row, payment_id, payment_method_used)

                except requests.exceptions.RequestException as e:
                    # This catches errors from the main API call (e.g., network error, timeout).
                    print(f"ERROR: API call failed for row {reader.line_num}: {e}")
                
                except (ValueError, SyntaxError) as e:
                    # This catches errors from prepare_api_payload (e.g., bad data in the CSV).
                    print(f"ERROR: Could not process data or prepare payload for row {reader.line_num}: {e}")
                
                # Step 4: Add the simulation results to the original row data.
                row['chosen_processor'] = chosen_processor
                row['final_outcome'] = final_outcome

                # Step 5: Write the complete, enriched row to the output CSV file.
                writer.writerow(row)

                # Step 6: Pause the script for a short time to respect the REQUESTS_PER_SECOND limit.
                time.sleep(1 / REQUESTS_PER_SECOND)

    except IOError as e:
        # This catches errors like not having permission to write the output file.
        print(f"FATAL: An error occurred with file I/O: {e}")
        return
    
    print(f"\nINFO: Script finished. Results saved to '{output_csv_path}'.")


# This is a standard Python construct. The code inside this 'if' block
# will only run when the script is executed directly from the command line
# (e.g., "python3 scripts/run_simulations.py").
# It won't run if this script is imported as a module into another script.
if __name__ == "__main__":
    main()
