# These are 'import' statements. They bring in pre-written code from Python's
# standard library or other installed packages, so we don't have to write
# everything from scratch.
import csv  # For reading and writing CSV files.
import os  # For interacting with the operating system, like finding file paths.
import requests  # For making HTTP requests to web services (APIs).
import yaml  # For reading and parsing YAML files.
import random  # For generating random numbers.
import time  # For pausing the script.
import ast  # For safely parsing string literals from CSV.
import argparse  # For command line argument parsing.
import sys  # For system-specific parameters and functions.

# --- Configuration ---
# These are global constants. They are variables whose values are set once and
# are not expected to change while the script is running.

# Default configuration path - can be overridden by command line arguments
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scene-1", "schema.yaml"
)

# The web address (URL) of the Decision Engine API we need to call.
DECISION_ENGINE_URL = "http://localhost:8080/decide-gateway"
# The URL for the API that we send feedback to after a simulated transaction.
FEEDBACK_API_URL = "http://localhost:8080/update-gateway-score"
# A throttle to limit how many requests we make per second, to avoid overwhelming the server.
REQUESTS_PER_SECOND = 8

# This is a Python dictionary that maps card network combinations to a specific
# 'cardIsin' number required by the API.
# The keys are tuples (e.g., ('Visa',)) which are like immutable lists, because
# dictionary keys must be unchangeable. We sort the networks before lookup to
# ensure ('Interlink', 'Visa') and ('Visa', 'Interlink') match the same key.
CARD_ISIN_MAP = {
    ("VISA",): "414141",
    ("MASTERCARD",): "414141",
    # Keys MUST be sorted alphabetically to match the logic in get_card_isin()
    ("ACCEL", "STAR", "VISA"): "440000",
    ("ACCEL", "MASTERCARD", "STAR"): "500251",
    ("DISCOVER", "NYCE", "PULSE"): "644564",
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
    with open(path, "r") as f:
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


def prepare_api_payload(csv_row, line_number, config, algorithm="SUPER_ROUTER"):
    """
    Prepares the JSON payload for the decision engine API call,
    handling different payload structures based on payment_method_type.
    """
    # Read the new, common columns from the CSV using the safer ast.literal_eval
    payment_method_type = ast.literal_eval(csv_row["payment_method_type"])[0]
    payment_method_list = ast.literal_eval(csv_row["payment_method"])

    # Get the list of all eligible gateways from the config
    eligible_gateways = [p["name"] for p in config["processors"]]

    # Dynamically create the eligibleGatewayPaymentMethodsList
    eligible_gateway_payment_methods = []
    for processor in config.get("processors", []):
        eligible_gateway_payment_methods.append(
            {
                "gateway": processor.get("name"),
                "payment_methods": processor.get("defaults", {}).get(
                    "supported_networks", []
                ),
            }
        )

    # --- Base Payload ---
    # This part is common to all payment types
    payload = {
        "merchantId": "m3",
        "eligibleGatewayList": eligible_gateways,  # Dynamically populated
        "eligibleGatewayPaymentMethodsList": eligible_gateway_payment_methods,
        "rankingAlgorithm": "SUPER_ROUTER",
        "eliminationEnabled": True,
        "paymentInfo": {
            "paymentId": csv_row.get(
                "paymentId", None
            ),  # A unique ID for this payment attempt.
            "amount": float(csv_row.get("amount", 0)),
            "currency": "USD",
            "customerId": "c1",
            "udfs": None,
            "preferredGateway": None,
            "paymentType": "ORDER_PAYMENT",
            "metadata": '{"merchant_category_code":"merchant_category_code_0001","acquirer_country":"US"}',
            "internalMetadata": None,
            "isEmi": False,
            "emiBank": None,
            "emiTenure": None,
            "paymentSource": None,
            "authType": None,
        },
    }

    # --- Conditional, Type-Specific Payload ---
    if payment_method_type == "CARD":
        card_isin = get_card_isin(payment_method_list)

        payload["paymentInfo"].update(
            {
                "paymentMethodType": "CARD",
                "paymentMethod": payment_method_list[
                    0
                ].upper(),  # Dynamically populated
                "cardIssuerBankName": None,
                "cardIsin": card_isin,  # The ISIN we looked up earlier.
                "cardType": "DEBIT",
                "cardSwitchProvider": None,
            }
        )

    elif payment_method_type == "WALLET":
        wallet_provider = payment_method_list[0]

        payload["paymentInfo"].update(
            {
                "paymentMethodType": "WALLET",
                "paymentMethod": wallet_provider.upper(),  # Dynamically populated
            }
        )

    else:
        raise ValueError(f"Unknown payment_method_type '{payment_method_type}' in CSV.")

    return payload


def send_feedback(processor, outcome, payment_id, network):
    """Sends feedback for a single, simulated transaction attempt."""
    if not processor or not network:
        return

    feedback_payload = {
        "merchantId": "m3",
        "gateway": processor,
        "status": "AUTHORIZED" if outcome == "success" else "FAILURE",
        "paymentId": payment_id,
        "paymentMethod": network.upper(),
        "txnLatency": {"gatewayLatency": random.randint(150, 6000)},
    }
    try:
        print(
            f"  -> Sending feedback for '{processor}' on network '{network}', outcome: '{outcome}'."
        )
        response = requests.post(FEEDBACK_API_URL, json=feedback_payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Feedback API call failed for gateway {processor}: {e}")
        print("Terminating session due to feedback API error.")
        sys.exit(1)


def check_service_health():
    """
    Performs a pre-flight health check to ensure the Decision Engine service
    is running and responding before processing any transactions.
    """
    print("INFO: Performing service health check...")

    # Create a minimal test payload to verify service availability
    test_payload = {
        "merchantId": "m3",
        "eligibleGatewayList": ["test_gateway"],
        "rankingAlgorithm": "SUPER_ROUTER",
        "eliminationEnabled": True,
        "paymentInfo": {
            "paymentId": "health_check_test",
            "amount": 100.0,
            "currency": "USD",
            "customerId": "c1",
            "paymentMethodType": "CARD",
            "paymentMethod": "VISA",
            "paymentType": "ORDER_PAYMENT",
        },
    }

    try:
        print(f"INFO: Testing connection to {DECISION_ENGINE_URL}...")
        response = requests.post(DECISION_ENGINE_URL, json=test_payload, timeout=10)
        response.raise_for_status()
        print("INFO: ✓ Service health check passed - Decision Engine is responding")
        return True

    except requests.exceptions.ConnectionError as e:
        print(
            f"FATAL: ✗ Cannot connect to Decision Engine service at {DECISION_ENGINE_URL}"
        )
        print("POSSIBLE CAUSES:")
        print("  - Decision Engine service is not running")
        print("  - Service is not listening on port 8080")
        print("  - Network connectivity issues")
        print("  - Firewall blocking the connection")
        print(f"ERROR DETAILS: {e}")
        sys.exit(1)

    except requests.exceptions.Timeout as e:
        print(f"FATAL: ✗ Decision Engine service timeout after 10 seconds")
        print("POSSIBLE CAUSES:")
        print("  - Service is running but overloaded")
        print("  - Network latency issues")
        print("  - Service is starting up (try again in a few moments)")
        print(f"ERROR DETAILS: {e}")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"FATAL: ✗ Decision Engine service returned HTTP error: {e}")
        print("POSSIBLE CAUSES:")
        print("  - Service configuration issues")
        print("  - Authentication/authorization problems")
        print("  - API endpoint changes")
        print("  - Service internal errors")
        sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"FATAL: ✗ Unexpected error during service health check: {e}")
        print("POSSIBLE CAUSES:")
        print("  - Unknown network issues")
        print("  - Service compatibility problems")
        print("  - System resource constraints")
        sys.exit(1)


def analyze_decision_and_run_simulation(decision, csv_row, payment_id, config):
    """
    Analyzes the decision engine's response to find the best possible and the chosen
    options, simulates the chosen option, and sends feedback.
    """
    results = {
        "chosen_processor": None,
        "chosen_network": None,
        "final_outcome": "fail",
        "savings": 0.0,
        "best_possible_processor": None,
        "best_possible_network": None,
        "best_possible_savings": 0.0,
    }

    try:
        priority_map = decision["super_router"]["priority_map"]
        payment_networks_available = ast.literal_eval(csv_row["payment_method"])
    except (KeyError, TypeError, IndexError, SyntaxError):
        print(f"ERROR: Could not parse decision response or CSV data: {decision}")
        return results

    # Helper to check if a processor in the config supports a given network
    def is_network_supported(processor_name, network):
        for p in config["processors"]:
            if p["name"] == processor_name:
                # Look for supported_networks in the processor's defaults
                return network in p.get("defaults", {}).get("supported_networks", [])
        return False

    # --- Two-Pass Analysis ---

    # Pass 1: Find all valid and successful options to determine the "best possible"
    valid_successful_options = []
    for option in priority_map:
        gateway = option.get("gateway")
        network = option.get("payment_method")  # CORRECTED KEY
        if not gateway or not network:
            continue

        is_valid = network in payment_networks_available and is_network_supported(
            gateway, network
        )
        if not is_valid:
            continue

        if csv_row.get(f"{gateway}_{network}_outcome") == "success":
            valid_successful_options.append(option)

    # Determine best possible option (highest savings, respecting priority for ties)
    if valid_successful_options:
        max_savings = max(
            opt["saving"] for opt in valid_successful_options
        )  # CORRECTED KEY
        best_options = [
            opt for opt in valid_successful_options if opt["saving"] == max_savings
        ]
        # Tie-break by choosing the one that appeared earliest in the original priority_map
        best_option = min(best_options, key=lambda opt: priority_map.index(opt))

        results["best_possible_processor"] = best_option.get("gateway")
        results["best_possible_network"] = best_option.get(
            "payment_method"
        )  # CORRECTED KEY
        results["best_possible_savings"] = best_option.get(
            "saving", 0.0
        )  # CORRECTED KEY

    # Pass 2: Find the first valid option to be the "chosen" one for simulation
    chosen_option = None
    for option in priority_map:
        gateway = option.get("gateway")
        network = option.get("payment_method")  # CORRECTED KEY
        if not gateway or not network:
            continue

        if network in payment_networks_available and is_network_supported(
            gateway, network
        ):
            chosen_option = option
            break  # Found the first valid one

    if not chosen_option:
        print(
            "WARNING: No valid processor/network combination found in decision response."
        )
        return results

    # Simulate the chosen option
    chosen_gateway = chosen_option.get("gateway")
    chosen_network = chosen_option.get("payment_method")  # CORRECTED KEY
    pre_determined_outcome = csv_row.get(
        f"{chosen_gateway}_{chosen_network}_outcome", "fail"
    )

    results.update(
        {
            "chosen_processor": chosen_gateway,
            "chosen_network": chosen_network,
            "final_outcome": pre_determined_outcome,
            "savings": chosen_option.get("saving", 0.0),  # CORRECTED KEY
        }
    )

    return (chosen_gateway, pre_determined_outcome, payment_id, chosen_network, results)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run decision engine simulations")
    parser.add_argument(
        "--scene-path", type=str, help="Path to scene folder (default: scene-1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: scene folder)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SUPER_ROUTER",
        help="Algorithm to use (default: SUPER_ROUTER)",
    )
    return parser.parse_args()


def main():
    """
    This is the main function that orchestrates the entire simulation process.
    It's called at the very end of the script.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Determine paths based on arguments or defaults
    if args.scene_path:
        scene_folder = args.scene_path
        config_path = os.path.join(scene_folder, "schema.yaml")
    else:
        scene_folder = "scene-1"
        config_path = DEFAULT_CONFIG_PATH

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = scene_folder

    algorithm = args.algorithm

    print(f"INFO: Starting simulation runner script for {scene_folder}...")
    print(f"INFO: Using algorithm: {algorithm}")

    # Perform service health check before any processing
    check_service_health()

    # Load the configuration from the YAML file first.
    config = load_config(config_path)
    # If loading fails, config will be None, and we exit the script.
    if config is None:
        return

    # Construct the full, absolute paths for the input and output files.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_csv_path = os.path.join(project_root, scene_folder, "transactions.csv")
    output_csv_path = os.path.join(project_root, output_dir, "output_results.csv")

    # Check if the input CSV file from the generator script actually exists.
    if not os.path.exists(input_csv_path):
        print(f"FATAL: The data file was not found at {input_csv_path}")
        print(
            "INFO: Please run the csv_generator.py script first to generate the data."
        )
        return  # Exit the script.

    print(f"INFO: Reading data from {input_csv_path}")

    # This 'try...except' block will catch any errors related to file operations.
    try:
        # We open both the input file for reading ('r') and the output file for
        # writing ('w') at the same time.
        with open(input_csv_path, "r") as infile, open(
            output_csv_path, "w", newline=""
        ) as outfile:

            # csv.DictReader reads the CSV file and treats each row as a dictionary,
            # where the keys are the header names. This is very convenient.
            reader = csv.DictReader(infile)

            # Define the headers for our output file, including all new analysis columns.
            output_headers = reader.fieldnames + [
                "chosen_processor",
                "chosen_network",
                "final_outcome",
                "savings",
                "best_possible_processor",
                "best_possible_network",
                "best_possible_savings",
            ]
            writer = csv.DictWriter(outfile, fieldnames=output_headers)
            writer.writeheader()

            print(f"INFO: Writing simulation results to: {output_csv_path}")
            print(f"INFO: Starting simulation...")

            # This is the main loop. It will iterate over every row in the input CSV.
            for row in reader:
                if reader.line_num % 500 == 0:
                    print(f"  ...processing transaction {reader.line_num}")

                try:
                    # Step 1: Prepare the payload for the API call.
                    api_payload = prepare_api_payload(
                        row, reader.line_num, config, algorithm
                    )

                    # Step 2: Call the decision engine API.
                    response = requests.post(
                        DECISION_ENGINE_URL, json=api_payload, timeout=5
                    )
                    response.raise_for_status()
                    decision = response.json()

                    # DEBUG: Print the request and response for the first two transactions.
                    if reader.line_num in [2, 3]:  # The first data row is line 2
                        print("\n--- DEBUG: REQUEST/RESPONSE --- ")
                        print(f"Transaction #{reader.line_num - 1}")
                        print("REQUEST to /decide-gateway:")
                        print(api_payload)
                        print("RESPONSE from /decide-gateway:")
                        print(decision)
                        print("--- END DEBUG ---\n")

                    # Step 3: Analyze the decision, simulate the transaction, and send feedback.
                    payment_id = api_payload["paymentInfo"]["paymentId"]
                    # simulation_results = analyze_decision_and_run_simulation(decision, row, payment_id, config)
                    (
                        chosen_gateway,
                        pre_determined_outcome,
                        payment_id,
                        chosen_network,
                        simulation_results,
                    ) = analyze_decision_and_run_simulation(
                        decision, row, payment_id, config
                    )

                    send_feedback(
                        chosen_gateway, "PENDING_VBV", payment_id, chosen_network
                    )
                    send_feedback(
                        chosen_gateway,
                        pre_determined_outcome,
                        payment_id,
                        chosen_network,
                    )

                except requests.exceptions.ConnectionError as e:
                    print(
                        f"FATAL: ✗ Cannot connect to Decision Engine during transaction {reader.line_num}: {e}"
                    )
                    print("POSSIBLE CAUSES:")
                    print("  - Decision Engine service stopped during execution")
                    print("  - Network connection lost")
                    print("  - Service crashed or restarted")
                    print("Terminating session due to connection failure.")
                    sys.exit(1)

                except requests.exceptions.Timeout as e:
                    print(
                        f"FATAL: ✗ Decision Engine timeout on transaction {reader.line_num}: {e}"
                    )
                    print("POSSIBLE CAUSES:")
                    print("  - Service overloaded or unresponsive")
                    print("  - Network latency issues")
                    print("  - Service performance degradation")
                    print("Terminating session due to timeout.")
                    sys.exit(1)

                except requests.exceptions.HTTPError as e:
                    print(
                        f"FATAL: ✗ Decision Engine HTTP error on transaction {reader.line_num}: {e}"
                    )
                    print("POSSIBLE CAUSES:")
                    print("  - 4xx: Invalid request payload or authentication issues")
                    print("  - 5xx: Service internal errors or configuration problems")
                    print("  - API contract changes or incompatibilities")
                    print("Terminating session due to HTTP error.")
                    sys.exit(1)

                except requests.exceptions.RequestException as e:
                    print(
                        f"FATAL: ✗ Unexpected API error on transaction {reader.line_num}: {e}"
                    )
                    print("POSSIBLE CAUSES:")
                    print("  - Unknown network or protocol issues")
                    print("  - Service compatibility problems")
                    print("  - System resource constraints")
                    print("Terminating session due to API error.")
                    sys.exit(1)

                except (ValueError, SyntaxError) as e:
                    print(
                        f"FATAL: ✗ Data processing error on transaction {reader.line_num}: {e}"
                    )
                    print("POSSIBLE CAUSES:")
                    print("  - Invalid CSV data format")
                    print("  - Configuration file issues")
                    print("  - Data corruption or format changes")
                    print("Terminating session due to data processing error.")
                    sys.exit(1)

                # Step 4: Add the simulation results to the original row data.
                row.update(simulation_results)

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
