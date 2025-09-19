# Configuration Schema Design (`config.yaml`)

## 1. Objective

This document details the design of the configuration file used by the Test Data Generator. The goal was to create a flexible, powerful, and human-readable schema that could evolve to support complex, real-world scenarios. The design is "schema-driven," meaning the configuration file itself defines the data model for the simulation, making the framework highly generic and reusable.

## 2. Format Selection: YAML

After discussing the trade-offs between JSON and YAML, the final decision was to use **YAML**. The primary reasons are:

- **Support for Comments:** This is critical for maintaining a complex configuration file, allowing for inline documentation.
- **Human Readability:** YAML's syntax is generally cleaner and less verbose than JSON's for nested structures, which is a key feature of our design.

## 3. Top-Level Structure

The `config.yaml` is organized into four main sections:

- `simulation`: Global parameters for the simulation run.
- `schema`: The core of the design. It defines the very structure of the data to be generated.
- `processors`: Defines the initial state of all processors in the simulation.
- `event_schedule`: Defines dynamic changes to the processors' state over the course of the simulation.

---

## 4. Detailed Section Breakdown

### `simulation`

This section holds global settings.

- `num_transactions`: The total number of transaction records to generate.
- `output_csv_path`: The path for the generated `.csv` file.

### `schema`

This section makes the framework generic. Instead of hard-coding fields like "amount" or "currency", this section defines them.

- **`transaction_fields`**: An array that defines the columns of the output data. Each entry contains:
    - `name`: The name of the field (e.g., `amount`).
    - `generator`: The method used to generate the data. The key design decision was to make the generators generic.
        - `random_number`: Generates a number within a range specified in `params`.
        - `random_choice`: A powerful, enhanced generator. If its `params` contain a simple `values` list, it picks one. If its `params` contain `primary_values` and `combinations`, it intelligently generates data for single-network or dual-network cards, respectively.
    - `params`: An object containing parameters for the specified `generator`.

- **`processor_properties`**: An array that defines the attributes each processor can have.
    - `name`: The name of the property (e.g., `success_rate`).
    - `type`: The data type (e.g., `number`, `boolean`, `object`).

### `processors`

This is a list where each item represents a processor and its starting state for the simulation. This is the primary section used to define a specific test scenario.

- `name`: The unique name of the processor.
- `defaults`: An object containing the default values for all properties defined in `schema.processor_properties`.

### `event_schedule`

This section makes the simulation dynamic. It's a list of events that override the default processor properties for a specific duration.

- `event_name`: A descriptive name for the event.
- `start_entry` / `end_entry`: The decision was made to use entry/row numbers instead of timestamps for V1. This simplifies the data generator, based on the assumption of a constant transaction rate.
- `targets`: A list of properties to change during the event.
    - `processor`: The name of the processor to affect.
    - `property`: The name of the property to change. A key feature is that this supports dot notation (e.g., `network_costs.Visa`) to target nested values.
    - `value`: The new value to apply.

## 5. Summary of Design Evolution

For context, the design evolved through several key decisions:

1.  **Shift to a Schema-Driven Model:** The initial idea of a simple config was replaced by a more powerful model where the config file defines its own schema. This was the most critical design pivot, enabling extreme flexibility.
2.  **Generalization of Generators:** Instead of creating many specific generators (e.g., `card_network_generator`), the decision was made to enhance a single generator (`random_choice`) to handle both simple and complex combination-based logic. This made the design cleaner and more reusable.
3.  **Entry-Based Time Model:** To balance realism and complexity for the initial version, we decided to define events based on `entry_number` rather than a complex `timestamp` system, operating on the assumption of a constant transaction rate.
