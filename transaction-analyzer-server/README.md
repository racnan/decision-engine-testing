# Transaction Analyzer Server

A Node.js backend server that provides APIs for the Transaction Success Rate Analyzer web application. This server automatically scans for test case CSV files in the decision-engine-testing project and serves them through a REST API.

## Features

- **Automatic Test Case Discovery**: Scans all scene directories for `transactions.csv` files
- **REST API**: Provides endpoints to list test cases and retrieve CSV data
- **Schema Integration**: Reads `schema.yaml` files for test case descriptions
- **Web Interface**: Serves an enhanced HTML interface with test case selection
- **CORS Support**: Enables cross-origin requests for development

## API Endpoints

### GET /api/testcases
Returns a list of all available test cases with metadata.

**Response:**
```json
{
  "testcases": [
    {
      "scene_name": "scene-1",
      "file_path": "scene-1/transactions.csv",
      "description": "Test case description from schema.yaml"
    }
  ],
  "total_count": 20
}
```

### GET /api/testcases/:sceneName/csv
Returns the CSV content for a specific test case.

**Parameters:**
- `sceneName`: The name of the scene (e.g., "scene-1")

**Response:** Raw CSV content with appropriate headers

### GET /
Serves the main web interface for analyzing transaction data.

## Installation

1. Navigate to the server directory:
```bash
cd transaction-analyzer-server
```

2. Install dependencies:
```bash
npm install
```

3. Start the server:
```bash
npm start
```

Or for development with auto-reload:
```bash
npm run dev
```

## Usage

1. Start the server (it will run on http://localhost:3001 by default)
2. Open your browser and navigate to http://localhost:3001
3. Select a test case from the grid or upload your own CSV files
4. Analyze transaction success rates with interactive charts

## Project Structure

```
transaction-analyzer-server/
├── package.json          # Node.js dependencies and scripts
├── server.js             # Main server application
├── public/
│   └── index.html        # Web interface
└── README.md            # This file
```

## Dependencies

- **express**: Web framework for Node.js
- **cors**: Enable CORS for cross-origin requests
- **yaml**: Parse YAML schema files
- **nodemon**: Development dependency for auto-reload

## Environment Variables

- `PORT`: Server port (default: 3001)

## Test Case Structure

The server expects test cases to be organized as follows:

```
../scene-X/
├── transactions.csv      # Required: Transaction data
└── schema.yaml          # Optional: Test case metadata
```

The `schema.yaml` file should contain a `description` field for test case documentation.
