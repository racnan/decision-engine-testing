const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const yaml = require('yaml');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// API endpoint to list all testcases
app.get('/api/testcases', (req, res) => {
    try {
        const testcases = scanForTestcases();
        res.json({
            testcases,
            total_count: testcases.length
        });
    } catch (error) {
        console.error('Error scanning for testcases:', error);
        res.status(500).json({ error: 'Failed to scan for testcases' });
    }
});

// API endpoint to get CSV content
app.get('/api/testcases/:sceneName/csv', (req, res) => {
    try {
        const { sceneName } = req.params;
        const csvPath = path.join(__dirname, '..', sceneName, 'transactions.csv');
        
        if (!fs.existsSync(csvPath)) {
            return res.status(404).json({ error: 'CSV file not found' });
        }
        
        const csvContent = fs.readFileSync(csvPath, 'utf8');
        res.setHeader('Content-Type', 'text/csv');
        res.send(csvContent);
    } catch (error) {
        console.error('Error reading CSV file:', error);
        res.status(500).json({ error: 'Failed to read CSV file' });
    }
});

// API endpoint to run simulations
app.post('/api/run-simulations', (req, res) => {
    try {
        const { scenes, numRuns, algorithm } = req.body;
        
        // Validate input
        if (!scenes || !Array.isArray(scenes) || scenes.length === 0) {
            return res.status(400).json({ error: 'Scenes array is required and must not be empty' });
        }
        
        if (!numRuns || numRuns < 1) {
            return res.status(400).json({ error: 'Number of runs must be at least 1' });
        }
        
        if (!algorithm || typeof algorithm !== 'string') {
            return res.status(400).json({ error: 'Algorithm name is required' });
        }
        
        // Validate scenes exist
        const availableScenes = scanForTestcases().map(tc => tc.scene_name);
        const invalidScenes = scenes.filter(scene => !availableScenes.includes(scene));
        
        if (invalidScenes.length > 0) {
            return res.status(400).json({ 
                error: `Invalid scenes: ${invalidScenes.join(', ')}`,
                available_scenes: availableScenes
            });
        }
        
        console.log(`Starting simulation for scenes: ${scenes.join(', ')}, runs: ${numRuns}, algorithm: ${algorithm}`);
        
        // Set response headers for streaming
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        // Start the simulation process
        runSimulations(scenes, numRuns, algorithm, res);
        
    } catch (error) {
        console.error('Error starting simulations:', error);
        res.status(500).json({ error: 'Failed to start simulations' });
    }
});

// API endpoint to get simulation results
app.get('/api/simulation-results/:sceneName/:runNumber', (req, res) => {
    try {
        const { sceneName, runNumber } = req.params;
        const runDir = path.join(__dirname, '..', sceneName, `run_${runNumber}`);
        
        if (!fs.existsSync(runDir)) {
            return res.status(404).json({ error: 'Simulation results not found' });
        }
        
        const results = getSimulationResults(runDir);
        res.json(results);
        
    } catch (error) {
        console.error('Error reading simulation results:', error);
        res.status(500).json({ error: 'Failed to read simulation results' });
    }
});

// API endpoint to get simulation CSV data
app.get('/api/simulation-csv/:sceneName/:runNumber', (req, res) => {
    try {
        const { sceneName, runNumber } = req.params;
        const csvPath = path.join(__dirname, '..', sceneName, `run_${runNumber}`, 'output_results.csv');
        
        if (!fs.existsSync(csvPath)) {
            return res.status(404).json({ error: 'Simulation CSV file not found' });
        }
        
        const csvContent = fs.readFileSync(csvPath, 'utf8');
        res.setHeader('Content-Type', 'text/csv');
        res.send(csvContent);
        
    } catch (error) {
        console.error('Error reading simulation CSV file:', error);
        res.status(500).json({ error: 'Failed to read simulation CSV file' });
    }
});

// Function to scan for testcases in scene directories
function scanForTestcases() {
    const testcases = [];
    const baseDir = path.join(__dirname, '..');
    
    try {
        const entries = fs.readdirSync(baseDir);
        
        for (const entry of entries) {
            const entryPath = path.join(baseDir, entry);
            const stat = fs.statSync(entryPath);
            
            if (stat.isDirectory() && entry.startsWith('scene-')) {
                const transactionsFile = path.join(entryPath, 'transactions.csv');
                
                if (fs.existsSync(transactionsFile)) {
                    // Try to read schema.yaml for description
                    const schemaFile = path.join(entryPath, 'schema.yaml');
                    let description = null;
                    
                    if (fs.existsSync(schemaFile)) {
                        try {
                            const schemaContent = fs.readFileSync(schemaFile, 'utf8');
                            const schemaData = yaml.parse(schemaContent);
                            description = schemaData.description || null;
                        } catch (err) {
                            console.warn(`Failed to parse schema for ${entry}:`, err.message);
                        }
                    }
                    
                    testcases.push({
                        scene_name: entry,
                        file_path: `${entry}/transactions.csv`,
                        description: description
                    });
                }
            }
        }
        
        // Sort by scene number for consistent ordering
        testcases.sort((a, b) => {
            const extractSceneNum = (name) => {
                const match = name.match(/scene-(\d+)/);
                return match ? parseInt(match[1]) : 0;
            };
            
            const aNum = extractSceneNum(a.scene_name);
            const bNum = extractSceneNum(b.scene_name);
            return aNum - bNum;
        });
        
    } catch (error) {
        console.error('Error scanning directory:', error);
    }
    
    return testcases;
}

// Function to run simulations
async function runSimulations(scenes, numRuns, algorithm, res) {
    const projectRoot = path.join(__dirname, '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'run_simulations.py');
    
    let totalRuns = scenes.length * numRuns;
    let completedRuns = 0;
    let results = {
        status: 'running',
        total_scenes: scenes.length,
        total_runs: totalRuns,
        completed_runs: 0,
        scenes: [],
        start_time: new Date().toISOString(),
        algorithm: algorithm
    };
    
    // Send initial status
    res.write(JSON.stringify({ type: 'status', data: results }) + '\n');
    
    try {
        for (let sceneIndex = 0; sceneIndex < scenes.length; sceneIndex++) {
            const sceneName = scenes[sceneIndex];
            const sceneResults = {
                scene_name: sceneName,
                runs: [],
                status: 'running'
            };
            
            results.scenes.push(sceneResults);
            
            // Send scene start update
            res.write(JSON.stringify({ 
                type: 'scene_start', 
                data: { scene: sceneName, index: sceneIndex + 1, total: scenes.length }
            }) + '\n');
            
            for (let runNum = 1; runNum <= numRuns; runNum++) {
                const runDir = path.join(projectRoot, sceneName, `run_${runNum}`);
                
                // Create run directory
                if (!fs.existsSync(runDir)) {
                    fs.mkdirSync(runDir, { recursive: true });
                }
                
                // Send run start update
                res.write(JSON.stringify({ 
                    type: 'run_start', 
                    data: { scene: sceneName, run: runNum, total_runs: numRuns }
                }) + '\n');
                
                try {
                    // Execute simulation
                    const success = await executeSimulation(sceneName, runDir, algorithm, projectRoot);
                    
                    if (success) {
                        // Generate reports
                        await generateReports(runDir, projectRoot);
                        
                        // Read results
                        const runResults = getSimulationResults(runDir);
                        
                        sceneResults.runs.push({
                            run_number: runNum,
                            status: 'success',
                            results: runResults
                        });
                        
                        completedRuns++;
                        results.completed_runs = completedRuns;
                        
                        // Send run complete update
                        res.write(JSON.stringify({ 
                            type: 'run_complete', 
                            data: { 
                                scene: sceneName, 
                                run: runNum, 
                                status: 'success',
                                results: runResults,
                                progress: (completedRuns / totalRuns) * 100
                            }
                        }) + '\n');
                        
                    } else {
                        sceneResults.runs.push({
                            run_number: runNum,
                            status: 'failed',
                            error: 'Simulation execution failed'
                        });
                        
                        // Send run failed update
                        res.write(JSON.stringify({ 
                            type: 'run_failed', 
                            data: { scene: sceneName, run: runNum, error: 'Simulation execution failed' }
                        }) + '\n');
                    }
                    
                } catch (error) {
                    console.error(`Error in run ${runNum} for ${sceneName}:`, error);
                    sceneResults.runs.push({
                        run_number: runNum,
                        status: 'failed',
                        error: error.message
                    });
                    
                    // Send run failed update
                    res.write(JSON.stringify({ 
                        type: 'run_failed', 
                        data: { scene: sceneName, run: runNum, error: error.message }
                    }) + '\n');
                }
            }
            
            sceneResults.status = 'completed';
            
            // Send scene complete update
            res.write(JSON.stringify({ 
                type: 'scene_complete', 
                data: { scene: sceneName, runs: sceneResults.runs.length }
            }) + '\n');
        }
        
        results.status = 'completed';
        results.end_time = new Date().toISOString();
        
        // Send final completion with compact data (keep performance analysis but remove detailed reports)
        const compactResults = {
            ...results,
            scenes: results.scenes.map(scene => ({
                ...scene,
                runs: scene.runs.map(run => ({
                    ...run,
                    results: run.results ? {
                        summary: run.results.summary,
                        performance_analysis: run.results.performance_analysis
                        // Remove detailed_report to reduce size but keep performance_analysis
                    } : null
                }))
            }))
        };
        
        res.write(JSON.stringify({ type: 'complete', data: compactResults }) + '\n');
        res.end();
        
    } catch (error) {
        console.error('Error during simulation:', error);
        results.status = 'failed';
        results.error = error.message;
        
        res.write(JSON.stringify({ type: 'error', data: results }) + '\n');
        res.end();
    }
}

// Function to execute a single simulation
function executeSimulation(sceneName, runDir, algorithm, projectRoot) {
    return new Promise((resolve) => {
        const python = process.platform === 'win32' ? 'python' : 'python3';
        const scriptPath = path.join(projectRoot, 'scripts', 'run_simulations.py');
        const scenePath = path.join(projectRoot, sceneName);
        
        const args = [
            scriptPath,
            '--scene-path', scenePath,
            '--output-dir', runDir,
            '--algorithm', algorithm
        ];
        
        const child = spawn(python, args, {
            cwd: projectRoot,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        let stdout = '';
        let stderr = '';
        
        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        
        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        
        child.on('close', (code) => {
            if (code === 0) {
                console.log(`Simulation completed for ${sceneName}`);
                resolve(true);
            } else {
                console.error(`Simulation failed for ${sceneName}:`, stderr);
                resolve(false);
            }
        });
        
        child.on('error', (error) => {
            console.error(`Failed to start simulation for ${sceneName}:`, error);
            resolve(false);
        });
    });
}

// Function to generate reports
function generateReports(runDir, projectRoot) {
    return new Promise((resolve) => {
        const python = process.platform === 'win32' ? 'python' : 'python3';
        const scriptPath = path.join(projectRoot, 'scripts', 'results.py');
        const inputFile = path.join(runDir, 'output_results.csv');
        
        const args = [
            scriptPath,
            '--input-file', inputFile,
            '--output-dir', runDir
        ];
        
        const child = spawn(python, args, {
            cwd: projectRoot,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        child.on('close', (code) => {
            resolve(code === 0);
        });
        
        child.on('error', () => {
            resolve(false);
        });
    });
}

// Function to get simulation results
function getSimulationResults(runDir) {
    const results = {
        summary: null,
        detailed_report: null,
        performance_analysis: {}
    };
    
    try {
        // Read detailed performance report
        const reportPath = path.join(runDir, 'detailed_performance_report.txt');
        if (fs.existsSync(reportPath)) {
            results.detailed_report = fs.readFileSync(reportPath, 'utf8');
        }
        
        // Read output results CSV and generate summary
        const csvPath = path.join(runDir, 'output_results.csv');
        if (fs.existsSync(csvPath)) {
            results.summary = analyzeOutputResults(csvPath);
        }
        
        // Read performance analysis CSVs
        const analysisDir = path.join(runDir, 'detailed_performance_analysis');
        if (fs.existsSync(analysisDir)) {
            const analysisFiles = [
                'overall_performance.csv',
                'processor_performance.csv',
                'network_performance.csv',
                'processor_network_breakdown.csv'
            ];
            
            for (const file of analysisFiles) {
                const filePath = path.join(analysisDir, file);
                if (fs.existsSync(filePath)) {
                    const key = file.replace('.csv', '');
                    results.performance_analysis[key] = readCSVFile(filePath);
                }
            }
        }
        
    } catch (error) {
        console.error('Error reading simulation results:', error);
    }
    
    return results;
}

// Function to analyze output results CSV
function analyzeOutputResults(csvPath) {
    try {
        console.log(`Analyzing results from: ${csvPath}`);
        
        if (!fs.existsSync(csvPath)) {
            console.log(`CSV file does not exist: ${csvPath}`);
            return null;
        }
        
        const csvContent = fs.readFileSync(csvPath, 'utf8');
        console.log(`CSV content length: ${csvContent.length}`);
        
        const lines = csvContent.trim().split('\n');
        console.log(`Number of lines: ${lines.length}`);
        
        if (lines.length < 2) {
            console.log('CSV file has insufficient data');
            return null;
        }
        
        const headers = lines[0].split(',');
        console.log(`Headers: ${headers.join(', ')}`);
        
        let totalTransactions = 0;
        let successfulTransactions = 0;
        let totalSavings = 0;
        let bestPossibleSavings = 0;
        let optimalChoices = 0;
        
        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            const rowData = {};
            
            headers.forEach((header, index) => {
                rowData[header] = row[index];
            });
            
            totalTransactions++;
            
            if (rowData.final_outcome === 'success') {
                successfulTransactions++;
            }
            
            totalSavings += parseFloat(rowData.savings || 0);
            bestPossibleSavings += parseFloat(rowData.best_possible_savings || 0);
            
            if (rowData.chosen_processor === rowData.best_possible_processor &&
                rowData.chosen_network === rowData.best_possible_network) {
                optimalChoices++;
            }
        }
        
        const successRate = totalTransactions > 0 ? (successfulTransactions / totalTransactions) * 100 : 0;
        const efficiency = bestPossibleSavings > 0 ? (totalSavings / bestPossibleSavings) * 100 : 0;
        const optimalRate = totalTransactions > 0 ? (optimalChoices / totalTransactions) * 100 : 0;
        
        const summary = {
            total_transactions: totalTransactions,
            successful_transactions: successfulTransactions,
            success_rate: successRate,
            total_savings: totalSavings,
            best_possible_savings: bestPossibleSavings,
            efficiency: efficiency,
            optimal_choices: optimalChoices,
            optimal_rate: optimalRate
        };
        
        console.log(`Analysis summary:`, summary);
        return summary;
        
    } catch (error) {
        console.error('Error analyzing output results:', error);
        return null;
    }
}

// Function to read CSV file
function readCSVFile(filePath) {
    try {
        const csvContent = fs.readFileSync(filePath, 'utf8');
        const lines = csvContent.trim().split('\n');
        const headers = lines[0].split(',');
        const data = [];
        
        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            const rowData = {};
            
            headers.forEach((header, index) => {
                rowData[header] = row[index];
            });
            
            data.push(rowData);
        }
        
        return { headers, data };
        
    } catch (error) {
        console.error(`Error reading CSV file ${filePath}:`, error);
        return null;
    }
}

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`Transaction Analyzer Server running on http://localhost:${PORT}`);
    console.log(`Available testcases: ${scanForTestcases().length}`);
});
