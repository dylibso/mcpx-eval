from . import Judge, Test, Database
import asyncio
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def print_result(result):
    print()
    print(f"{result.model}")
    print("=" * len(result.model))
    print()
    print(f"Time: {result.duration}s")
    print("Output:")
    print(result.llm_output)
    print()
    print("Score:")
    print(result.description)

    print("Number of tool calls:", result.tool_calls)
    if result.redundant_tool_calls > 0:
        print("Redundant tool calls:", result.redundant_tool_calls)
    print("Tool use:", result.tool_use)
    print("Accuracy:", result.accuracy)
    print("Clarity:", result.clarity)
    print("Helpfulness:", result.helpfulness)

    # Hallucination metrics
    print("Hallucination score:", result.hallucination_score)
    if result.false_claims and len(result.false_claims) > 0:
        print("False claims detected:")
        for claim in result.false_claims:
            print(f"  - {claim}")

    # Tool analysis
    if result.tool_analysis and len(result.tool_analysis) > 0:
        print("\nTool analysis:")
        for tool_id, analysis in result.tool_analysis.items():
            if isinstance(analysis, list):
                for a in analysis:
                    print(f"  {tool_id}: {a['name']}")
                    print(f"    - Redundancy: {a['redundancy']}")
            else:
                print(f"  {tool_id}: {analysis['name']}")
                print(f"    - Redundancy: {analysis['redundancy']}")

    print("\nOverall score:", result.overall)


def summary(args):
    db = Database()
    res = db.average_results(args.name)
    for result in res.scores:
        print_result(result)


def generate_table(args):
    """Generate an HTML table from test results"""
    db = Database()

    # Parse test names from comma-separated string
    test_names = [name.strip() for name in args.tests.split(",")]

    # Prepare HTML content
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .model-header { background-color: #e6f2ff; font-weight: bold; }
        .highlight { background-color: #ffffcc; }
        .best { font-weight: bold; color: #006600; }
        h1, h2 { color: #333; }
        .metric-group { margin-top: 10px; border-top: 2px solid #ccc; padding-top: 10px; }
        .hallucination { color: #cc0000; }
    </style>
</head>
<body>
    <h1>LLM Evaluation Results</h1>
"""

    # Generate a table for each test
    for test_name in test_names:
        results = db.average_results(test_name)

        scores_with_avg = []
        # Calculate average scores for each model
        for score in results.scores:
            numeric_scores = [
                score.accuracy,
                score.tool_use,
                score.clarity,
                score.helpfulness,
                score.overall
            ]
            scores_with_avg.append((score, sum(numeric_scores) / len(numeric_scores)))
        
        # Sort models by average score (highest first)
        scores_with_avg.sort(key=lambda x: x[1], reverse=True)

        html += f"""
    <h2>Test: {test_name}</h2>
    <table>
        <tr>
            <th>Model</th>
"""

        # Define metrics for vertical table
        all_metrics = {
            "Average Score": (lambda _, a: a, True),
            "Accuracy": (lambda s, _: s.accuracy, True),
            "Tool Use": (lambda s, _: s.tool_use, True),
            "Clarity": (lambda s, _: s.clarity, True),
            "Helpfulness": (lambda s, _: s.helpfulness, True),
            "Overall": (lambda s, _: s.overall, True),
            "Hallucination Score": (lambda s, _: s.hallucination_score, False),
            "Tool Calls": (lambda s, _: s.tool_calls, None),
            "Redundant Tool Calls": (lambda s, _: s.redundant_tool_calls, None),
        }

        # Add metrics as column headers
        for metric_name in all_metrics:
            html += f"<th>{metric_name}</th>\n"
        html += "</tr>\n"

        # Add a row for each model
        for i, (score, avg) in enumerate(scores_with_avg):
            # Set row class
            row_class = "model-header"
            
            html += f'<tr class="{row_class}">'
            html += f"<td><strong>{score.model}</strong></td>"

            # Add values for each metric
            for metric_name, (get_value, higher_is_better) in all_metrics.items():
                value = get_value(score, avg)
                
                # Format based on type
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                # Determine if this is the best value across all models for this metric
                is_best = False
                if higher_is_better is not None:
                    all_values = [get_value(s, a) for s, a in scores_with_avg]
                    best_value = max(all_values) if higher_is_better else min(all_values)
                    # Mark as best if this model has the best value (allow ties)
                    is_best = value == best_value
                
                if is_best:
                    html += f'<td class="best">{formatted_value}</td>'
                else:
                    html += f"<td>{formatted_value}</td>"
            
            html += "</tr>\n"

        # Close the table
        html += "</table>\n"

        # If there are false claims, add them
        # for score in results.scores:
        #     if score.false_claims and len(score.false_claims) > 0:
        #         html += f"""
        # <h3 class="hallucination">{score.model} - False Claims Detected:</h3>
        # <ul>
        # """
        #         for claim in score.false_claims:
        #             html += f"<li>{claim}</li>\n"
        #         html += "</ul>\n"

    # Close HTML
    html += """
</body>
</html>
"""

    # Save to file
    output_path = args.output or f"results_{'-'.join(test_names)}.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"HTML table saved to {output_path}")


# Visualization functions removed


def json_summary(args):
    """Generate a JSON summary of all test data"""
    import json
    
    db = Database()
    summary = db.generate_json_summary()
    
    # Format JSON with indentation for readability
    formatted_json = json.dumps(summary, indent=2)
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(formatted_json)
        print(f"JSON summary saved to {args.output}")
        print(f"To visualize this file, run: uv run python -m mcpx_eval viz-json {args.output}")
    else:
        print(formatted_json)
    
    # If visualization is requested, create and open it
    if args.visualize:
        visualize_json(summary, args.viz_output)

def visualize_json(data, output_path=None):
    """Create an interactive HTML visualization of JSON data"""
    import json
    import webbrowser
    from tempfile import NamedTemporaryFile
    from datetime import datetime
    
    # Create HTML content with comparison tables and JSON viewer
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>MCPX Evaluation JSON Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }
            h1, h2, h3 {
                color: #333;
                text-align: center;
            }
            h1 {
                margin-bottom: 20px;
            }
            h2 {
                margin-top: 40px;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .timestamp {
                text-align: center;
                color: #777;
                font-size: 0.9em;
                margin-bottom: 20px;
            }
            /* Table Styles */
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .model-header {
                background-color: #e6f2ff;
                font-weight: bold;
            }
            .best {
                font-weight: bold;
                color: #006600;
            }
            .worst {
                color: #cc0000;
            }
            .metric-name {
                font-weight: bold;
            }
            .false-claims {
                margin-top: 5px;
                font-size: 0.9em;
                color: #cc0000;
            }
            .summary-card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
            }
            .summary-item {
                text-align: center;
                flex: 1;
            }
            .summary-label {
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }
            .summary-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }
            /* Removed hallucination-details styling */
        </style>
    </head>
    <body>
        <h1>MCPX Evaluation JSON Visualization</h1>
        <div class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
        
        <div id="comparison-tab">
                <!-- Overall Summary -->
                <div class="container">
                    <h2>Overall Summary</h2>
                    <div class="summary-card">
                        <div class="summary-item">
                            <div class="summary-label">Tests</div>
                            <div class="summary-value" id="total-tests"></div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Models</div>
                            <div class="summary-value" id="total-models"></div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Avg Accuracy</div>
                            <div class="summary-value" id="avg-accuracy"></div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Avg Tool Use</div>
                            <div class="summary-value" id="avg-tool-use"></div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Avg Overall</div>
                            <div class="summary-value" id="avg-overall"></div>
                        </div>
                    </div>
                    
                    <h3>Model Rankings (All Tests)</h3>
                    <table id="overall-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Model</th>
                                <th>Average Score</th>
                                <th>Accuracy</th>
                                <th>Tool Use</th>
                                <th>Clarity</th>
                                <th>Helpfulness</th>
                                <th>Overall</th>
                                <th>Hallucination</th>
                                <th>Tool Calls</th>
                                <th>Tests</th>
                            </tr>
                        </thead>
                        <tbody id="overall-table-body">
                            <!-- Filled by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <!-- Individual Test Results -->
                <div id="test-results">
                    <!-- Filled by JavaScript -->
                </div>
            </div>
        </div>
        
        <script>
            // The JSON data
            const jsonData = """ + json.dumps(data) + """;
            
            // Format number as percentage
            function formatPercent(value) {
                if (typeof value !== 'number') return 'N/A';
                return value.toFixed(2) + '%';
            }
            
            // Find best and worst values in an array
            function findBestWorst(values, higherIsBetter = true) {
                if (!values.length) return { best: null, worst: null };
                
                const numValues = values.filter(v => typeof v === 'number');
                if (!numValues.length) return { best: null, worst: null };
                
                if (higherIsBetter) {
                    return {
                        best: Math.max(...numValues),
                        worst: Math.min(...numValues)
                    };
                } else {
                    return {
                        best: Math.min(...numValues),
                        worst: Math.max(...numValues)
                    };
                }
            }
            
            // Calculate average of numeric values
            function calculateAverage(values) {
                const numValues = values.filter(v => typeof v === 'number');
                if (!numValues.length) return 0;
                return numValues.reduce((sum, val) => sum + val, 0) / numValues.length;
            }
            
            // Populate the overall summary
            function populateOverallSummary() {
                const totalTests = jsonData.test_count || Object.keys(jsonData.tests || {}).length;
                const totalModels = jsonData.model_count || Object.keys(jsonData.total?.models || {}).length;
                
                document.getElementById('total-tests').textContent = totalTests;
                document.getElementById('total-models').textContent = totalModels;
                
                if (jsonData.total && jsonData.total.metrics) {
                    document.getElementById('avg-accuracy').textContent = formatPercent(jsonData.total.metrics.accuracy);
                    document.getElementById('avg-tool-use').textContent = formatPercent(jsonData.total.metrics.tool_use);
                    document.getElementById('avg-overall').textContent = formatPercent(jsonData.total.metrics.overall);
                }
            }
            
            // Populate the overall model rankings table
            function populateOverallTable() {
                const tableBody = document.getElementById('overall-table-body');
                tableBody.innerHTML = '';
                
                if (!jsonData.total || !jsonData.total.models) return;
                
                // Get models and calculate average scores
                const models = Object.entries(jsonData.total.models).map(([name, data]) => {
                    const avgScore = calculateAverage([
                        data.accuracy,
                        data.tool_use,
                        data.clarity,
                        data.helpfulness,
                        data.overall
                    ]);
                    
                    return {
                        name,
                        avgScore,
                        ...data
                    };
                });
                
                // Sort by average score (highest first)
                models.sort((a, b) => b.avgScore - a.avgScore);
                
                // Get all values for each metric to determine best/worst
                const allValues = {
                    avgScore: models.map(m => m.avgScore),
                    accuracy: models.map(m => m.accuracy),
                    tool_use: models.map(m => m.tool_use),
                    clarity: models.map(m => m.clarity),
                    helpfulness: models.map(m => m.helpfulness),
                    overall: models.map(m => m.overall),
                    hallucination_score: models.map(m => m.hallucination_score)
                };
                
                // Find best/worst values
                const bestWorst = {
                    avgScore: findBestWorst(allValues.avgScore),
                    accuracy: findBestWorst(allValues.accuracy),
                    tool_use: findBestWorst(allValues.tool_use),
                    clarity: findBestWorst(allValues.clarity),
                    helpfulness: findBestWorst(allValues.helpfulness),
                    overall: findBestWorst(allValues.overall),
                    hallucination_score: findBestWorst(allValues.hallucination_score, false)
                };
                
                // Add rows to the table
                models.forEach((model, index) => {
                    const row = document.createElement('tr');
                    row.className = 'model-header';
                    
                    // Rank
                    const rankCell = document.createElement('td');
                    rankCell.textContent = index + 1;
                    row.appendChild(rankCell);
                    
                    // Model name
                    const nameCell = document.createElement('td');
                    nameCell.textContent = model.name;
                    row.appendChild(nameCell);
                    
                    // Average score
                    const avgScoreCell = document.createElement('td');
                    avgScoreCell.textContent = formatPercent(model.avgScore);
                    if (model.avgScore === bestWorst.avgScore.best) avgScoreCell.className = 'best';
                    else if (bestWorst.avgScore.best !== bestWorst.avgScore.worst && 
                            model.avgScore === bestWorst.avgScore.worst) avgScoreCell.className = 'worst';
                    row.appendChild(avgScoreCell);
                    
                    // Accuracy
                    const accuracyCell = document.createElement('td');
                    accuracyCell.textContent = formatPercent(model.accuracy);
                    if (model.accuracy === bestWorst.accuracy.best) accuracyCell.className = 'best';
                    else if (bestWorst.accuracy.best !== bestWorst.accuracy.worst && 
                            model.accuracy === bestWorst.accuracy.worst) accuracyCell.className = 'worst';
                    row.appendChild(accuracyCell);
                    
                    // Tool Use
                    const toolUseCell = document.createElement('td');
                    toolUseCell.textContent = formatPercent(model.tool_use);
                    if (model.tool_use === bestWorst.tool_use.best) toolUseCell.className = 'best';
                    else if (bestWorst.tool_use.best !== bestWorst.tool_use.worst && 
                            model.tool_use === bestWorst.tool_use.worst) toolUseCell.className = 'worst';
                    row.appendChild(toolUseCell);
                    
                    // Clarity
                    const clarityCell = document.createElement('td');
                    clarityCell.textContent = formatPercent(model.clarity);
                    if (model.clarity === bestWorst.clarity.best) clarityCell.className = 'best';
                    else if (bestWorst.clarity.best !== bestWorst.clarity.worst && 
                            model.clarity === bestWorst.clarity.worst) clarityCell.className = 'worst';
                    row.appendChild(clarityCell);
                    
                    // Helpfulness
                    const helpfulnessCell = document.createElement('td');
                    helpfulnessCell.textContent = formatPercent(model.helpfulness);
                    if (model.helpfulness === bestWorst.helpfulness.best) helpfulnessCell.className = 'best';
                    else if (bestWorst.helpfulness.best !== bestWorst.helpfulness.worst && 
                            model.helpfulness === bestWorst.helpfulness.worst) helpfulnessCell.className = 'worst';
                    row.appendChild(helpfulnessCell);
                    
                    // Overall
                    const overallCell = document.createElement('td');
                    overallCell.textContent = formatPercent(model.overall);
                    if (model.overall === bestWorst.overall.best) overallCell.className = 'best';
                    else if (bestWorst.overall.best !== bestWorst.overall.worst && 
                            model.overall === bestWorst.overall.worst) overallCell.className = 'worst';
                    row.appendChild(overallCell);
                    
                    // Hallucination
                    const hallucinationCell = document.createElement('td');
                    hallucinationCell.textContent = formatPercent(model.hallucination_score);
                    if (model.hallucination_score === bestWorst.hallucination_score.best) hallucinationCell.className = 'best';
                    else if (bestWorst.hallucination_score.best !== bestWorst.hallucination_score.worst && 
                            model.hallucination_score === bestWorst.hallucination_score.worst) hallucinationCell.className = 'worst';
                    row.appendChild(hallucinationCell);
                    
                    // Tool Calls
                    const toolCallsCell = document.createElement('td');
                    toolCallsCell.textContent = model.tool_calls || 0;
                    row.appendChild(toolCallsCell);
                    
                    // Test Count
                    const testCountCell = document.createElement('td');
                    testCountCell.textContent = model.test_count || 'N/A';
                    row.appendChild(testCountCell);
                    
                    tableBody.appendChild(row);
                });
            }
            
            // Create tables for each individual test
            function createTestTables() {
                const testResultsContainer = document.getElementById('test-results');
                testResultsContainer.innerHTML = '';
                
                if (!jsonData.tests) return;
                
                // Process each test
                Object.entries(jsonData.tests).forEach(([testName, testData]) => {
                    if (!testData.models || Object.keys(testData.models).length === 0) return;
                    
                    // Create container for this test
                    const testContainer = document.createElement('div');
                    testContainer.className = 'container';
                    
                    // Add test header
                    const testHeader = document.createElement('h2');
                    testHeader.textContent = `Test: ${testName}`;
                    testContainer.appendChild(testHeader);
                    
                    // Get models and calculate average scores
                    const models = Object.entries(testData.models).map(([name, data]) => {
                        const avgScore = calculateAverage([
                            data.accuracy,
                            data.tool_use,
                            data.clarity,
                            data.helpfulness,
                            data.overall
                        ]);
                        
                        return {
                            name,
                            avgScore,
                            ...data
                        };
                    });
                    
                    // Sort by average score (highest first)
                    models.sort((a, b) => b.avgScore - a.avgScore);
                    
                    // Get all values for each metric to determine best/worst
                    const allValues = {
                        avgScore: models.map(m => m.avgScore),
                        accuracy: models.map(m => m.accuracy),
                        tool_use: models.map(m => m.tool_use),
                        clarity: models.map(m => m.clarity),
                        helpfulness: models.map(m => m.helpfulness),
                        overall: models.map(m => m.overall),
                        hallucination_score: models.map(m => m.hallucination_score)
                    };
                    
                    // Find best/worst values
                    const bestWorst = {
                        avgScore: findBestWorst(allValues.avgScore),
                        accuracy: findBestWorst(allValues.accuracy),
                        tool_use: findBestWorst(allValues.tool_use),
                        clarity: findBestWorst(allValues.clarity),
                        helpfulness: findBestWorst(allValues.helpfulness),
                        overall: findBestWorst(allValues.overall),
                        hallucination_score: findBestWorst(allValues.hallucination_score, false)
                    };
                    
                    // Create table
                    const table = document.createElement('table');
                    
                    // Create table header
                    const thead = document.createElement('thead');
                    const headerRow = document.createElement('tr');
                    
                    ['Rank', 'Model', 'Average Score', 'Accuracy', 'Tool Use', 'Clarity', 'Helpfulness', 
                     'Overall', 'Hallucination', 'Tool Calls', 'Redundant Calls'].forEach(header => {
                        const th = document.createElement('th');
                        th.textContent = header;
                        headerRow.appendChild(th);
                    });
                    
                    thead.appendChild(headerRow);
                    table.appendChild(thead);
                    
                    // Create table body
                    const tbody = document.createElement('tbody');
                    
                    // Add rows for each model (without false claims)
                    models.forEach((model, index) => {
                        const row = document.createElement('tr');
                        row.className = 'model-header';
                        
                        // Rank
                        const rankCell = document.createElement('td');
                        rankCell.textContent = index + 1;
                        row.appendChild(rankCell);
                        
                        // Model name
                        const nameCell = document.createElement('td');
                        nameCell.textContent = model.name;
                        row.appendChild(nameCell);
                        
                        // Average score
                        const avgScoreCell = document.createElement('td');
                        avgScoreCell.textContent = formatPercent(model.avgScore);
                        if (model.avgScore === bestWorst.avgScore.best) avgScoreCell.className = 'best';
                        else if (bestWorst.avgScore.best !== bestWorst.avgScore.worst && 
                                model.avgScore === bestWorst.avgScore.worst) avgScoreCell.className = 'worst';
                        row.appendChild(avgScoreCell);
                        
                        // Accuracy
                        const accuracyCell = document.createElement('td');
                        accuracyCell.textContent = formatPercent(model.accuracy);
                        if (model.accuracy === bestWorst.accuracy.best) accuracyCell.className = 'best';
                        else if (bestWorst.accuracy.best !== bestWorst.accuracy.worst && 
                                model.accuracy === bestWorst.accuracy.worst) accuracyCell.className = 'worst';
                        row.appendChild(accuracyCell);
                        
                        // Tool Use
                        const toolUseCell = document.createElement('td');
                        toolUseCell.textContent = formatPercent(model.tool_use);
                        if (model.tool_use === bestWorst.tool_use.best) toolUseCell.className = 'best';
                        else if (bestWorst.tool_use.best !== bestWorst.tool_use.worst && 
                                model.tool_use === bestWorst.tool_use.worst) toolUseCell.className = 'worst';
                        row.appendChild(toolUseCell);
                        
                        // Clarity
                        const clarityCell = document.createElement('td');
                        clarityCell.textContent = formatPercent(model.clarity);
                        if (model.clarity === bestWorst.clarity.best) clarityCell.className = 'best';
                        else if (bestWorst.clarity.best !== bestWorst.clarity.worst && 
                                model.clarity === bestWorst.clarity.worst) clarityCell.className = 'worst';
                        row.appendChild(clarityCell);
                        
                        // Helpfulness
                        const helpfulnessCell = document.createElement('td');
                        helpfulnessCell.textContent = formatPercent(model.helpfulness);
                        if (model.helpfulness === bestWorst.helpfulness.best) helpfulnessCell.className = 'best';
                        else if (bestWorst.helpfulness.best !== bestWorst.helpfulness.worst && 
                                model.helpfulness === bestWorst.helpfulness.worst) helpfulnessCell.className = 'worst';
                        row.appendChild(helpfulnessCell);
                        
                        // Overall
                        const overallCell = document.createElement('td');
                        overallCell.textContent = formatPercent(model.overall);
                        if (model.overall === bestWorst.overall.best) overallCell.className = 'best';
                        else if (bestWorst.overall.best !== bestWorst.overall.worst && 
                                model.overall === bestWorst.overall.worst) overallCell.className = 'worst';
                        row.appendChild(overallCell);
                        
                        // Hallucination
                        const hallucinationCell = document.createElement('td');
                        hallucinationCell.textContent = formatPercent(model.hallucination_score);
                        if (model.hallucination_score === bestWorst.hallucination_score.best) hallucinationCell.className = 'best';
                        else if (bestWorst.hallucination_score.best !== bestWorst.hallucination_score.worst && 
                                model.hallucination_score === bestWorst.hallucination_score.worst) hallucinationCell.className = 'worst';
                        row.appendChild(hallucinationCell);
                        
                        // Tool Calls
                        const toolCallsCell = document.createElement('td');
                        toolCallsCell.textContent = model.tool_calls || 0;
                        row.appendChild(toolCallsCell);
                        
                        // Redundant Tool Calls
                        const redundantCallsCell = document.createElement('td');
                        redundantCallsCell.textContent = model.redundant_tool_calls || 0;
                        row.appendChild(redundantCallsCell);
                        
                        tbody.appendChild(row);
                        
                        // False claims are not displayed in this view
                    });
                    
                    table.appendChild(tbody);
                    testContainer.appendChild(table);
                    testResultsContainer.appendChild(testContainer);
                });
            }
            
            // Initialize the page
            document.addEventListener('DOMContentLoaded', function() {
                // Populate the comparison tables
                populateOverallSummary();
                populateOverallTable();
                createTestTables();
                
                // Initially collapse all nodes except the first level in JSON view
                const topLevelItems = treeContainer.querySelector('ul').children;
                Array.from(topLevelItems).forEach(item => {
                    const collapsibles = item.querySelectorAll('.collapsible');
                    Array.from(collapsibles).slice(1).forEach(el => {
                        el.classList.add('collapsed');
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Write to temporary file and open in browser
    with NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
        f.write(html)
        temp_path = f.name
    
    print(f"Opening JSON visualization in web browser...")
    webbrowser.open(f"file://{temp_path}")
    
    # Also save a copy to the specified location if provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"JSON visualization saved to {output_path}")
        print(f"To view this visualization again later, open the file in your browser.")

# Display visualization function removed


async def run():
    from argparse import ArgumentParser

    parser = ArgumentParser("mcpx-eval", description="LLM tool use evaluator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Main test command (default)
    test_parser = subparsers.add_parser("test", help="Run evaluation tests")
    test_parser.add_argument("--name", default="", help="Test name")
    test_parser.add_argument(
        "--model",
        "-m",
        default=[],
        help="Model to include in test",
        action="append",
    )
    test_parser.add_argument("--prompt", help="Test prompt")
    test_parser.add_argument("--check", help="Test check")
    test_parser.add_argument(
        "--max-tool-calls", default=None, help="Maximum number of tool calls", type=int
    )
    test_parser.add_argument("--config", help="Test config file")
    test_parser.add_argument(
        "--iter", 
        default=1, 
        type=int,
        help="Number of times to run the test for each model"
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show test results summary")
    summary_parser.add_argument("name", help="Test name to summarize")

    # Table command
    table_parser = subparsers.add_parser("table", help="Generate HTML table of results")
    table_parser.add_argument(
        "--tests",
        "-t",
        required=True,
        help="Comma-separated list of test names to include",
    )
    table_parser.add_argument(
        "--output",
        "-o",
        help="Output HTML file path (default: results_[test-names].html)",
    )
    
    # JSON summary command
    json_parser = subparsers.add_parser("json", help="Generate JSON summary of all test data")
    json_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: print to stdout)",
    )
    json_parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Create an interactive HTML visualization of the JSON data",
    )
    json_parser.add_argument(
        "--viz-output",
        help="Output path for HTML visualization (optional)",
    )
    
    # JSON visualization command (standalone)
    viz_json_parser = subparsers.add_parser("viz-json", help="Visualize JSON data from a file")
    viz_json_parser.add_argument(
        "input",
        help="Input JSON file path",
    )
    viz_json_parser.add_argument(
        "--output",
        "-o",
        help="Output HTML file path (optional)",
    )

    # Visualization commands removed

    # Global options
    parser.add_argument(
        "--log",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = args.log or "INFO"
    log_level = getattr(logging, level, None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % level)
    logging.basicConfig(level=log_level)

    if not args.verbose:
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter("mcpx_eval"))

    # Handle command routing
    command = getattr(args, "command", "test")  # Default to test if not specified

    # Visualization commands removed

    # Summary command
    if command == "summary":
        summary(args)
        return

    # Table command
    elif command == "table":
        generate_table(args)
        return
        
    # JSON summary command
    elif command == "json":
        json_summary(args)
        return
        
    # JSON visualization command
    elif command == "viz-json":
        import json
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
            visualize_json(data, args.output)
        except FileNotFoundError:
            print(f"Error: File '{args.input}' not found.")
            print(f"Generate a JSON file first with: uv run python -m mcpx_eval json -o {args.input}")
            return
        except json.JSONDecodeError:
            print(f"Error: '{args.input}' is not a valid JSON file.")
            return

    # Test command (default)
    else:
        test = None
        if hasattr(args, "config") and args.config is not None:
            test = Test.load(args.config)
            for model in args.model:
                test.models.append(model)
            if args.name is None or args.name == "":
                args.name = test.name

        if test is None:
            test = Test(
                name=getattr(args, "name", ""),
                prompt=args.prompt or "",
                check=args.check or "",
                models=args.model,
                max_tool_calls=args.max_tool_calls,
            )

        iterations = args.iter
        logger.info(f"Running {test.name}: {', '.join(test.models)} ({iterations} iteration{'s' if iterations > 1 else ''})")
        judge = Judge(models=test.models)
        judge.db.save_test(test)
        
        all_results = []
        total_duration = 0
        
        for i in range(iterations):
            if iterations > 1:
                logger.info(f"Iteration {i+1}/{iterations}")
            
            # For multiple iterations, pass save=True to ensure each run is saved to DB
            res = await judge.run_test(test, save=True)
            total_duration += res.duration
            all_results.extend(res.scores)
            
            if iterations > 1:
                logger.info(f"Iteration {i+1} finished in {res.duration}s")
        
        logger.info(f"{test.name} finished in {total_duration}s total")
        
        # When multiple iterations are run, only show the last iteration's results
        # to avoid overwhelming the user with output
        results_to_print = all_results if iterations == 1 else res.scores
        
        if iterations > 1:
            print(f"\nShowing results from iteration {iterations} of {iterations}.")
            print(f"All {iterations} iterations have been saved to the database.")
            print(f"Use 'mcpx-eval summary {test.name}' to see aggregated results.\n")
        
        for result in results_to_print:
            if result is None:
                continue
            print_result(result)


def main():
    asyncio.run(run())
    
    # Print helpful usage examples at the end
    print("\nUsage examples:")
    print("  Generate JSON summary:                uv run python -m mcpx_eval json")
    print("  Generate and visualize JSON:          uv run python -m mcpx_eval json --visualize")
    print("  Save JSON to file:                    uv run python -m mcpx_eval json -o results.json")
    print("  Visualize existing JSON file:         uv run python -m mcpx_eval viz-json results.json")
    print("  Save visualization to HTML:           uv run python -m mcpx_eval viz-json results.json -o viz.html")


if __name__ == "__main__":
    main()
