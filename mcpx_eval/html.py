def visualize_json(data, output_path=None):
    """Create an interactive HTML visualization of JSON data"""
    import json
    from datetime import datetime
    import matplotlib.pyplot as plt
    import io
    import base64

    def create_performance_graph(data):
        """Create a matplotlib graph of model performance"""
        if not data.get("total", {}).get("models"):
            return None

        models = data["total"]["models"]
        model_names = list(models.keys())
        accuracies = [models[m]["accuracy"] for m in model_names]
        tool_scores = [models[m]["tool_use"] for m in model_names]

        # Sort by overall score
        sorted_indices = sorted(range(len(tool_scores)), key=lambda k: tool_scores[k], reverse=True)
        model_names = [model_names[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        tool_scores = [tool_scores[i] for i in sorted_indices]

        plt.figure(figsize=(12, 6))
        x = range(len(model_names))
        width = 0.35

        plt.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='skyblue')
        plt.bar([i + width/2 for i in x], tool_scores, width, label='Tool Use', color='lightgreen')

        plt.xlabel('Models')
        plt.ylabel('Score (%)')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # Create HTML content with comparison tables and JSON viewer
    html = (
        """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>mcpx-eval Scoreboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 90vw;
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
                cursor: pointer;
                position: relative;
            }
            th:hover {
                background-color: #e6e6e6;
            }
            th::after {
                content: '↕';
                position: absolute;
                right: 8px;
                opacity: 0.5;
            }
            th.asc::after {
                content: '↑';
                opacity: 1;
            }
            th.desc::after {
                content: '↓';
                opacity: 1;
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
            /* Removed hallucination-details styling */
        </style>
    </head>
    <body>
        <h1>mcpx-eval Open-Ended Tool Calling Scoreboard</h1>
        <div class="timestamp">Generated on: """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """</div>

        <div class="container">
            <h2>Overview</h2>
            <img src="data:image/png;base64,"""
        + create_performance_graph(data)
        + """" alt="Model Performance Graph" style="width:100%; max-width:1000px; display:block; margin:0 auto;">
        </div>

        <div id="comparison-tab">
                <div id="overall-rankings" style="display: none;">
                    <div class="container">
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
                                    <th>Duration (s)</th>
                                    <th>Tool Calls</th>
                                    <th>Redundant Calls</th>
                                    <th>Failed Calls</th>
                                </tr>
                            </thead>
                            <tbody id="overall-table-body">
                                <!-- Filled by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Individual Test Results -->
                <div id="test-results">
                    <!-- Filled by JavaScript -->
                </div>
            </div>
        </div>

        <script>
            // The JSON data
            const jsonData = """
        + json.dumps(data)
        + """;

            // Format number as percentage
            function formatPercent(value) {
                if (typeof value !== 'number') return 'N/A';
                return value.toFixed(3) + '%';
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
                    hallucination_score: models.map(m => m.hallucination_score),
                    duration: models.map(m => m.duration || 0),
                    tool_calls: models.map(m => m.tool_calls || 0),
                    redundant_tool_calls: models.map(m => m.redundant_tool_calls || 0),
                    failed_tool_calls: models.map(m => m.failed_tool_calls || 0)
                };

                // Find best/worst values
                const bestWorst = {
                    avgScore: findBestWorst(allValues.avgScore),
                    accuracy: findBestWorst(allValues.accuracy),
                    tool_use: findBestWorst(allValues.tool_use),
                    clarity: findBestWorst(allValues.clarity),
                    helpfulness: findBestWorst(allValues.helpfulness),
                    overall: findBestWorst(allValues.overall),
                    hallucination_score: findBestWorst(allValues.hallucination_score, false),
                    duration: findBestWorst(allValues.duration, false),
                    tool_calls: findBestWorst(allValues.tool_calls, false),
                    redundant_tool_calls: findBestWorst(allValues.redundant_tool_calls, false),
                    failed_tool_calls: findBestWorst(allValues.failed_tool_calls, false)
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

                    // Duration
                    const durationCell = document.createElement('td');
                    durationCell.textContent = (model.duration || 0).toFixed(3);
                    if (model.duration === bestWorst.duration.best) durationCell.className = 'best';
                    else if (bestWorst.duration.best !== bestWorst.duration.worst &&
                            model.duration === bestWorst.duration.worst) durationCell.className = 'worst';
                    row.appendChild(durationCell);

                    // Tool Calls
                    const toolCallsCell = document.createElement('td');
                    toolCallsCell.textContent = (model.tool_calls || 0).toFixed(1);
                    if (model.tool_calls === bestWorst.tool_calls.best) toolCallsCell.className = 'best';
                    else if (bestWorst.tool_calls.best !== bestWorst.tool_calls.worst &&
                            model.tool_calls === bestWorst.tool_calls.worst) toolCallsCell.className = 'worst';
                    row.appendChild(toolCallsCell);

                    // Redundant Calls
                    const redundantCallsCell = document.createElement('td');
                    redundantCallsCell.textContent = (model.redundant_tool_calls || 0).toFixed(1);
                    if (model.redundant_tool_calls === bestWorst.redundant_tool_calls.best) redundantCallsCell.className = 'best';
                    else if (bestWorst.redundant_tool_calls.best !== bestWorst.redundant_tool_calls.worst &&
                            model.redundant_tool_calls === bestWorst.redundant_tool_calls.worst) redundantCallsCell.className = 'worst';
                    row.appendChild(redundantCallsCell);

                    // Failed Calls
                    const failedCallsCell = document.createElement('td');
                    failedCallsCell.textContent = (model.failed_tool_calls || 0).toFixed(1);
                    if (model.failed_tool_calls === bestWorst.failed_tool_calls.best && model.failed_tool_calls === 0) failedCallsCell.className = 'best';
                    else if (model.failed_tool_calls === bestWorst.failed_tool_calls.worst && model.failed_tool_calls > 0) failedCallsCell.className = 'worst';
                    row.appendChild(failedCallsCell);

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
                        hallucination_score: models.map(m => m.hallucination_score),
                        duration: models.map(m => m.duration || 0),
                        tool_calls: models.map(m => m.tool_calls || 0),
                        redundant_tool_calls: models.map(m => m.redundant_tool_calls || 0),
                        failed_tool_calls: models.map(m => m.failed_tool_calls || 0)
                    };

                    // Find best/worst values
                    const bestWorst = {
                        avgScore: findBestWorst(allValues.avgScore),
                        accuracy: findBestWorst(allValues.accuracy),
                        tool_use: findBestWorst(allValues.tool_use),
                        clarity: findBestWorst(allValues.clarity),
                        helpfulness: findBestWorst(allValues.helpfulness),
                        overall: findBestWorst(allValues.overall),
                        hallucination_score: findBestWorst(allValues.hallucination_score, false),
                        duration: findBestWorst(allValues.duration, false),
                        tool_calls: findBestWorst(allValues.tool_calls, false),
                        redundant_tool_calls: findBestWorst(allValues.redundant_tool_calls, false),
                        failed_tool_calls: findBestWorst(allValues.failed_tool_calls, false)
                    };

                    // Create table
                    const table = document.createElement('table');

                    // Create table header
                    const thead = document.createElement('thead');
                    const headerRow = document.createElement('tr');

                    ['Rank', 'Model', 'Average Score', 'Accuracy', 'Tool Use', 'Clarity', 'Helpfulness',
                     'Overall', 'Hallucination', 'Duration (s)', 'Tool Calls', 'Redundant Calls', 'Failed Calls'].forEach(header => {
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
                                model.avgScore === bestWorst.avgScore.worst &&
                                (bestWorst.avgScore.best - model.avgScore) >= 30) avgScoreCell.className = 'worst';
                        row.appendChild(avgScoreCell);

                        // Accuracy
                        const accuracyCell = document.createElement('td');
                        accuracyCell.textContent = formatPercent(model.accuracy);
                        if (model.accuracy === bestWorst.accuracy.best) accuracyCell.className = 'best';
                        else if (bestWorst.accuracy.best !== bestWorst.accuracy.worst &&
                                model.accuracy === bestWorst.accuracy.worst &&
                                (bestWorst.accuracy.best - model.accuracy) >= 30) accuracyCell.className = 'worst';
                        row.appendChild(accuracyCell);

                        // Tool Use
                        const toolUseCell = document.createElement('td');
                        toolUseCell.textContent = formatPercent(model.tool_use);
                        if (model.tool_use === bestWorst.tool_use.best) toolUseCell.className = 'best';
                        else if (bestWorst.tool_use.best !== bestWorst.tool_use.worst &&
                                model.tool_use === bestWorst.tool_use.worst &&
                                (bestWorst.tool_use.best - model.tool_use) >= 30) toolUseCell.className = 'worst';
                        row.appendChild(toolUseCell);

                        // Clarity
                        const clarityCell = document.createElement('td');
                        clarityCell.textContent = formatPercent(model.clarity);
                        if (model.clarity === bestWorst.clarity.best) clarityCell.className = 'best';
                        else if (bestWorst.clarity.best !== bestWorst.clarity.worst &&
                                model.clarity === bestWorst.clarity.worst &&
                                (bestWorst.clarity.best - model.clarity) >= 30) clarityCell.className = 'worst';
                        row.appendChild(clarityCell);

                        // Helpfulness
                        const helpfulnessCell = document.createElement('td');
                        helpfulnessCell.textContent = formatPercent(model.helpfulness);
                        if (model.helpfulness === bestWorst.helpfulness.best) helpfulnessCell.className = 'best';
                        else if (bestWorst.helpfulness.best !== bestWorst.helpfulness.worst &&
                                model.helpfulness === bestWorst.helpfulness.worst &&
                                (bestWorst.helpfulness.best - model.helpfulness) >= 30) helpfulnessCell.className = 'worst';
                        row.appendChild(helpfulnessCell);

                        // Overall
                        const overallCell = document.createElement('td');
                        overallCell.textContent = formatPercent(model.overall);
                        if (model.overall === bestWorst.overall.best) overallCell.className = 'best';
                        else if (bestWorst.overall.best !== bestWorst.overall.worst &&
                                model.overall === bestWorst.overall.worst &&
                                (bestWorst.overall.best - model.overall) >= 30) overallCell.className = 'worst';
                        row.appendChild(overallCell);

                        // Hallucination
                        const hallucinationCell = document.createElement('td');
                        hallucinationCell.textContent = formatPercent(model.hallucination_score);
                        if (model.hallucination_score === bestWorst.hallucination_score.best) hallucinationCell.className = 'best';
                        else if (bestWorst.hallucination_score.best !== bestWorst.hallucination_score.worst &&
                                model.hallucination_score === bestWorst.hallucination_score.worst) hallucinationCell.className = 'worst';
                        row.appendChild(hallucinationCell);

                        // Duration
                        const durationCell = document.createElement('td');
                        durationCell.textContent = (model.duration || 0).toFixed(3);
                        if (model.duration === bestWorst.duration.best) durationCell.className = 'best';
                        else if (bestWorst.duration.best !== bestWorst.duration.worst &&
                                model.duration === bestWorst.duration.worst) durationCell.className = 'worst';
                        row.appendChild(durationCell);

                        // Tool Calls
                        const toolCallsCell = document.createElement('td');
                        toolCallsCell.textContent = ((model.tool_calls || 0).toFixed(1));
                        if (model.tool_calls === bestWorst.tool_calls.best) toolCallsCell.className = 'best';
                        else if (bestWorst.tool_calls.best !== bestWorst.tool_calls.worst &&
                                model.tool_calls === bestWorst.tool_calls.worst) toolCallsCell.className = 'worst';
                        row.appendChild(toolCallsCell);

                        // Redundant Tool Calls
                        const redundantCallsCell = document.createElement('td');
                        redundantCallsCell.textContent = ((model.redundant_tool_calls || 0).toFixed(1));
                        if (model.redundant_tool_calls === bestWorst.redundant_tool_calls.best) redundantCallsCell.className = 'best';
                        else if (bestWorst.redundant_tool_calls.best !== bestWorst.redundant_tool_calls.worst &&
                                model.redundant_tool_calls === bestWorst.redundant_tool_calls.worst) redundantCallsCell.className = 'worst';
                        row.appendChild(redundantCallsCell);

                        // Failed Tool Calls
                        const failedCallsCell = document.createElement('td');
                        failedCallsCell.textContent = ((model.failed_tool_calls || 0).toFixed(1));
                        if (model.failed_tool_calls === bestWorst.failed_tool_calls.best && model.failed_tool_calls === 0) failedCallsCell.className = 'best';
                        else if (model.failed_tool_calls === bestWorst.failed_tool_calls.worst && model.failed_tool_calls > 0) failedCallsCell.className = 'worst';
                        row.appendChild(failedCallsCell);

                        tbody.appendChild(row);

                        // False claims are not displayed in this view
                    });

                    table.appendChild(tbody);
                    testContainer.appendChild(table);
                    testResultsContainer.appendChild(testContainer);
                });
            }

            // Sort table by column
            function sortTable(table, columnIndex, asc = true) {
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));

                // Clear all sort indicators
                table.querySelectorAll('th').forEach(th => {
                    th.classList.remove('asc', 'desc');
                });

                // Add sort indicator to current column
                const th = table.querySelectorAll('th')[columnIndex];
                th.classList.add(asc ? 'asc' : 'desc');

                // Sort rows
                const sortedRows = rows.sort((a, b) => {
                    const aCol = a.querySelectorAll('td')[columnIndex];
                    const bCol = b.querySelectorAll('td')[columnIndex];

                    let aValue = aCol.textContent.trim();
                    let bValue = bCol.textContent.trim();

                    // Convert percentage strings to numbers
                    if (aValue.endsWith('%')) {
                        aValue = parseFloat(aValue);
                        bValue = parseFloat(bValue);
                    }
                    // Convert numeric strings to numbers
                    else if (!isNaN(aValue)) {
                        aValue = parseFloat(aValue);
                        bValue = parseFloat(bValue);
                    }

                    if (aValue < bValue) return asc ? -1 : 1;
                    if (aValue > bValue) return asc ? 1 : -1;
                    return 0;
                });

                // Update row order
                tbody.innerHTML = '';
                sortedRows.forEach(row => tbody.appendChild(row));

                // Update ranks if sorting by a metric column
                if (columnIndex > 1) {
                    sortedRows.forEach((row, index) => {
                        row.querySelector('td').textContent = index + 1;
                    });
                }
            }

            // Add click handlers to table headers
            function addTableSorting(table) {
                const headers = table.querySelectorAll('th');
                headers.forEach((header, index) => {
                    header.addEventListener('click', () => {
                        const isAsc = !header.classList.contains('asc');
                        sortTable(table, index, isAsc);
                    });
                });
            }

            // Initialize the page
            document.addEventListener('DOMContentLoaded', function() {
                // Only show overall rankings if there is more than one test
                const testCount = Object.keys(jsonData.tests || {}).length;
                if (testCount > 1) {
                    document.getElementById('overall-rankings').style.display = 'block';
                    populateOverallTable();
                    addTableSorting(document.getElementById('overall-table'));
                }

                createTestTables();
                // Add sorting to all test tables
                document.querySelectorAll('#test-results table').forEach(table => {
                    addTableSorting(table);
                });
            });
        </script>
    </body>
    </html>
    """
    )

    return html
