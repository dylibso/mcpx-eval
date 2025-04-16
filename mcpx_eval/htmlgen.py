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
            return ""

        models = data["total"]["models"]
        model_names = list(models.keys())
        metrics = {
            "accuracy": [models[m]["accuracy"] for m in model_names],
            "tool_use": [models[m]["tool_use"] for m in model_names],
            "completeness": [models[m]["completeness"] for m in model_names],
            "quality": [models[m]["quality"] for m in model_names],
            "hallucination": [models[m]["hallucination_score"] for m in model_names],
        }

        # Sort by quality score
        sorted_indices = sorted(
            range(len(metrics["quality"])),
            key=lambda k: metrics["quality"][k],
            reverse=True,
        )
        model_names = [model_names[i] for i in sorted_indices]
        for metric in metrics:
            metrics[metric] = [metrics[metric][i] for i in sorted_indices]

        plt.figure(figsize=(15, 8))
        x = range(len(model_names))
        width = 0.15  # Narrower bars to fit all metrics

        # Plot each metric with offset positions
        plt.bar(
            [i - width * 2 for i in x],
            metrics["accuracy"],
            width,
            label="Accuracy",
            color="skyblue",
        )
        plt.bar(
            [i - width for i in x],
            metrics["tool_use"],
            width,
            label="Tool Use",
            color="lightgreen",
        )
        plt.bar(
            [i for i in x],
            metrics["completeness"],
            width,
            label="Completeness",
            color="orange",
        )
        plt.bar(
            [i + width for i in x],
            metrics["quality"],
            width,
            label="Quality",
            color="purple",
        )
        plt.bar(
            [i + width * 2 for i in x],
            metrics["hallucination"],
            width,
            label="Hallucination",
            color="red",
        )

        plt.xlabel("Models", fontsize=12)
        plt.ylabel("Score (%)", fontsize=12)
        plt.xticks(x, model_names, rotation=45, ha="right", fontsize=14)
        plt.legend(loc="upper right", title="Metrics", fontsize=10)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

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
                                    <th>Combined Score</th>
                                    <th>Accuracy</th>
                                    <th>Tool Use</th>
                                    <th>Completeness</th>
                                    <th>Quality</th>
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
                        data.completeness,
                        data.quality
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
                    completeness: models.map(m => m.completeness),
                    quality: models.map(m => m.quality),
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
                    completeness: findBestWorst(allValues.completeness),
                    quality: findBestWorst(allValues.quality),
                    hallucination_score: findBestWorst(allValues.hallucination_score, false),
                    duration: findBestWorst(allValues.duration, false),
                    tool_calls: findBestWorst(allValues.tool_calls, false),
                    redundant_tool_calls: findBestWorst(allValues.redundant_tool_calls, false),
                    failed_tool_calls: findBestWorst(allValues.failed_tool_calls, false)
                };

                // Generate and add rows
                const rows = generateModelRows(models, bestWorst);
                rows.forEach(row => tableBody.appendChild(row));
            }

            // Helper function to generate table rows for models
            function generateModelRows(models, bestWorst) {
                const rows = [];
                
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

                    // Completeness
                    const completenessCell = document.createElement('td');
                    completenessCell.textContent = formatPercent(model.completeness);
                    if (model.completeness === bestWorst.completeness.best) completenessCell.className = 'best';
                    else if (bestWorst.completeness.best !== bestWorst.completeness.worst &&
                            model.completeness === bestWorst.completeness.worst) completenessCell.className = 'worst';
                    row.appendChild(completenessCell);

                    // Quality
                    const qualityCell = document.createElement('td');
                    qualityCell.textContent = formatPercent(model.quality);
                    if (model.quality === bestWorst.quality.best) qualityCell.className = 'best';
                    else if (bestWorst.quality.best !== bestWorst.quality.worst &&
                            model.quality === bestWorst.quality.worst) qualityCell.className = 'worst';
                    row.appendChild(qualityCell);

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
                    // Don't highlight tool calls - it's not inherently better/worse to use more/fewer tools
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

                    rows.push(row);
                });

                return rows;
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
                            data.completeness,
                            data.quality
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
                        completeness: models.map(m => m.completeness),
                        quality: models.map(m => m.quality),
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
                        completeness: findBestWorst(allValues.completeness),
                        quality: findBestWorst(allValues.quality),
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

                    ['Rank', 'Model', 'Combined Score', 'Accuracy', 'Tool Use', 'Completeness', 'Quality',
                     'Hallucination', 'Duration (s)', 'Tool Calls', 'Redundant Calls', 'Failed Calls'].forEach(header => {
                        const th = document.createElement('th');
                        th.textContent = header;
                        headerRow.appendChild(th);
                    });

                    thead.appendChild(headerRow);
                    table.appendChild(thead);

                    // Create table body
                    const tbody = document.createElement('tbody');

                    // Generate and add rows
                    const rows = generateModelRows(models, bestWorst);
                    rows.forEach(row => tbody.appendChild(row));

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
