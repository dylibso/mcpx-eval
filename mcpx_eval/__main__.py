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


def create_visualization(args):
    db = Database()

    # Parse test names from comma-separated string
    test_names = [name.strip() for name in args.tests.split(",")]

    # Get data for each test
    test_data = {}
    for test_name in test_names:
        results = db.average_results(test_name)
        test_data[test_name] = results

    # Create visualization data based on chart type
    chart_data = {}

    if args.type == "bar":
        # Create bar chart comparing models across tests
        models = set()
        metrics = [
            "accuracy",
            "tool_use",
            "tool_calls",
            "redundant_tool_calls",
            "clarity",
            "helpfulness",
            "overall",
            "hallucination_score",
        ]

        for test_name, results in test_data.items():
            for score in results.scores:
                models.add(score.model)

        for metric in metrics:
            chart_data[metric] = {model: {} for model in models}

            for test_name, results in test_data.items():
                for score in results.scores:
                    model_key = score.model
                    chart_data[metric][model_key][test_name] = getattr(score, metric)

    elif args.type == "radar":
        # Create radar chart comparing metrics across models for each test
        for test_name, results in test_data.items():
            chart_data[test_name] = {}
            for score in results.scores:
                model_key = score.model
                chart_data[test_name][model_key] = {
                    "accuracy": score.accuracy,
                    "tool_use": score.tool_use,
                    "clarity": score.clarity,
                    "helpfulness": score.helpfulness,
                    "overall": score.overall,
                    "hallucination_score": score.hallucination_score,
                }

    # Save visualization
    db.create_visualization(
        name=args.name,
        description=args.description,
        test_names=test_names,
        chart_type=args.type,
        chart_data=chart_data,
    )

    print(f"Visualization '{args.name}' created successfully!")


def list_visualizations(args):
    db = Database()
    visualizations = db.get_visualizations()

    if not visualizations:
        print("No visualizations found.")
        return

    print("\nAvailable visualizations:")
    print("=" * 50)

    for viz in visualizations:
        viz_id, name, description, test_names, chart_type = viz
        tests = json.loads(test_names)

        print(f"ID: {viz_id}")
        print(f"Name: {name}")
        print(f"Type: {chart_type}")
        print(f"Tests: {', '.join(tests)}")
        print(f"Description: {description}")
        print("-" * 50)


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
    
    # Create HTML content with JSON viewer
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
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .timestamp {
                text-align: center;
                color: #777;
                font-size: 0.9em;
                margin-bottom: 20px;
            }
            /* JSON Tree Viewer Styles */
            .json-tree {
                font-family: monospace;
                font-size: 14px;
                line-height: 1.4;
            }
            .json-tree ul {
                list-style: none;
                margin: 0;
                padding: 0 0 0 20px;
            }
            .json-tree li {
                position: relative;
            }
            .json-key {
                color: #881391;
                font-weight: bold;
            }
            .json-string {
                color: #1a1aa6;
            }
            .json-number {
                color: #1e7f1e;
            }
            .json-boolean {
                color: #994500;
            }
            .json-null {
                color: #7f7f7f;
            }
            .collapsible {
                cursor: pointer;
                user-select: none;
            }
            .collapsible::before {
                content: "▼";
                display: inline-block;
                margin-right: 5px;
                transition: transform 0.2s;
            }
            .collapsed::before {
                transform: rotate(-90deg);
            }
            .collapsed + ul {
                display: none;
            }
            .search-container {
                margin-bottom: 20px;
                text-align: center;
            }
            #search-input {
                padding: 8px;
                width: 300px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .highlight {
                background-color: yellow;
            }
            .controls {
                margin-bottom: 20px;
                text-align: center;
            }
            button {
                padding: 8px 12px;
                margin: 0 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>MCPX Evaluation JSON Visualization</h1>
        <div class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
        
        <div class="controls">
            <button id="expand-all">Expand All</button>
            <button id="collapse-all">Collapse All</button>
        </div>
        
        <div class="search-container">
            <input type="text" id="search-input" placeholder="Search in JSON...">
        </div>
        
        <div class="container">
            <div id="json-tree" class="json-tree"></div>
        </div>
        
        <script>
            // The JSON data
            const jsonData = """ + json.dumps(data) + """;
            
            // Function to create the JSON tree view
            function createJsonTree(data, container) {
                const ul = document.createElement('ul');
                
                if (Array.isArray(data)) {
                    // Handle array
                    for (let i = 0; i < data.length; i++) {
                        const li = document.createElement('li');
                        
                        if (typeof data[i] === 'object' && data[i] !== null) {
                            const span = document.createElement('span');
                            span.className = 'collapsible';
                            span.innerHTML = `<span class="json-key">[${i}]</span>: `;
                            span.onclick = toggleCollapse;
                            li.appendChild(span);
                            
                            createJsonTree(data[i], li);
                        } else {
                            li.innerHTML = `<span class="json-key">[${i}]</span>: ${formatValue(data[i])}`;
                        }
                        
                        ul.appendChild(li);
                    }
                } else if (typeof data === 'object' && data !== null) {
                    // Handle object
                    for (const key in data) {
                        if (data.hasOwnProperty(key)) {
                            const li = document.createElement('li');
                            
                            if (typeof data[key] === 'object' && data[key] !== null) {
                                const span = document.createElement('span');
                                span.className = 'collapsible';
                                span.innerHTML = `<span class="json-key">${key}</span>: `;
                                span.onclick = toggleCollapse;
                                li.appendChild(span);
                                
                                createJsonTree(data[key], li);
                            } else {
                                li.innerHTML = `<span class="json-key">${key}</span>: ${formatValue(data[key])}`;
                            }
                            
                            ul.appendChild(li);
                        }
                    }
                }
                
                container.appendChild(ul);
            }
            
            // Format values with appropriate styling
            function formatValue(value) {
                if (typeof value === 'string') {
                    return `<span class="json-string">"${escapeHtml(value)}"</span>`;
                } else if (typeof value === 'number') {
                    return `<span class="json-number">${value}</span>`;
                } else if (typeof value === 'boolean') {
                    return `<span class="json-boolean">${value}</span>`;
                } else if (value === null) {
                    return `<span class="json-null">null</span>`;
                }
                return escapeHtml(String(value));
            }
            
            // Escape HTML special characters
            function escapeHtml(text) {
                return text
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }
            
            // Toggle collapse/expand
            function toggleCollapse(event) {
                this.classList.toggle('collapsed');
                event.stopPropagation();
            }
            
            // Initialize the tree
            const treeContainer = document.getElementById('json-tree');
            createJsonTree(jsonData, treeContainer);
            
            // Search functionality
            const searchInput = document.getElementById('search-input');
            searchInput.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                clearHighlights();
                
                if (searchTerm.length > 0) {
                    searchInTree(treeContainer, searchTerm);
                }
            });
            
            function clearHighlights() {
                const highlights = document.querySelectorAll('.highlight');
                highlights.forEach(el => {
                    el.classList.remove('highlight');
                });
            }
            
            function searchInTree(element, term) {
                let found = false;
                
                // Check text content
                if (element.textContent.toLowerCase().includes(term)) {
                    found = true;
                    
                    // Highlight the key or value that contains the search term
                    const keys = element.querySelectorAll('.json-key');
                    const values = element.querySelectorAll('.json-string, .json-number, .json-boolean, .json-null');
                    
                    [...keys, ...values].forEach(el => {
                        if (el.textContent.toLowerCase().includes(term)) {
                            el.classList.add('highlight');
                            
                            // Expand parents
                            let parent = el.parentElement;
                            while (parent) {
                                if (parent.previousElementSibling && 
                                    parent.previousElementSibling.classList.contains('collapsible')) {
                                    parent.previousElementSibling.classList.remove('collapsed');
                                }
                                parent = parent.parentElement;
                            }
                        }
                    });
                }
                
                // Recursively search in children
                Array.from(element.children).forEach(child => {
                    if (searchInTree(child, term)) {
                        found = true;
                    }
                });
                
                return found;
            }
            
            // Expand/Collapse All buttons
            document.getElementById('expand-all').addEventListener('click', function() {
                const collapsibles = document.querySelectorAll('.collapsible');
                collapsibles.forEach(el => {
                    el.classList.remove('collapsed');
                });
            });
            
            document.getElementById('collapse-all').addEventListener('click', function() {
                const collapsibles = document.querySelectorAll('.collapsible');
                collapsibles.forEach(el => {
                    el.classList.add('collapsed');
                });
            });
            
            // Initially collapse all nodes except the first level
            window.addEventListener('load', function() {
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

def display_visualization(args):
    """Display a visualization in a web browser"""
    import webbrowser
    from tempfile import NamedTemporaryFile
    import json
    
    db = Database()
    viz_id = args.id
    visualization = db.get_visualization(viz_id)
    
    if not visualization:
        print(f"Visualization with ID {viz_id} not found.")
        return
    
    viz_id, name, description, test_names, chart_type, chart_data = visualization
    tests = json.loads(test_names)
    data = json.loads(chart_data)
    
    # Create HTML content with Chart.js
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{name} - MCPX Evaluation Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-radial-gauge@1.0.3/build/Chart.RadialGauge.umd.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }}
            h1, h2 {{ color: #333; }}
            .chart-container {{ width: 100%; margin-bottom: 30px; }}
            .description {{ margin-bottom: 20px; color: #555; }}
            .tests-included {{ font-style: italic; color: #777; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>{name}</h1>
        <div class="description">{description}</div>
        <div class="tests-included">Tests included: {', '.join(tests)}</div>
    """
    
    # Add appropriate chart based on chart type
    if chart_type == "bar":
        # Create a section for each metric
        for metric, model_data in data.items():
            chart_id = f"chart_{metric}"
            html += f"""
            <div class="chart-container">
                <h2>{metric.replace('_', ' ').title()}</h2>
                <canvas id="{chart_id}"></canvas>
            </div>
            """
        
        # Define function to generate random colors in Python
        def getRandomColor():
            import random
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f"rgba({r}, {g}, {b}, 0.8)"
            
        # Add JavaScript for bar charts
        html += """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
        """
        
        for metric, model_data in data.items():
            chart_id = f"chart_{metric}"
            
            # Prepare chart data
            labels = []
            datasets = []
            
            # For each model create a dataset
            for model, test_scores in model_data.items():
                if not labels:
                    # Get test names from the first model (all should have the same tests)
                    labels = list(test_scores.keys())
                
                datasets.append({
                    "label": model,
                    "data": [test_scores[test] for test in labels],
                    "backgroundColor": getRandomColor(),
                })
            
            html += f"""
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: {json.dumps(datasets, default=str)}
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        title: {{
                            display: true,
                            text: '{metric.replace("_", " ").title()}'
                        }}
                    }}
                }}
            }});
            """
        
        html += """
        });
        </script>
        """
    
    elif chart_type == "radar":
        # Create a section for each test
        for test_name, model_data in data.items():
            chart_id = f"chart_{test_name.replace(' ', '_')}"
            html += f"""
            <div class="chart-container">
                <h2>Test: {test_name}</h2>
                <canvas id="{chart_id}"></canvas>
            </div>
            """
        
        # Add JavaScript for radar charts
        html += """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
        """
        
        for test_name, model_data in data.items():
            chart_id = f"chart_{test_name.replace(' ', '_')}"
            
            # Get all metrics from the first model
            first_model = list(model_data.keys())[0]
            metrics = list(model_data[first_model].keys())
            
            datasets = []
            for model, metric_scores in model_data.items():
                datasets.append({
                    "label": model,
                    "data": [metric_scores[metric] for metric in metrics],
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "borderColor": getRandomColor(),
                    "pointBackgroundColor": getRandomColor(),
                    "borderWidth": 2
                })
            
            html += f"""
            new Chart(document.getElementById('{chart_id}'), {{
                type: 'radar',
                data: {{
                    labels: {json.dumps([m.replace('_', ' ').title() for m in metrics])},
                    datasets: {json.dumps(datasets, default=str)}
                }},
                options: {{
                    elements: {{
                        line: {{
                            borderWidth: 3
                        }}
                    }},
                    scales: {{
                        r: {{
                            angleLines: {{
                                display: true
                            }},
                            suggestedMin: 0,
                            suggestedMax: 100
                        }}
                    }}
                }}
            }});
            """
        
        html += """
        });
        </script>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write to temporary file and open in browser
    with NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
        f.write(html)
        temp_path = f.name
    
    print(f"Opening visualization '{name}' in web browser...")
    webbrowser.open(f"file://{temp_path}")
    
    # Also save a copy to the specified location if provided
    if args.output:
        with open(args.output, 'w') as f:
            f.write(html)
        print(f"Visualization saved to {args.output}")


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

    # Visualization commands
    viz_parser = subparsers.add_parser("viz", help="Visualization subcommands")
    viz_subparsers = viz_parser.add_subparsers(
        dest="viz_command", help="Visualization command"
    )

    # Create visualization
    create_viz_parser = viz_subparsers.add_parser(
        "create", help="Create a new visualization"
    )
    create_viz_parser.add_argument("name", help="Name for the visualization")
    create_viz_parser.add_argument(
        "--description", "-d", default="", help="Description of the visualization"
    )
    create_viz_parser.add_argument(
        "--tests",
        "-t",
        required=True,
        help="Comma-separated list of test names to include",
    )
    create_viz_parser.add_argument(
        "--type", choices=["bar", "radar"], default="bar", help="Type of visualization"
    )

    # List visualizations
    list_viz_parser = viz_subparsers.add_parser(
        "list", help="List saved visualizations"
    )
    
    # Display visualization
    display_viz_parser = viz_subparsers.add_parser(
        "display", help="Display a visualization in a web browser"
    )
    display_viz_parser.add_argument(
        "id", type=int, help="ID of the visualization to display"
    )
    display_viz_parser.add_argument(
        "--output", "-o", help="Save visualization to specified HTML file path"
    )

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

    # Visualization commands
    if command == "viz":
        viz_command = getattr(args, "viz_command", None)
        if viz_command == "create":
            create_visualization(args)
            return
        elif viz_command == "list":
            list_visualizations(args)
            return
        elif viz_command == "display":
            display_visualization(args)
            return
        else:
            parser.print_help()
            return

    # Summary command
    elif command == "summary":
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
                name=args.name,
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


if __name__ == "__main__":
    main()
