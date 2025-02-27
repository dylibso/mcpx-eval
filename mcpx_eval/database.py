import sqlite3
import json
import pandas as pd
from datetime import datetime
from .models import Score, Results, Test

class Database:
    conn: sqlite3.Connection

    def __init__(self, path: str = "eval.db"):
        self.conn = sqlite3.connect(path)

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                max_tool_calls INTEGER,
                prompt TEXT NOT NULL,
                prompt_check TEXT NOT NULL,
                UNIQUE(name)
            );
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY,
                t TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                test_name TEXT NOT NULL,
                model TEXT NOT NULL,
                duration REAL NOT NULL,
                output TEXT NOT NULL,
                description TEXT NOT NULL,
                accuracy REAL NOT NULL,
                tool_use REAL NOT NULL,
                tool_calls INT NOT NULL,
                redundant_tool_calls INT NOT NULL DEFAULT 0,
                clarity REAL NOT NULL DEFAULT 0.0,
                helpfulness REAL NOT NULL DEFAULT 0.0, 
                overall REAL NOT NULL,
                hallucination_score REAL NOT NULL DEFAULT 0.0,
                false_claims TEXT NOT NULL DEFAULT '[]',
                tool_analysis TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(test_name) REFERENCES tests(name)
            );
        """
        )
        self.conn.commit()

    def save_score(self, name: str, score: Score, commit=True):
        if name == "":
            return

        self.conn.execute(
            """
                INSERT INTO eval_results (
                    test_name,
                    model,
                    duration,
                    output,
                    description,
                    accuracy,
                    tool_use,
                    tool_calls,
                    redundant_tool_calls,
                    clarity,
                    helpfulness,
                    overall,
                    hallucination_score,
                    false_claims,
                    tool_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                name,
                score.model,
                score.duration,
                score.llm_output,
                score.description,
                score.accuracy,
                score.tool_use,
                score.tool_calls,
                score.redundant_tool_calls,
                score.clarity,
                score.helpfulness,
                score.overall,
                score.hallucination_score,
                json.dumps(score.false_claims),
                json.dumps(score.tool_analysis),
            ),
        )
        if commit:
            self.conn.commit()

    def save_test(self, test: "Test"):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tests (name, max_tool_calls, prompt, prompt_check) VALUES (?, ?, ?, ?);
            """,
            (test.name, test.max_tool_calls, test.prompt, test.check),
        )
        self.conn.commit()

    def save_results(self, name: str, results: Results):
        for score in results.scores:
            self.save_score(name, score, commit=False)
        self.conn.commit()

    def average_results(self, name: str) -> Results:
        # Read results into a pandas DataFrame
        df = pd.read_sql_query(
            """
            SELECT *
            FROM eval_results
            WHERE test_name = ?
            """,
            self.conn,
            params=(name,)
        )
        
        if df.empty:
            return Results(scores=[])
            
        # Convert false_claims and tool_analysis from JSON strings
        df['false_claims'] = df['false_claims'].apply(json.loads)
        df['tool_analysis'] = df['tool_analysis'].apply(json.loads)
        
        # Group by model and aggregate
        grouped = df.groupby('model').agg({
            'duration': 'mean',
            'output': 'first',  # take first output as example
            'description': 'first',  # take first description as example
            'accuracy': 'mean',
            'tool_use': 'mean',
            'tool_calls': 'mean',
            'redundant_tool_calls': 'mean',
            'clarity': 'mean',
            'helpfulness': 'mean',
            'overall': 'mean',
            'hallucination_score': 'mean',
            'false_claims': 'sum',  # combine all false claims
            'tool_analysis': 'first'  # take first tool analysis
        }).reset_index()
        
        # Convert back to Score objects
        scores = [
            Score(
                model=row['model'],
                duration=row['duration'],
                llm_output=row['output'],
                description=row['description'],
                accuracy=row['accuracy'],
                tool_use=row['tool_use'],
                tool_calls=int(row['tool_calls']),
                redundant_tool_calls=int(row['redundant_tool_calls']),
                clarity=row['clarity'],
                helpfulness=row['helpfulness'],
                overall=row['overall'],
                hallucination_score=row['hallucination_score'],
                false_claims=row['false_claims'],
                tool_analysis=row['tool_analysis']
            )
            for _, row in grouped.iterrows()
        ]
        
        return Results(scores=scores)

    def generate_json_summary(self):
        # Read results into a pandas DataFrame
        df = pd.read_sql_query(
            """
            SELECT 
                test_name,
                model,
                AVG(accuracy) as accuracy,
                AVG(tool_use) as tool_use,
                AVG(tool_calls) as tool_calls,
                AVG(redundant_tool_calls) as redundant_tool_calls,
                AVG(clarity) as clarity,
                AVG(helpfulness) as helpfulness,
                AVG(overall) as overall,
                AVG(hallucination_score) as hallucination_score,
                COUNT(*) as runs
            FROM eval_results
            GROUP BY test_name, model
            """,
            self.conn
        )
        
        # Convert DataFrame to nested dictionary structure
        summary = {}
        for test_name in df['test_name'].unique():
            test_df = df[df['test_name'] == test_name]
            summary[test_name] = {
                row['model']: {
                    'accuracy': row['accuracy'],
                    'tool_use': row['tool_use'],
                    'tool_calls': row['tool_calls'],
                    'redundant_tool_calls': row['redundant_tool_calls'],
                    'clarity': row['clarity'],
                    'helpfulness': row['helpfulness'],
                    'overall': row['overall'],
                    'hallucination_score': row['hallucination_score'],
                    'runs': row['runs']
                }
                for _, row in test_df.iterrows()
            }
        
        return summary
