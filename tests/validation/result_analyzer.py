"""
Result Analyzer for Chatbot Validation Tests

Automatically analyzes test results using LLM to:
- Identify recurring patterns in failures
- Categorize issues by type
- Prioritize which issues need attention
- Generate concise summary report

Author: Claude Code
Version: 1.0.0
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from llm import LLMHandler


class ResultAnalyzer:
    """
    Analyzes chatbot test results to identify patterns and priorities.

    Uses LLM to perform semantic analysis of test failures and low scores,
    identifying recurring issues and recommending fixes.
    """

    def __init__(self, config_path: str = 'config/config.yaml', prompt_path: str = 'prompts/result_analysis_prompt.txt'):
        """
        Initialize the ResultAnalyzer.

        Args:
            config_path (str): Path to config file
            prompt_path (str): Path to analysis prompt file
        """
        self.llm_handler = LLMHandler(config_path)
        self.analysis_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str) -> str:
        """
        Load the analysis prompt from file.

        Args:
            prompt_path (str): Path to prompt file

        Returns:
            str: Analysis prompt template

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Analysis prompt file not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            prompt = f.read()

        logging.info(f"Loaded analysis prompt from {prompt_path}")
        return prompt

    def load_test_results(self, results_file: str) -> pd.DataFrame:
        """
        Load test results from CSV file.

        Args:
            results_file (str): Path to chatbot_test_results.csv

        Returns:
            pd.DataFrame: Test results
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")

        df = pd.read_csv(results_file)
        logging.info(f"Loaded {len(df)} test results from {results_file}")
        return df

    def prepare_results_summary(self, df: pd.DataFrame) -> str:
        """
        Prepare a concise summary of test results for LLM analysis.

        Args:
            df (pd.DataFrame): Test results DataFrame

        Returns:
            str: Formatted summary for LLM
        """
        summary_parts = []

        # Overall statistics
        summary_parts.append(f"OVERALL STATISTICS:")
        summary_parts.append(f"Total tests: {len(df)}")
        summary_parts.append(f"Average score: {df['evaluation_score'].mean():.1f}")
        summary_parts.append(f"Execution success rate: {df['execution_success'].mean():.1%}")
        summary_parts.append("")

        # Score distribution
        summary_parts.append(f"SCORE DISTRIBUTION:")
        summary_parts.append(f"Excellent (90-100): {len(df[df['evaluation_score'] >= 90])}")
        summary_parts.append(f"Good (70-89): {len(df[(df['evaluation_score'] >= 70) & (df['evaluation_score'] < 90)])}")
        summary_parts.append(f"Fair (50-69): {len(df[(df['evaluation_score'] >= 50) & (df['evaluation_score'] < 70)])}")
        summary_parts.append(f"Poor (<50): {len(df[df['evaluation_score'] < 50])}")
        summary_parts.append("")

        # Category breakdown
        summary_parts.append(f"CATEGORY BREAKDOWN:")
        category_stats = df.groupby('category').agg({
            'evaluation_score': 'mean',
            'execution_success': 'mean'
        }).round(1)
        for category, row in category_stats.iterrows():
            count = len(df[df['category'] == category])
            summary_parts.append(f"  {category}: {count} tests, avg score {row['evaluation_score']:.1f}, success rate {row['execution_success']:.1%}")
        summary_parts.append("")

        # Problematic tests (score < 70 or execution failed)
        problematic = df[(df['evaluation_score'] < 70) | (~df['execution_success'])]

        if len(problematic) > 0:
            summary_parts.append(f"PROBLEMATIC TESTS ({len(problematic)} tests):")
            summary_parts.append("")

            # Group by category for easier pattern identification
            for category in problematic['category'].unique():
                category_problems = problematic[problematic['category'] == category]
                summary_parts.append(f"Category: {category} ({len(category_problems)} issues)")

                # Show up to 5 examples per category
                for idx, row in category_problems.head(5).iterrows():
                    summary_parts.append(f"  - Question: {row['question']}")
                    summary_parts.append(f"    Score: {row['evaluation_score']}")
                    summary_parts.append(f"    Execution success: {row['execution_success']}")
                    summary_parts.append(f"    Reasoning: {row['evaluation_reasoning']}")

                    # Include SQL snippet if available
                    if pd.notna(row['sql_query']):
                        sql_snippet = str(row['sql_query'])[:200]
                        summary_parts.append(f"    SQL snippet: {sql_snippet}...")
                    summary_parts.append("")

                if len(category_problems) > 5:
                    summary_parts.append(f"  ... and {len(category_problems) - 5} more issues in this category")
                    summary_parts.append("")

        return "\n".join(summary_parts)

    def analyze_results(self, df: pd.DataFrame) -> dict:
        """
        Analyze test results using LLM to identify patterns and priorities.

        Args:
            df (pd.DataFrame): Test results DataFrame

        Returns:
            dict: Analysis results with patterns, priorities, and recommendations
        """
        logging.info("Preparing test results summary for LLM analysis...")
        results_summary = self.prepare_results_summary(df)

        # Format prompt with results
        prompt = self.analysis_prompt.format(test_results_summary=results_summary)

        logging.info("Sending analysis request to LLM...")
        logging.info(f"Prompt length: {len(prompt)} characters")

        try:
            # Get LLM analysis
            # Note: query_llm takes (url, prompt, schema_type) - url is for logging purposes
            response = self.llm_handler.query_llm("test_results_analysis", prompt)

            # Strip markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            elif response.startswith("```"):
                response = response[3:]  # Remove ```

            if response.endswith("```"):
                response = response[:-3]  # Remove trailing ```

            response = response.strip()

            # Parse JSON response
            analysis = json.loads(response)
            logging.info("Successfully parsed LLM analysis")

            return analysis

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Raw response: {response[:500]}...")

            # Fallback: Create basic analysis from data
            return self._fallback_analysis(df)

        except Exception as e:
            logging.error(f"Error during LLM analysis: {e}")
            return self._fallback_analysis(df)

    def _fallback_analysis(self, df: pd.DataFrame) -> dict:
        """
        Create basic rule-based analysis if LLM fails.

        Args:
            df (pd.DataFrame): Test results DataFrame

        Returns:
            dict: Basic analysis structure
        """
        logging.warning("Using fallback rule-based analysis")

        problematic = df[df['evaluation_score'] < 70]

        return {
            "summary": {
                "total_issues_identified": len(problematic),
                "critical_issues": len(df[df['evaluation_score'] < 50]),
                "high_priority_issues": len(df[(df['evaluation_score'] >= 50) & (df['evaluation_score'] < 70)]),
                "medium_priority_issues": 0,
                "low_priority_issues": 0
            },
            "recurring_patterns": [
                {
                    "pattern_name": "Low scoring tests",
                    "description": f"{len(problematic)} tests scored below 70",
                    "affected_tests": len(problematic),
                    "affected_categories": problematic['category'].unique().tolist(),
                    "root_cause": "Unable to determine - LLM analysis failed",
                    "severity": "Unknown",
                    "example_questions": problematic['question'].head(3).tolist(),
                    "recommended_fix": "Manual review required"
                }
            ],
            "priority_recommendations": [
                {
                    "priority": 1,
                    "issue": "LLM analysis unavailable",
                    "reason": "Automated pattern detection failed",
                    "action": "Review test results manually or retry analysis"
                }
            ],
            "acceptable_issues": []
        }

    def generate_report(self, analysis: dict, output_file: str):
        """
        Generate human-readable analysis report.

        Args:
            analysis (dict): Analysis results from analyze_results()
            output_file (str): Path to save report
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("CHATBOT TEST RESULTS ANALYSIS")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        summary = analysis.get('summary', {})
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total issues identified: {summary.get('total_issues_identified', 0)}")
        report_lines.append(f"  Critical:  {summary.get('critical_issues', 0)}")
        report_lines.append(f"  High:      {summary.get('high_priority_issues', 0)}")
        report_lines.append(f"  Medium:    {summary.get('medium_priority_issues', 0)}")
        report_lines.append(f"  Low:       {summary.get('low_priority_issues', 0)}")
        report_lines.append("")

        # Priority recommendations
        recommendations = analysis.get('priority_recommendations', [])
        if recommendations:
            report_lines.append("PRIORITY RECOMMENDATIONS")
            report_lines.append("-" * 80)
            for rec in recommendations:
                report_lines.append(f"{rec['priority']}. {rec['issue']}")
                report_lines.append(f"   Reason: {rec['reason']}")
                report_lines.append(f"   Action: {rec['action']}")
                report_lines.append("")

        # Recurring patterns
        patterns = analysis.get('recurring_patterns', [])
        if patterns:
            report_lines.append("RECURRING ISSUE PATTERNS")
            report_lines.append("-" * 80)

            # Sort by severity and affected test count
            severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Unknown': 4}
            patterns_sorted = sorted(
                patterns,
                key=lambda x: (severity_order.get(x.get('severity', 'Unknown'), 99), -x.get('affected_tests', 0))
            )

            for i, pattern in enumerate(patterns_sorted, 1):
                report_lines.append(f"{i}. {pattern.get('pattern_name', 'Unknown')} [{pattern.get('severity', 'Unknown')}]")
                report_lines.append(f"   Description: {pattern.get('description', 'N/A')}")
                report_lines.append(f"   Affected tests: {pattern.get('affected_tests', 0)}")
                report_lines.append(f"   Categories: {', '.join(pattern.get('affected_categories', []))}")
                report_lines.append(f"   Root cause: {pattern.get('root_cause', 'N/A')}")

                examples = pattern.get('example_questions', [])
                if examples:
                    report_lines.append(f"   Examples:")
                    for example in examples[:3]:
                        report_lines.append(f"     - {example}")

                report_lines.append(f"   Recommended fix: {pattern.get('recommended_fix', 'N/A')}")
                report_lines.append("")

        # Acceptable issues
        acceptable = analysis.get('acceptable_issues', [])
        if acceptable:
            report_lines.append("ACCEPTABLE ISSUES (Low Priority)")
            report_lines.append("-" * 80)
            for issue in acceptable:
                report_lines.append(f"- {issue.get('pattern', 'N/A')}")
                report_lines.append(f"  Reason: {issue.get('reason', 'N/A')}")
                report_lines.append("")

        # Write report
        report_text = "\n".join(report_lines)

        with open(output_file, 'w') as f:
            f.write(report_text)

        logging.info(f"Analysis report saved to: {output_file}")

        # Also print to console
        print("\n" + report_text)

        # Save JSON version
        json_output = output_file.replace('.txt', '.json')
        with open(json_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logging.info(f"JSON analysis saved to: {json_output}")


def main():
    """Main entry point for result analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Resolve output directory from config to align with validation reports
    try:
        with open('config/config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        output_dir = (
            cfg.get('testing', {})
               .get('validation', {})
               .get('reporting', {})
               .get('output_dir', 'tests/output')
        )
    except Exception:
        # Fallback to previous default if config missing/unreadable
        output_dir = 'tests/output'

    # Paths
    results_file = os.path.join(output_dir, 'chatbot_test_results.csv')
    output_file = os.path.join(output_dir, 'analysis_report.txt')

    try:
        # Initialize analyzer
        logging.info("Initializing ResultAnalyzer...")
        analyzer = ResultAnalyzer()

        # Load results
        logging.info(f"Loading test results from {results_file}...")
        df = analyzer.load_test_results(results_file)

        # Analyze
        logging.info("Analyzing test results with LLM...")
        analysis = analyzer.analyze_results(df)

        # Generate report
        logging.info("Generating analysis report...")
        analyzer.generate_report(analysis, output_file)

        logging.info("Analysis complete!")

        # Return success
        sys.exit(0)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Make sure validation tests have completed and results file exists")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
