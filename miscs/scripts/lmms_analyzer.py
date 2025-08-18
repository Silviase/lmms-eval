#!/usr/bin/env python3
"""
Unified LMMS-Eval Analysis Tool
Combines all analysis functionalities into one comprehensive tool.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import html
import sys
import subprocess
import base64


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_BASE_DIR = Path("/home/silviase/LIMIT-Lab/typographic_atk/artifacts")
LMMS_OUTPUT_DIR = Path("/home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/outputs")
IMAGE_BASE_DIR = Path("/home/silviase/LIMIT-Lab/typographic_atk/lmms-eval/data")

TASK_METRICS = {
    'chartqa': {
        'metrics': ['relaxed_overall', 'relaxed_human_split', 'relaxed_augmented_split'],
        'threshold': 0.5,
        'description': 'ChartQA uses relaxed correctness (5% tolerance for numbers)'
    },
    'docvqa_val': {
        'metrics': ['anls'],
        'threshold': 0.5,
        'description': 'DocVQA uses ANLS (Average Normalized Levenshtein Similarity)'
    },
    'infovqa_val': {
        'metrics': ['anls'],
        'threshold': 0.5,
        'description': 'InfoVQA uses ANLS metric'
    },
    'textvqa': {
        'metrics': ['accuracy'],
        'threshold': 0.5,
        'description': 'TextVQA uses exact match accuracy'
    },
    'textvqa_val': {
        'metrics': ['accuracy'],
        'threshold': 0.5,
        'description': 'TextVQA validation uses exact match accuracy'
    }
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EvaluationResult:
    """Represents a single evaluation result."""
    doc_id: int
    target: Any
    prediction: str
    score: float
    task_name: str
    model_name: str
    input_text: str
    is_correct: bool
    metric_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ErrorAnalysis:
    """Represents an error analysis for a specific question."""
    doc_id: int
    task_name: str
    input_text: str
    target: str
    models_failed: List[str]
    model_predictions: Dict[str, str]
    model_scores: Dict[str, float]
    failure_rate: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# Core Analyzer Class
# ============================================================================

class LMMSAnalyzer:
    """Unified analyzer for LMMS-Eval results."""
    
    def __init__(self):
        self.results = []
        self.results_by_model = defaultdict(list)
        self.results_by_question = defaultdict(dict)
        self.ensure_output_dirs()
    
    def ensure_output_dirs(self):
        """Ensure output directories exist."""
        for subdir in ['reports', 'csv', 'labeled', 'images']:
            (OUTPUT_BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    
    def load_results(self, model_filter: Optional[str] = None, 
                    task_filter: Optional[str] = None, clear_existing: bool = True) -> None:
        """Load evaluation results from output directory."""
        if clear_existing:
            self.results = []
            self.results_by_model.clear()
            self.results_by_question.clear()
        
        for model_dir in LMMS_OUTPUT_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            if model_filter and model_filter not in model_name:
                continue
            
            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                task_name = task_dir.name
                
                if task_filter and task_filter not in task_name:
                    continue
                
                # Find JSONL files
                for jsonl_file in task_dir.rglob("*_samples_*.jsonl"):
                    print(f"Loading {jsonl_file}")
                    self._parse_jsonl_file(jsonl_file, task_name, model_name)
        
        print(f"Loaded {len(self.results)} total results")
    
    def _parse_jsonl_file(self, file_path: Path, task_name: str, model_name: str) -> None:
        """Parse a JSONL sample file."""
        task_info = TASK_METRICS.get(task_name, {})
        metric_names = task_info.get('metrics', [])
        threshold = task_info.get('threshold', 0.5)
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # Extract score
                    score = None
                    metric_name = None
                    for metric in metric_names:
                        if metric in data:
                            score = data[metric]
                            metric_name = metric
                            break
                    
                    if score is None:
                        for field in ['accuracy', 'exact_match', 'score']:
                            if field in data:
                                score = float(data[field])
                                metric_name = field
                                break
                        if score is None:
                            score = 0
                            metric_name = 'unknown'
                    
                    # Create result object
                    doc_id = data.get('doc_id', line_num)
                    question_id = f"{task_name}_{doc_id}"
                    
                    result = EvaluationResult(
                        doc_id=doc_id,
                        target=data.get('target', ''),
                        prediction=data.get('filtered_resps', [''])[0] if 'filtered_resps' in data else '',
                        score=score,
                        task_name=task_name,
                        model_name=model_name,
                        input_text=data.get('input', ''),
                        is_correct=score >= threshold,
                        metric_name=metric_name
                    )
                    
                    self.results.append(result)
                    self.results_by_model[model_name].append(result)
                    self.results_by_question[question_id][model_name] = result
                    
                except Exception as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
    
    # ========================================================================
    # Analysis Functions
    # ========================================================================
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics."""
        stats = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            key = (result.model_name, result.task_name, result.metric_name)
            stats[key]['scores'].append(result.score)
            stats[key]['correct'].append(result.is_correct)
        
        summary_data = []
        for (model, task, metric), data in stats.items():
            summary_data.append({
                'model': model,
                'task': task,
                'metric': metric,
                'num_samples': len(data['scores']),
                'accuracy': sum(data['correct']) / len(data['correct']) if data['correct'] else 0,
                'avg_score': sum(data['scores']) / len(data['scores']) if data['scores'] else 0,
                'num_correct': sum(data['correct']),
                'num_incorrect': len(data['correct']) - sum(data['correct'])
            })
        
        return pd.DataFrame(summary_data)
    
    def find_common_errors(self, min_models_failed: int = 2) -> List[ErrorAnalysis]:
        """Find questions where multiple models failed."""
        common_errors = []
        
        for question_id, model_results in self.results_by_question.items():
            if not model_results:
                continue
            
            failed_models = []
            model_predictions = {}
            model_scores = {}
            
            for model_name, result in model_results.items():
                if not result.is_correct:
                    failed_models.append(model_name)
                model_predictions[model_name] = result.prediction
                model_scores[model_name] = result.score
            
            if len(failed_models) >= min_models_failed:
                first_result = next(iter(model_results.values()))
                
                error_analysis = ErrorAnalysis(
                    doc_id=first_result.doc_id,
                    task_name=first_result.task_name,
                    input_text=first_result.input_text,
                    target=str(first_result.target),
                    models_failed=failed_models,
                    model_predictions=model_predictions,
                    model_scores=model_scores,
                    failure_rate=len(failed_models) / len(model_results)
                )
                common_errors.append(error_analysis)
        
        common_errors.sort(key=lambda x: x.failure_rate, reverse=True)
        return common_errors
    
    # ========================================================================
    # Export Functions
    # ========================================================================
    
    def export_csv(self, filename: str = None, data: Any = None):
        """Export data to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.csv"
        
        output_path = OUTPUT_BASE_DIR / 'csv' / filename
        
        if data is None:
            data = [r.to_dict() for r in self.results]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Exported to {output_path}")
        return output_path
    
    def generate_html_report(self, 
                           errors: List[ErrorAnalysis],
                           output_filename: str = None,
                           task_name: str = None) -> Path:
        """Generate HTML report with images."""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_suffix = f"_{task_name}" if task_name else ""
            output_filename = f"error_report{task_suffix}_{timestamp}.html"
        
        output_path = OUTPUT_BASE_DIR / 'reports' / output_filename
        
        # Generate HTML content
        html_content = self._generate_html_template(errors, task_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Generated HTML report: {output_path}")
        return output_path
    
    def _generate_html_template(self, errors: List[ErrorAnalysis], task_name: str = None) -> str:
        """Generate HTML template."""
        
        # Statistics
        model_names = list(self.results_by_model.keys())
        total_questions = len(self.results_by_question)
        
        # Generate error cards
        error_cards_html = []
        for idx, error in enumerate(errors[:100], 1):  # Limit to 100 errors
            
            # Check for image
            image_path = IMAGE_BASE_DIR / error.task_name / 'images' / f"{error.doc_id}.png"
            image_html = ""
            
            if image_path.exists():
                # Read image and convert to base64
                try:
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_html = f'''
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_base64}" alt="Question image">
                    </div>
                    '''
                except Exception as e:
                    image_html = f'''
                    <div class="image-container">
                        <div class="no-image">
                            ⚠️ Error loading image<br>
                            <small>{str(e)}</small>
                        </div>
                    </div>
                    '''
            else:
                image_html = f'''
                <div class="image-container">
                    <div class="no-image">
                        📷 No image available<br>
                        <small>Run: python extract_dataset_images.py extract --task {error.task_name}</small>
                    </div>
                </div>
                '''
            
            # Model predictions table
            predictions_html = []
            for model_name, pred in error.model_predictions.items():
                status = "❌" if model_name in error.models_failed else "✅"
                score = error.model_scores[model_name]
                predictions_html.append(f'''
                    <tr>
                        <td>{status}</td>
                        <td>{html.escape(model_name)}</td>
                        <td>{html.escape(str(pred))}</td>
                        <td>{score:.3f}</td>
                    </tr>
                ''')
            
            error_card = f'''
            <div class="error-card">
                <div class="error-header">
                    <h3>#{idx}. Doc ID: {error.doc_id}</h3>
                    <span class="failure-badge">{len(error.models_failed)}/{len(error.model_predictions)} failed</span>
                </div>
                <div class="error-content">
                    <div class="question">
                        <strong>Question:</strong> {html.escape(error.input_text)}
                    </div>
                    <div class="answer">
                        <strong>Answer:</strong> {html.escape(str(error.target))}
                    </div>
                    {image_html}
                    <table class="predictions">
                        <thead>
                            <tr><th>Status</th><th>Model</th><th>Prediction</th><th>Score</th></tr>
                        </thead>
                        <tbody>
                            {''.join(predictions_html)}
                        </tbody>
                    </table>
                </div>
            </div>
            '''
            error_cards_html.append(error_card)
        
        # Build full HTML
        task_title = f" - {task_name}" if task_name else ""
        html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LMMS-Eval Analysis Report{task_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .error-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 20px 0;
            overflow: hidden;
        }}
        .error-header {{
            background: #f8f9fa;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .failure-badge {{
            background: #f44336;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .error-content {{
            padding: 20px;
        }}
        .question, .answer {{
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .answer {{
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .no-image {{
            padding: 40px;
            background: #fff3cd;
            border-radius: 8px;
            color: #856404;
        }}
        table.predictions {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        table.predictions th, table.predictions td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        table.predictions th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .search-box {{
            margin: 20px 0;
        }}
        .search-box input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 LMMS-Eval Error Analysis Report{task_title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(errors)}</div>
                <div>Common Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(model_names)}</div>
                <div>Models Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_questions}</div>
                <div>Total Questions</div>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="🔍 Search questions, answers, or predictions..." onkeyup="filterCards()">
        </div>
        
        <div id="errorCards">
            {''.join(error_cards_html)}
        </div>
    </div>
    
    <script>
        function filterCards() {{
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const cards = document.getElementsByClassName('error-card');
            
            for (let card of cards) {{
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(filter) ? '' : 'none';
            }}
        }}
    </script>
</body>
</html>
        '''
        
        return html_template


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified LMMS-Eval Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze evaluation results')
    analyze_parser.add_argument('--task', type=str, help='Task name filter')
    analyze_parser.add_argument('--model', type=str, help='Model name filter')
    analyze_parser.add_argument('--export-csv', type=str, help='Export to CSV file')
    analyze_parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    # Find common errors command
    errors_parser = subparsers.add_parser('errors', help='Find common errors across models')
    errors_parser.add_argument('--task', type=str, required=True, help='Task to analyze')
    errors_parser.add_argument('--models', nargs='+', help='Model names to compare')
    errors_parser.add_argument('--min-failed', type=int, default=2, 
                              help='Minimum models that must fail')
    errors_parser.add_argument('--export-csv', type=str, help='Export to CSV')
    errors_parser.add_argument('--export-html', type=str, help='Export to HTML')
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive HTML report')
    report_parser.add_argument('--task', type=str, required=True, help='Task to analyze')
    report_parser.add_argument('--models', nargs='+', 
                              default=['Qwen2.5-VL-3B', 'Qwen2.5-VL-7B'],
                              help='Models to compare')
    report_parser.add_argument('--min-failed', type=int, default=2,
                              help='Minimum models that must fail')
    report_parser.add_argument('--output', type=str, help='Output filename')
    
    # Extract images command
    extract_parser = subparsers.add_parser('extract-images', help='Extract dataset images')
    extract_parser.add_argument('--task', type=str, help='Task name')
    extract_parser.add_argument('--all', action='store_true', help='Extract all tasks')
    extract_parser.add_argument('--limit', type=int, help='Limit number of images')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    analyzer = LMMSAnalyzer()
    
    if args.command == 'analyze':
        analyzer.load_results(model_filter=args.model, task_filter=args.task)
        
        if args.summary:
            summary = analyzer.get_summary_stats()
            print("\n=== Summary Statistics ===")
            print(summary.to_string())
        
        if args.export_csv:
            analyzer.export_csv(args.export_csv)
    
    elif args.command == 'errors':
        # Filter models if specified
        model_filter = None
        if args.models:
            # Load for each model
            for model in args.models:
                analyzer.load_results(model_filter=model, task_filter=args.task)
        else:
            analyzer.load_results(task_filter=args.task)
        
        errors = analyzer.find_common_errors(min_models_failed=args.min_failed)
        
        print(f"\nFound {len(errors)} common errors")
        
        if args.export_csv:
            data = [e.to_dict() for e in errors]
            analyzer.export_csv(args.export_csv, data)
        
        if args.export_html:
            analyzer.generate_html_report(errors, args.export_html, args.task)
    
    elif args.command == 'report':
        # Load results for specified models
        for i, model in enumerate(args.models):
            analyzer.load_results(model_filter=model, task_filter=args.task, clear_existing=(i==0))
        
        # Find common errors
        errors = analyzer.find_common_errors(min_models_failed=args.min_failed)
        
        # Generate HTML report
        output_file = args.output or f"{args.task}_report.html"
        report_path = analyzer.generate_html_report(errors, output_file, args.task)
        
        print(f"\n✅ Report generated: {report_path}")
        print(f"   Found {len(errors)} common errors")
        print(f"   Open in browser: file://{report_path.absolute()}")
    
    elif args.command == 'extract-images':
        # Call the extract script
        script_path = Path(__file__).parent / 'extract_dataset_images.py'
        cmd = [sys.executable, str(script_path), 'extract']
        
        if args.task:
            cmd.extend(['--task', args.task])
        elif args.all:
            cmd.append('--all')
        else:
            print("Please specify --task or --all")
            return
        
        if args.limit:
            cmd.extend(['--limit', str(args.limit)])
        
        subprocess.run(cmd)


if __name__ == "__main__":
    main()