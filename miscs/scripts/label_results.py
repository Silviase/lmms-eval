#!/usr/bin/env python3
"""
Interactive labeling tool for lmms-eval results.
Allows manual review and re-labeling of evaluation results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime


class InteractiveLabeler:
    """Interactive tool for labeling and reviewing evaluation results."""
    
    def __init__(self, jsonl_file: str):
        self.jsonl_file = Path(jsonl_file)
        self.results = []
        self.labels = []
        self.current_index = 0
        self.load_results()
    
    def load_results(self):
        """Load results from JSONL file."""
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self.results.append(data)
                    # Initialize label with existing score or None
                    existing_score = self._get_existing_score(data)
                    self.labels.append({
                        'auto_score': existing_score,
                        'manual_label': None,
                        'notes': ''
                    })
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.results)} results from {self.jsonl_file}")
    
    def _get_existing_score(self, data: Dict) -> Optional[float]:
        """Extract existing evaluation score from result data."""
        # Check for common metric fields
        metric_fields = [
            'relaxed_overall', 'relaxed_human_split', 'relaxed_augmented_split',
            'anls', 'accuracy', 'exact_match', 'score'
        ]
        
        for field in metric_fields:
            if field in data:
                return float(data[field])
        return None
    
    def display_current_item(self):
        """Display the current item for labeling."""
        if self.current_index >= len(self.results):
            print("No more items to label!")
            return False
        
        item = self.results[self.current_index]
        label = self.labels[self.current_index]
        
        print("\n" + "="*80)
        print(f"Item {self.current_index + 1} of {len(self.results)}")
        print("="*80)
        
        # Display question/input
        if 'input' in item:
            print(f"\nInput/Question:")
            print(f"  {item['input']}")
        
        # Display target answer
        if 'target' in item:
            print(f"\nTarget Answer:")
            target = item['target']
            if isinstance(target, list):
                for t in target:
                    print(f"  - {t}")
            else:
                print(f"  {target}")
        
        # Display model prediction
        if 'filtered_resps' in item:
            print(f"\nModel Prediction:")
            pred = item['filtered_resps'][0] if item['filtered_resps'] else ''
            print(f"  {pred}")
        
        # Display automatic score
        print(f"\nAutomatic Score: {label['auto_score']}")
        
        # Display manual label if exists
        if label['manual_label'] is not None:
            print(f"Manual Label: {'CORRECT' if label['manual_label'] else 'INCORRECT'}")
        
        if label['notes']:
            print(f"Notes: {label['notes']}")
        
        return True
    
    def label_current_item(self):
        """Interactively label the current item."""
        print("\n" + "-"*40)
        print("Label this item:")
        print("  [c] Correct")
        print("  [i] Incorrect")
        print("  [s] Skip")
        print("  [n] Add note")
        print("  [p] Previous item")
        print("  [q] Quit and save")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'c':
            self.labels[self.current_index]['manual_label'] = True
            print("✓ Marked as CORRECT")
            self.current_index += 1
        elif choice == 'i':
            self.labels[self.current_index]['manual_label'] = False
            print("✗ Marked as INCORRECT")
            self.current_index += 1
        elif choice == 's':
            print("→ Skipped")
            self.current_index += 1
        elif choice == 'n':
            note = input("Enter note: ").strip()
            self.labels[self.current_index]['notes'] = note
            print("📝 Note added")
        elif choice == 'p':
            if self.current_index > 0:
                self.current_index -= 1
                print("← Going to previous item")
            else:
                print("Already at first item")
        elif choice == 'q':
            return False
        
        return True
    
    def save_labels(self, output_file: Optional[str] = None):
        """Save labeled results to a new file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.jsonl_file.parent / f"labeled_{timestamp}_{self.jsonl_file.name}"
        
        output_path = Path(output_file)
        
        # Combine original results with labels
        labeled_data = []
        for i, (result, label) in enumerate(zip(self.results, self.labels)):
            combined = result.copy()
            combined['manual_label'] = label['manual_label']
            combined['label_notes'] = label['notes']
            combined['auto_vs_manual'] = None
            
            if label['manual_label'] is not None and label['auto_score'] is not None:
                auto_correct = label['auto_score'] >= 0.5
                combined['auto_vs_manual'] = 'agree' if auto_correct == label['manual_label'] else 'disagree'
            
            labeled_data.append(combined)
        
        # Save as JSONL
        with open(output_path, 'w') as f:
            for item in labeled_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"\nSaved labeled results to: {output_path}")
        
        # Generate summary
        self.print_summary()
    
    def print_summary(self):
        """Print labeling summary statistics."""
        total = len(self.labels)
        labeled = sum(1 for l in self.labels if l['manual_label'] is not None)
        correct = sum(1 for l in self.labels if l['manual_label'] is True)
        incorrect = sum(1 for l in self.labels if l['manual_label'] is False)
        
        print("\n=== Labeling Summary ===")
        print(f"Total items: {total}")
        print(f"Labeled: {labeled} ({labeled/total*100:.1f}%)")
        print(f"  - Correct: {correct}")
        print(f"  - Incorrect: {incorrect}")
        print(f"Unlabeled: {total - labeled}")
        
        # Compare with automatic scores
        agreements = 0
        disagreements = 0
        for label in self.labels:
            if label['manual_label'] is not None and label['auto_score'] is not None:
                auto_correct = label['auto_score'] >= 0.5
                if auto_correct == label['manual_label']:
                    agreements += 1
                else:
                    disagreements += 1
        
        if agreements + disagreements > 0:
            print(f"\nAgreement with automatic scoring:")
            print(f"  - Agree: {agreements}")
            print(f"  - Disagree: {disagreements}")
            print(f"  - Agreement rate: {agreements/(agreements+disagreements)*100:.1f}%")
    
    def run(self):
        """Run the interactive labeling session."""
        print("\n🏷️  Interactive Labeling Tool")
        print("Navigate through items and label them as correct or incorrect.")
        
        while self.current_index < len(self.results):
            if not self.display_current_item():
                break
            
            if not self.label_current_item():
                break
        
        # Ask to save
        if input("\nSave labels? (y/n): ").strip().lower() == 'y':
            self.save_labels()


class BatchLabeler:
    """Tool for batch labeling based on rules or thresholds."""
    
    @staticmethod
    def apply_threshold_labeling(jsonl_file: str, threshold: float = 0.5, output_file: Optional[str] = None):
        """Apply threshold-based labeling to all results."""
        input_path = Path(jsonl_file)
        
        if output_file is None:
            output_file = input_path.parent / f"threshold_labeled_{input_path.name}"
        
        results = []
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Find score field
                    score = None
                    for field in ['relaxed_overall', 'anls', 'accuracy', 'exact_match']:
                        if field in data:
                            score = float(data[field])
                            break
                    
                    if score is not None:
                        data['threshold_label'] = score >= threshold
                        data['label_confidence'] = abs(score - threshold)
                    
                    results.append(data)
                except json.JSONDecodeError:
                    continue
        
        # Save results
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        print(f"Applied threshold labeling (threshold={threshold}) to {len(results)} items")
        print(f"Saved to: {output_file}")
        
        # Print summary
        labeled = sum(1 for r in results if 'threshold_label' in r)
        correct = sum(1 for r in results if r.get('threshold_label', False))
        print(f"\nSummary:")
        print(f"  - Total: {len(results)}")
        print(f"  - Labeled: {labeled}")
        print(f"  - Correct: {correct} ({correct/labeled*100:.1f}%)" if labeled > 0 else "")


def main():
    parser = argparse.ArgumentParser(description='Interactive labeling tool for lmms-eval results')
    
    subparsers = parser.add_subparsers(dest='mode', help='Labeling mode')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive labeling')
    interactive_parser.add_argument('jsonl_file', type=str, help='JSONL file to label')
    interactive_parser.add_argument('--output', type=str, help='Output file for labeled results')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Batch labeling with threshold')
    batch_parser.add_argument('jsonl_file', type=str, help='JSONL file to label')
    batch_parser.add_argument('--threshold', type=float, default=0.5, 
                             help='Score threshold for correct/incorrect')
    batch_parser.add_argument('--output', type=str, help='Output file for labeled results')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        labeler = InteractiveLabeler(args.jsonl_file)
        labeler.run()
    elif args.mode == 'batch':
        BatchLabeler.apply_threshold_labeling(
            args.jsonl_file, 
            threshold=args.threshold,
            output_file=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()