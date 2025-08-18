#!/usr/bin/env python3
"""
Extract and save all images from datasets for later visualization.
This script downloads/extracts images from HuggingFace datasets and saves them locally.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
import json


class DatasetImageExtractor:
    """Extract images from various VQA datasets."""
    
    DATASET_CONFIGS = {
        'chartqa': {
            'dataset_path': 'lmms-lab/ChartQA',
            'split': 'test',
            'image_field': 'image',
            'id_field': None,  # Will use index
            'use_auth_token': True
        },
        'docvqa_val': {
            'dataset_path': 'lmms-lab/DocVQA',
            'config_name': 'DocVQA',
            'split': 'validation',
            'image_field': 'image',
            'id_field': 'questionId',
            'use_auth_token': True
        },
        'infovqa_val': {
            'dataset_path': 'lmms-lab/DocVQA',
            'config_name': 'InfographicVQA',
            'split': 'validation',
            'image_field': 'image',
            'id_field': 'question_id',
            'use_auth_token': True
        },
        'textvqa_val': {
            'dataset_path': 'lmms-lab/TextVQA',
            'split': 'validation',
            'image_field': 'image',
            'id_field': 'question_id',
            'use_auth_token': True
        },
        'vqav2': {
            'dataset_path': 'HuggingFaceM4/VQAv2',
            'split': 'validation',
            'image_field': 'image',
            'id_field': 'question_id',
            'use_auth_token': False
        }
    }
    
    def __init__(self, output_base_dir: str = "/data/silviase/lmms_eval"):
        self.output_base_dir = Path(output_base_dir)
    
    def extract_images(self, task_name: str, limit: int = None, force: bool = False):
        """Extract images from a specific dataset."""
        
        if task_name not in self.DATASET_CONFIGS:
            print(f"❌ Unknown task: {task_name}")
            print(f"Available tasks: {', '.join(self.DATASET_CONFIGS.keys())}")
            return
        
        config = self.DATASET_CONFIGS[task_name]
        output_dir = self.output_base_dir / task_name / 'images'
        
        # Check if already extracted
        if output_dir.exists() and not force:
            existing_count = len(list(output_dir.glob('*.*')))
            if existing_count > 0:
                print(f"ℹ️  {existing_count} images already exist in {output_dir}")
                response = input("Do you want to re-extract? (y/N): ").strip().lower()
                if response != 'y':
                    print("Skipping extraction.")
                    return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Loading dataset: {config['dataset_path']}...")
        
        # Load dataset
        try:
            dataset_kwargs = {}
            if config.get('use_auth_token'):
                dataset_kwargs['token'] = True
            
            # Add config name if specified
            if config.get('config_name'):
                dataset = load_dataset(
                    config['dataset_path'],
                    config['config_name'],
                    split=config['split'],
                    **dataset_kwargs
                )
            else:
                dataset = load_dataset(
                    config['dataset_path'],
                    split=config['split'],
                    **dataset_kwargs
                )
            
            # Apply limit if specified
            total_samples = len(dataset)
            if limit and limit < total_samples:
                dataset = dataset.select(range(limit))
                print(f"✅ Loaded {limit} samples (limited from {total_samples})")
            else:
                print(f"✅ Loaded {total_samples} samples")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("You may need to authenticate with HuggingFace:")
            print("  huggingface-cli login")
            return
        
        # Extract and save images
        actual_limit = min(limit, len(dataset)) if limit else len(dataset)
        print(f"💾 Extracting {actual_limit} images to {output_dir}...")
        
        saved_count = 0
        error_count = 0
        
        # Create index mapping file
        index_mapping = {}
        
        for idx, sample in enumerate(tqdm(dataset, desc="Extracting images", total=actual_limit)):
            try:
                # Get image
                image = sample.get(config['image_field'])
                if image is None:
                    error_count += 1
                    continue
                
                # Determine ID
                if config['id_field']:
                    doc_id = sample.get(config['id_field'], idx)
                else:
                    doc_id = idx
                
                # Convert to PIL Image if needed
                if not isinstance(image, Image.Image):
                    if isinstance(image, bytes):
                        image = Image.open(io.BytesIO(image))
                    else:
                        error_count += 1
                        continue
                
                # Save image
                image_path = output_dir / f"{doc_id}.png"
                image.save(image_path, 'PNG')
                
                # Store mapping
                index_mapping[str(doc_id)] = {
                    'index': idx,
                    'filename': f"{doc_id}.png",
                    'question': sample.get('question', ''),
                    'answer': sample.get('answer', sample.get('answers', ''))
                }
                
                saved_count += 1
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {e}")
                error_count += 1
        
        # Save index mapping
        mapping_file = output_dir.parent / 'image_index.json'
        with open(mapping_file, 'w') as f:
            json.dump(index_mapping, f, indent=2)
        
        print(f"\n✅ Extraction complete!")
        print(f"  • Saved: {saved_count} images")
        print(f"  • Errors: {error_count}")
        print(f"  • Output: {output_dir}")
        print(f"  • Index: {mapping_file}")
    
    def extract_all_datasets(self, limit: int = None, force: bool = False):
        """Extract images from all configured datasets."""
        for task_name in self.DATASET_CONFIGS.keys():
            print(f"\n{'='*60}")
            print(f"Processing: {task_name}")
            print('='*60)
            self.extract_images(task_name, limit=limit, force=force)
    
    def verify_images(self, task_name: str):
        """Verify extracted images for a task."""
        output_dir = self.output_base_dir / task_name / 'images'
        
        if not output_dir.exists():
            print(f"❌ No images found for {task_name}")
            return
        
        images = list(output_dir.glob('*.*'))
        print(f"📊 Images for {task_name}:")
        print(f"  • Total: {len(images)}")
        
        # Check file sizes
        total_size = sum(img.stat().st_size for img in images)
        print(f"  • Total size: {total_size / (1024*1024):.1f} MB")
        
        # Sample some images
        if images:
            print(f"  • Sample files:")
            for img in images[:5]:
                size_kb = img.stat().st_size / 1024
                print(f"    - {img.name} ({size_kb:.1f} KB)")
        
        # Check index file
        index_file = output_dir.parent / 'image_index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
            print(f"  • Index entries: {len(index)}")


def main():
    parser = argparse.ArgumentParser(description='Extract dataset images for visualization')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract images from dataset')
    extract_parser.add_argument('--task', type=str, help='Task name (e.g., chartqa, docvqa_val)')
    extract_parser.add_argument('--all', action='store_true', help='Extract all datasets')
    extract_parser.add_argument('--limit', type=int, help='Limit number of images to extract')
    extract_parser.add_argument('--force', action='store_true', help='Force re-extraction')
    extract_parser.add_argument('--output-dir', type=str,
                               default='/data/silviase/lmms_eval',
                               help='Base output directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify extracted images')
    verify_parser.add_argument('--task', type=str, required=True, help='Task name to verify')
    verify_parser.add_argument('--output-dir', type=str,
                               default='/data/silviase/lmms_eval',
                               help='Base output directory')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extractor = DatasetImageExtractor(args.output_dir)
        if args.all:
            extractor.extract_all_datasets(limit=args.limit, force=args.force)
        elif args.task:
            extractor.extract_images(args.task, limit=args.limit, force=args.force)
        else:
            print("Please specify --task or --all")
    
    elif args.command == 'verify':
        extractor = DatasetImageExtractor(args.output_dir)
        extractor.verify_images(args.task)
    
    elif args.command == 'list':
        print("Available datasets:")
        for task_name, config in DatasetImageExtractor.DATASET_CONFIGS.items():
            print(f"  • {task_name}: {config['dataset_path']} ({config['split']})")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()