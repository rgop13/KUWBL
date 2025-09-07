#!/usr/bin/env python3
"""
Script to align JSONL data structure.
Reorganizes the response generated files to have:
- ["code"] = list of samples
- ["tool"] = list of samples

Usage: python align_data_structure.py <input_directory> <output_directory>
"""

import os
import json
import jsonlines
import argparse
from typing import Dict, List, Any
from collections import defaultdict

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                data.append(line)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data

def save_jsonl_file(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(data)

def align_data_structure(input_dir: str, output_dir: str):
    """
    Align the data structure from individual category files to grouped structure.
    
    Input: Multiple files like code_response_generated.jsonl, tool_response_generated.jsonl, etc.
    Output: Aligned structure where each category has its own organized format.
    """
    
    # Find all response generated files
    response_files = []
    for file in os.listdir(input_dir):
        if file.endswith('_response_generated.jsonl'):
            response_files.append(file)
    
    print(f"Found {len(response_files)} response files:")
    for file in response_files:
        print(f"  - {file}")
    
    # Process each file
    aligned_data = {}
    
    for file in response_files:
        # Extract category name (e.g., 'code' from 'code_response_generated.jsonl')
        category = file.replace('_response_generated.jsonl', '')
        file_path = os.path.join(input_dir, file)
        
        print(f"\nProcessing {category}...")
        
        # Load the data
        samples = load_jsonl_file(file_path)
        print(f"  Loaded {len(samples)} samples")
        
        # Store in aligned structure
        aligned_data[category] = samples
    
    # Save the aligned structure
    output_file = os.path.join(output_dir, 'aligned_data_structure.json')
    print(f"\nSaving aligned structure to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, ensure_ascii=False, indent=2)
    
    # Also save individual category files in the new structure
    for category, samples in aligned_data.items():
        category_file = os.path.join(output_dir, f"{category}_aligned.jsonl")
        save_jsonl_file(samples, category_file)
        print(f"Saved {len(samples)} samples to {category_file}")
    
    # Create a summary
    summary = {
        "total_categories": len(aligned_data),
        "categories": {},
        "structure_info": {
            "format": "Each category contains a list of samples",
            "example_access": "data['code'] = list of code samples, data['tool'] = list of tool samples"
        }
    }
    
    for category, samples in aligned_data.items():
        summary["categories"][category] = {
            "sample_count": len(samples),
            "file_size_mb": round(os.path.getsize(os.path.join(input_dir, f"{category}_response_generated.jsonl")) / 1024 / 1024, 2)
        }
    
    summary_file = os.path.join(output_dir, 'alignment_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nAlignment complete!")
    print(f"Summary saved to {summary_file}")
    print(f"\nStructure overview:")
    for category, info in summary["categories"].items():
        print(f"  [{category}] = {info['sample_count']} samples ({info['file_size_mb']} MB)")

def create_sample_viewer(output_dir: str):
    """Create a sample viewer script to explore the aligned data."""
    viewer_script = '''#!/usr/bin/env python3
"""
Sample viewer for aligned data structure.
Usage: python view_aligned_data.py [category]
"""

import json
import jsonlines
import sys
from pprint import pprint

def load_aligned_data():
    """Load the aligned data structure."""
    try:
        with open('aligned_data_structure.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: aligned_data_structure.json not found. Run align_data_structure.py first.")
        sys.exit(1)

def main():
    data = load_aligned_data()
    
    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category in data:
            print(f"\\n=== {category.upper()} SAMPLES ===")
            print(f"Total samples: {len(data[category])}")
            if data[category]:
                print("\\nFirst sample:")
                pprint(data[category][0])
                if len(data[category]) > 1:
                    print("\\nLast sample:")
                    pprint(data[category][-1])
        else:
            print(f"Category '{category}' not found.")
            print(f"Available categories: {list(data.keys())}")
    else:
        print("=== DATA STRUCTURE OVERVIEW ===")
        for category, samples in data.items():
            print(f"[\\"{category}\\"] = {len(samples)} samples")
        print(f"\\nUsage: python {sys.argv[0]} <category>")
        print(f"Available categories: {list(data.keys())}")

if __name__ == "__main__":
    main()
'''
    
    viewer_file = os.path.join(output_dir, 'view_aligned_data.py')
    with open(viewer_file, 'w') as f:
        f.write(viewer_script)
    os.chmod(viewer_file, 0o755)
    print(f"Created data viewer: {viewer_file}")

def main():
    parser = argparse.ArgumentParser(description="Align JSONL data structure")
    parser.add_argument("input_dir", help="Directory containing *_response_generated.jsonl files")
    parser.add_argument("output_dir", help="Directory to save aligned data structure")
    parser.add_argument("--create-viewer", action="store_true", help="Create a sample viewer script")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    align_data_structure(args.input_dir, args.output_dir)
    
    if args.create_viewer:
        create_sample_viewer(args.output_dir)

if __name__ == "__main__":
    main()
