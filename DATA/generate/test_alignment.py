#!/usr/bin/env python3
"""
Test script to demonstrate the data alignment process.
Creates sample data and shows how the alignment works.
"""

import os
import json
import jsonlines
import tempfile
from align_data_structure import align_data_structure

def create_sample_data():
    """Create sample JSONL files for testing."""
    sample_dir = tempfile.mkdtemp(prefix="sample_data_")
    
    # Sample code data
    code_samples = [
        {
            "id": "code_001",
            "query_and_response": [
                {"role": "user", "content": "Write a Python function to sort a list"},
                {"role": "assistant", "content": "def sort_list(lst):\n    return sorted(lst)"}
            ],
            "dedup_priority": 1
        },
        {
            "id": "code_002", 
            "query_and_response": [
                {"role": "user", "content": "Create a class for a binary tree"},
                {"role": "assistant", "content": "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None"}
            ],
            "dedup_priority": 1
        }
    ]
    
    # Sample tool data
    tool_samples = [
        {
            "id": "tool_001",
            "query_and_response": [
                {"role": "user", "content": "How do I use grep to find files?"},
                {"role": "assistant", "content": "Use 'grep -r pattern directory' to search recursively"}
            ],
            "dedup_priority": 1
        }
    ]
    
    # Save sample files
    with jsonlines.open(os.path.join(sample_dir, "code_response_generated.jsonl"), 'w') as writer:
        writer.write_all(code_samples)
    
    with jsonlines.open(os.path.join(sample_dir, "tool_response_generated.jsonl"), 'w') as writer:
        writer.write_all(tool_samples)
    
    return sample_dir

def main():
    print("=== Testing Data Alignment ===")
    
    # Create sample data
    sample_dir = create_sample_data()
    output_dir = tempfile.mkdtemp(prefix="aligned_output_")
    
    print(f"Created sample data in: {sample_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run alignment
    print("\n--- Running Alignment ---")
    align_data_structure(sample_dir, output_dir)
    
    # Show results
    print("\n--- Results ---")
    aligned_file = os.path.join(output_dir, "aligned_data_structure.json")
    with open(aligned_file, 'r') as f:
        aligned_data = json.load(f)
    
    print("Aligned structure:")
    for category, samples in aligned_data.items():
        print(f"  [\"{category}\"] = {len(samples)} samples")
    
    print(f"\nExample access:")
    print(f"  data['code'][0] = {aligned_data['code'][0]['id']}")
    print(f"  data['tool'][0] = {aligned_data['tool'][0]['id']}")
    
    print(f"\nFiles created in {output_dir}:")
    for file in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, file))
        print(f"  - {file} ({size} bytes)")
    
    # Cleanup
    import shutil
    shutil.rmtree(sample_dir)
    shutil.rmtree(output_dir)
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
