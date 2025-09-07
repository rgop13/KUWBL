#!/usr/bin/env python3
"""
Convenient script to run data alignment on your response files.
This script will help you locate and align your JSONL files.
"""

import os
import sys
import glob
from align_data_structure import align_data_structure, create_sample_viewer

def find_response_files(search_paths=None):
    """Find all *_response_generated.jsonl files in common locations."""
    if search_paths is None:
        search_paths = [
            "/data/sjy/project/WBL/DATA/",
            "/data/sjy/project/WBL/",
            "/data/datasets/",
            "/data/",
            ".",
            "../",
            "../../"
        ]
    
    found_files = {}
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            pattern = os.path.join(search_path, "**/*_response_generated.jsonl")
            files = glob.glob(pattern, recursive=True)
            
            if files:
                for file in files:
                    dir_path = os.path.dirname(file)
                    if dir_path not in found_files:
                        found_files[dir_path] = []
                    found_files[dir_path].append(os.path.basename(file))
    
    return found_files

def main():
    print("=== Data Alignment Tool ===")
    print("Looking for *_response_generated.jsonl files...")
    
    # Find files
    found_files = find_response_files()
    
    if not found_files:
        print("No *_response_generated.jsonl files found in common locations.")
        print("Please specify the directory containing your files:")
        print("  python run_alignment.py /path/to/your/files/")
        print()
        print("Expected files:")
        expected_files = [
            "code_response_generated.jsonl",
            "conversation_response_generated.jsonl", 
            "culture_response_generated.jsonl",
            "engineering_response_generated.jsonl",
            "if_response_generated.jsonl",
            "logical_response_generated.jsonl",
            "math_response_generated.jsonl",
            "other_response_generated.jsonl",
            "safety_response_generated.jsonl",
            "science_response_generated.jsonl",
            "technology_response_generated.jsonl",
            "tool_response_generated.jsonl",
            "writing_response_generated.jsonl"
        ]
        for file in expected_files:
            print(f"  - {file}")
        return
    
    # Show found files
    print(f"Found response files in {len(found_files)} location(s):")
    for dir_path, files in found_files.items():
        print(f"\\n  {dir_path}:")
        for file in sorted(files):
            file_path = os.path.join(dir_path, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"    - {file} ({size_mb:.1f} MB)")
    
    # Handle command line argument
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        if not os.path.isdir(input_dir):
            print(f"Error: '{input_dir}' is not a valid directory.")
            return
    else:
        # Use the directory with the most files
        if len(found_files) == 1:
            input_dir = list(found_files.keys())[0]
        else:
            print("\\nMultiple directories found. Please specify which one to use:")
            for i, (dir_path, files) in enumerate(found_files.items(), 1):
                print(f"  {i}. {dir_path} ({len(files)} files)")
            
            try:
                choice = int(input("Enter choice (1-{}): ".format(len(found_files))))
                input_dir = list(found_files.keys())[choice - 1]
            except (ValueError, IndexError):
                print("Invalid choice.")
                return
    
    # Set output directory
    output_dir = os.path.join(input_dir, "aligned_output")
    
    print(f"\\n=== Running Alignment ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run alignment
    try:
        align_data_structure(input_dir, output_dir)
        create_sample_viewer(output_dir)
        
        print(f"\\n=== Success! ===")
        print(f"Aligned data saved to: {output_dir}")
        print(f"\\nTo explore your aligned data:")
        print(f"  cd {output_dir}")
        print(f"  python view_aligned_data.py")
        print(f"  python view_aligned_data.py code")
        print(f"  python view_aligned_data.py tool")
        
    except Exception as e:
        print(f"Error during alignment: {e}")
        return

if __name__ == "__main__":
    main()
