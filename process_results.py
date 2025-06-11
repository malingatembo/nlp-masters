#!/usr/bin/env python3
import os
import shutil
import glob
import json
import re
from datetime import datetime

# Define paths
source_dir = "/home/mtee/projects/arista-nlp-thesis/output/results"
dest_dir = os.path.expanduser("~/Desktop/dataset")

# Ensure destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Get task type from user
valid_tasks = ["qa", "cg", "sm"]
task = ""
while task not in valid_tasks:
    task = input("Enter task type (qa/cg/sm): ").strip().lower()
    if task not in valid_tasks:
        print(f"Invalid task. Please enter one of: {', '.join(valid_tasks)}")

# Get run number from user
while True:
    try:
        run_num = int(input("Enter run number: "))
        padded_num = f"{run_num:03d}"  # Zero-pad to 3 digits
        break
    except ValueError:
        print("Please enter a valid number.")

# Find the three model files in the source directory
source_files = glob.glob(os.path.join(source_dir, "*.json"))

# Check if we have files to process
if not source_files:
    print(f"No files found in {source_dir}")
    exit(1)

# Define the models we're looking for
models = ["tiny_llama", "gpt4_no_rag", "gpt4_rag"]
found_files = {}

# Find files for each model
for file_path in source_files:
    filename = os.path.basename(file_path)
    for model in models:
        if filename.startswith(model):
            found_files[model] = file_path

# Check if we found all model types
if len(found_files) != 3:
    print(f"Warning: Expected 3 model types, but found {len(found_files)}")
    print(f"Missing models: {set(models) - set(found_files.keys())}")

# Process each found file
combined_content = ""
for model, file_path in found_files.items():
    # Create new filename
    new_filename = f"{task}_{padded_num}_{model}.txt"
    dest_path = os.path.join(dest_dir, new_filename)
    
    # Extract the response from JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract response
        if "response" in data:
            response_content = data["response"]
            
            # Also extract context for gpt4_rag
            context_content = ""
            if model == "gpt4_rag" and "context_used" in data:
                context_data = data["context_used"]
                
                # Handle if context_used is a list
                if isinstance(context_data, list):
                    context_text = "\n".join(str(item) for item in context_data)
                else:
                    context_text = str(context_data)
                
                context_content = "\n\nCONTEXT USED:\n" + context_text
            
            # Write extracted content to destination file
            with open(dest_path, 'w') as f:
                f.write(response_content + context_content)
            
            # Add content to combined file
            combined_content += f"=== {model} ===\n"
            combined_content += response_content
            if context_content:
                combined_content += context_content
            combined_content += "\n\n"
            
            print(f"Extracted response from {os.path.basename(file_path)} to {new_filename}")
        else:
            print(f"Warning: No 'response' key found in {os.path.basename(file_path)}")
            # Copy the whole file as fallback
            shutil.copy(file_path, dest_path)
            print(f"Copied whole file instead: {os.path.basename(file_path)} to {new_filename}")
            
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON in {os.path.basename(file_path)}")
        # Copy the whole file as fallback
        shutil.copy(file_path, dest_path)
        print(f"Copied whole file instead: {os.path.basename(file_path)} to {new_filename}")

# Write combined file
combined_filename = f"{task}_{padded_num}.txt"
combined_path = os.path.join(dest_dir, combined_filename)
with open(combined_path, 'w') as f:
    f.write(combined_content)
print(f"Created combined file: {combined_filename}")

# Clean the source directory
for file_path in source_files:
    os.remove(file_path)
print(f"Cleaned {len(source_files)} files from {source_dir}")

print("Task completed successfully!")
