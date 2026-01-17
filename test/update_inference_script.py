#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update the 0_Inference_QWen2.5.py script to use the locally downloaded model.
This script creates a new version called 0_Inference_QWen2.5_local.py.
"""

import os
import sys
import re
import shutil
from pathlib import Path

def update_script(input_file, output_file, model_path):
    """
    Update the model path in the script
    """
    print(f"Updating script: {input_file} -> {output_file}")
    
    # Read the original file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the model path
    # Method 1: Try to find the exact model_name line
    if 'model_name = ' in content:
        content = re.sub(
            r'model_name\s*=\s*r?["\'].*?["\']',
            f'model_name = r"{model_path}"',
            content
        )
    else:
        print("Warning: Could not find model_name line in the script.")
    
    # Add an explanatory comment
    header_comment = """
# This file was automatically generated to use the locally downloaded model.
# Original file: 0_Inference_QWen2.5.py
# Generated on: {datetime.datetime.now()}
"""
    
    # Write the updated content to the new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully created {output_file}")
    return True

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    
    # Check if the original script exists
    original_script = script_dir / "0_Inference_QWen2.5.py"
    if not original_script.exists():
        print(f"Error: Original script {original_script} not found.")
        return False
    
    # Set the output script path
    output_script = script_dir / "0_Inference_QWen2.5_local.py"
    
    # Look for downloaded models
    models_dir = script_dir / "models"
    if not models_dir.exists():
        print(f"Error: Models directory {models_dir} not found. Please download models first.")
        return False
    
    # List available models
    available_models = [d for d in models_dir.iterdir() if d.is_dir()]
    if not available_models:
        print(f"Error: No models found in {models_dir}. Please download models first.")
        return False
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model.name}")
    
    # Get user input for which model to use
    try:
        choice = int(input("\nSelect a model to use (enter number): "))
        if choice < 1 or choice > len(available_models):
            print(f"Error: Invalid choice. Please select a number between 1 and {len(available_models)}.")
            return False
    except ValueError:
        print("Error: Please enter a valid number.")
        return False
    
    selected_model = available_models[choice - 1]
    model_path = selected_model.resolve()
    
    # Update the script
    return update_script(original_script, output_script, model_path)

if __name__ == "__main__":
    import datetime
    if main():
        print("\nDone! You can now run 0_Inference_QWen2.5_local.py to use the locally downloaded model.")
    else:
        print("\nFailed to update the script.")
