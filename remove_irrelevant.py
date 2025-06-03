import json
import os
from pathlib import Path
import argparse

def filter_relevant_entries(input_dir, output_dir):
    """
    Filter JSON files to keep only entries where sentiment is not 'irrelevant'
    
    Args:
        input_dir (str): Directory containing input JSON files
        output_dir (str): Directory to save filtered JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in input directory
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    processed_count = 0
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if not isinstance(data, list):
                print(f"Warning: {json_file.name} does not contain a list. Skipping.")
                continue
            
            # Filter entries where sentiment is not "irrelevant"
            filtered_data = []
            for entry in data:
                if isinstance(entry, dict) and 'sentiment' in entry:
                    if entry['sentiment'] != 'irrelevant':
                        filtered_data.append(entry)
                else:
                    # Keep entries that don't have sentiment key (optional behavior)
                    print(f"Warning: Entry in {json_file.name} missing 'sentiment' key")
                    filtered_data.append(entry)
            
            # Create output file path
            output_file = output_path / json_file.name
            
            # Write filtered data to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"Processed {json_file.name}: {len(data)} entries -> {len(filtered_data)} relevant entries")
            processed_count += 1
            
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: Invalid JSON format - {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nCompleted! Processed {processed_count} files.")

def main():
    
    input_dir = "output/prompt2"  # Default input directory
    output_dir = "output/relevant"  # Default output directory
    
    # Validate input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    filter_relevant_entries(input_dir, output_dir)

if __name__ == "__main__":
    main()