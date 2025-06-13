import argparse
import sys
import collections

def merge_ner_files(source_file_path, base_file_path, ner_type_to_move, output_file_path):
    """
    Replaces a specific NER type in a base file with entries from a source file.

    Args:
        source_file_path (str): Path to the file providing the NER entries (file1).
        base_file_path (str): Path to the base file to be modified (file2).
        ner_type_to_move (str): The specific NER type to move from source to base.
        output_file_path (str): Path to save the merged output file.
    """
    # 1. Read lines from the base file, filtering out the specified NER type.
    final_lines = []
    try:
        print(f"Reading base file and filtering out '{ner_type_to_move}' types...")
        with open(base_file_path, 'r', encoding='utf-8') as f_base:
            for line in f_base:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                # Keep the line if it's malformed or not the type we want to replace
                if len(parts) < 2 or parts[1] != ner_type_to_move:
                    final_lines.append(line)

    except FileNotFoundError:
        print(f"Error: Base file not found at '{base_file_path}'")
        sys.exit(1)

    # 2. Read lines from the source file, keeping only the specified NER type.
    try:
        print(f"Reading source file and extracting '{ner_type_to_move}' types...")
        with open(source_file_path, 'r', encoding='utf-8') as f_source:
            for line in f_source:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                # Keep the line only if it's the specific NER type we are moving
                if len(parts) > 1 and parts[1] == ner_type_to_move:
                    final_lines.append(line)

    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_file_path}'")
        sys.exit(1)

    # 3. Sort the combined lines for consistency (by data_id, then start_time).
    print("Sorting merged data...")
    def sort_key(line):
        parts = line.split('\t')
        try:
            # Attempt to convert data_id to an integer for natural sorting.
            # Handles cases like '82198' and '82198.wav'.
            data_id_str = os.path.splitext(parts[0])[0] if len(parts) > 0 else ''
            data_id = int(data_id_str)
        except (ValueError, IndexError):
            data_id = parts[0] if parts else '' # Fallback to string sort
            
        try:
            # start_time is the 3rd column (index 2)
            start_time = float(parts[2]) 
        except (ValueError, IndexError):
            # Place lines with bad start times at the end
            start_time = float('inf') 
        
        return (data_id, start_time)

    final_lines.sort(key=sort_key)

    # 4. Write the merged and sorted lines to the output file.
    try:
        print(f"Writing merged data to '{output_file_path}'...")
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for line in final_lines:
                f_out.write(line + '\n')
        print("\nOperation successful!")
        print(f"  - Replaced '{ner_type_to_move}' in '{base_file_path}' with data from '{source_file_path}'.")
        print(f"  - Saved result to '{output_file_path}'.")
    except IOError as e:
        print(f"Error: Could not write to output file at '{output_file_path}': {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merges a specific NER type from a source file into a base file, replacing the existing entries of that type.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source_file",
        required=True,
        help="Path to the source file (file1.txt) containing the NER type to move."
    )
    parser.add_argument(
        "--base_file",
        required=True,
        help="Path to the base file (file2.txt) where the NER type will be replaced."
    )
    parser.add_argument(
        "--ner_type",
        required=True,
        help="The NER type to move (e.g., 'DOCTOR', 'PATIENT')."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path for the new merged output file."
    )
    
    args = parser.parse_args()

    merge_ner_files(args.source_file, args.base_file, args.ner_type, args.output_file)


if __name__ == "__main__":
    # The os module is used in the sort key to handle file extensions like .wav
    import os
    main()