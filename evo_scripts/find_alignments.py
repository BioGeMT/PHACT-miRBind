import os
import sys
import shutil
import argparse

def process_subfolders_with_aln(input_dir, output_dir, clean_comments=True):
   
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_folders = 0
    
    # Get immediate subfolders
    try:
        subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    except PermissionError:
        print(f"Error: Permission denied when accessing '{input_dir}'")
        return 0
    
    # Process each subfolder
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        try:
            aln_files = [f for f in os.listdir(subfolder) if f.endswith('.aln')]
            
            if aln_files:  # If folder contains .aln files
                # Select which .aln file to copy (prioritize result.aln if it exists)
                selected_file = "result.aln" if "result.aln" in aln_files else aln_files[0]
                source_path = os.path.join(subfolder, selected_file)
                
                # Use subfolder name as the output filename
                output_filename = f"{subfolder_name}.aln"
                dest_path = os.path.join(output_dir, output_filename)
                
                # Copy and clean the file if requested
                if clean_comments:
                    with open(source_path, 'r') as src, open(dest_path, 'w') as dst:
                        for line in src:
                            if not line.startswith('#A'):
                                dst.write(line)
                else:
                    shutil.copy2(source_path, dest_path)
                
                processed_folders += 1
                print(f"Processed folder {processed_folders}: {subfolder} -> {dest_path}")
        
        except PermissionError:
            print(f"Warning: Permission denied when accessing '{subfolder}'")
        except Exception as e:
            print(f"Error processing '{subfolder}': {e}")
    
    return processed_folders

def main():
    parser = argparse.ArgumentParser(
        description='Copy one .aln file from each subfolder containing .aln files')
    parser.add_argument('--input', dest='input_dir', required=True,
                        help='Input directory to search in')
    parser.add_argument('--output', dest='output_dir', required=True,
                        help='Output directory for copied files')
    parser.add_argument('--keep-comments', action='store_true', 
                        help='Keep #A comment lines in the output files')
    
    args = parser.parse_args()
    
    folder_count = process_subfolders_with_aln(
        args.input_dir, 
        args.output_dir, 
        clean_comments=not args.keep_comments
    )
    
    print(f"\nComplete! Processed {folder_count} subfolders containing .aln files")
    print(f"Each subfolder contributed exactly ONE .aln file (prioritizing result.aln)")
    print(f"All files have been processed and saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()