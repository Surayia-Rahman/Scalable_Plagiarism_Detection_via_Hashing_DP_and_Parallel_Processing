# src/organize.py

import os
import shutil
import glob
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import config

# Base folder where the reorganized dataset will be written
OUTPUT_BASE = "sorted_dataset"

# The three labeled plagiarism categories used by PAN-PC11
PLAGIARISM_TYPES = ["artificial", "translation", "simulated"]


def setup_folders():
    """
    Creates a clean folder structure for the reorganized dataset.
    Structure:
        sorted_dataset/
            artificial/suspicious
            artificial/source
            translation/suspicious
            translation/source
            simulated/suspicious
            simulated/source
            clean/suspicious
    """
    # Remove existing folder to avoid mixing old and new data
    if os.path.exists(OUTPUT_BASE):
        print(f"Cleaning up old {OUTPUT_BASE} to ensure accuracy...")
        shutil.rmtree(OUTPUT_BASE)
        
    os.makedirs(OUTPUT_BASE)

    # Create directory trees for each plagiarism type
    for t in PLAGIARISM_TYPES:
        base = os.path.join(OUTPUT_BASE, t)
        os.makedirs(os.path.join(base, "suspicious"), exist_ok=True)
        os.makedirs(os.path.join(base, "source"), exist_ok=True)

    # Create clean control group folder
    os.makedirs(os.path.join(OUTPUT_BASE, "clean", "suspicious"), exist_ok=True)


def build_source_map():
    """
    Index ALL source .txt files recursively across all PAN-PC11 "part" folders.

    Returns:
        dict: {filename: full_path}
              Ensures we can find the source document regardless of its folder location.
    """
    print("Indexing ALL source files recursively...")
    source_map = {}

    # Recursively collect all .txt files in source directory hierarchy
    all_sources = glob.glob(os.path.join(config.SOURCE_DIR, "**", "*.txt"), recursive=True)
    
    # Build a fast lookup map
    for path in tqdm(all_sources, desc="Indexing Source Paths"):
        fname = os.path.basename(path)
        source_map[fname] = path
        
    print(f"✅ Indexed {len(source_map)} unique source files.")
    return source_map


def organize_dataset():
    """
    Main routine:
    - Sets up output folders
    - Loads source file index
    - Reads XML files inside suspicious directories
    - Extracts plagiarism type and source_reference
    - Copies files into correct category folders
    - Creates a clean control group with 500 random clean files
    - Prints final verification report
    """
    print("--- STEP 0.5: ROBUST DATASET ORGANIZATION ---")
    
    # 1. Prepare output structure and source mapping
    setup_folders()
    source_map = build_source_map()
    
    # 2. Scan suspicious XML files recursively
    print(f"Scanning Suspicious Directory: {config.SUSPICIOUS_DIR}")
    xml_files = glob.glob(os.path.join(config.SUSPICIOUS_DIR, "**", "*.xml"), recursive=True)
    print(f"✅ Found {len(xml_files)} XML files total.")
    
    # Counters for monitoring the sorting process
    counts = {t: 0 for t in PLAGIARISM_TYPES}
    counts["clean"] = 0
    missing_source_count = 0

    # Temporary storage to later extract 500 clean examples
    clean_files_buffer = []
    
    # 3. Process each XML file one by one
    for xml_path in tqdm(xml_files, desc="Sorting & Copying"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract plagiarism annotations
            features = root.findall(".//feature[@name='plagiarism']")
            susp_filename = os.path.basename(xml_path).replace('.xml', '.txt')
            original_susp_path = os.path.join(os.path.dirname(xml_path), susp_filename)
            
            # Skip if corresponding suspicious .txt file is missing
            if not os.path.exists(original_susp_path):
                continue

            # --- CASE A: CLEAN FILE (no plagiarism tags) ---
            if not features:
                clean_files_buffer.append((original_susp_path, xml_path))
                counts["clean"] += 1
                continue
                
            # --- CASE B: PLAGIARIZED FILE ---
            # Classify based on FIRST plagiarism occurrence
            plag_type = features[0].get('type')
            source_ref = features[0].get('source_reference')
            
            if plag_type in PLAGIARISM_TYPES:
                # Lookup actual source path from master map
                source_path = source_map.get(source_ref)
                
                if source_path:
                    target_dir = os.path.join(OUTPUT_BASE, plag_type)
                    
                    # Copy suspicious text & XML
                    shutil.copy(original_susp_path, os.path.join(target_dir, "suspicious", susp_filename))
                    shutil.copy(xml_path, os.path.join(target_dir, "suspicious", os.path.basename(xml_path)))
                    
                    # Copy the matching source file
                    shutil.copy(source_path, os.path.join(target_dir, "source", source_ref))
                    
                    counts[plag_type] += 1
                else:
                    # Edge case: source file not found in map
                    missing_source_count += 1

        except Exception:
            # XML may be malformed; skip silently
            continue

    # 4. Build a clean control set of exactly 500 examples
    print("Creating Clean Control Group (500 files)...")
    if clean_files_buffer:
        random.shuffle(clean_files_buffer)
        control_group = clean_files_buffer[:500]
        clean_target_dir = os.path.join(OUTPUT_BASE, "clean", "suspicious")
        
        for txt_path, xml_path in control_group:
            shutil.copy(txt_path, clean_target_dir)
            shutil.copy(xml_path, clean_target_dir)

    # 5. Print verification summary
    print("\n" + "="*40)
    print("FINAL VERIFICATION REPORT")
    print("="*40)
    print(f"Total XMLs Scanned:      {len(xml_files)}")
    print("-" * 20)
    print(f"Sorted into 'artificial':  {counts['artificial']}")
    print(f"Sorted into 'simulated':   {counts['simulated']}")
    print(f"Sorted into 'translation': {counts['translation']}")
    print(f"Identified as 'clean':     {counts['clean']}")
    print("-" * 20)
    print(f"Missing Source Files:      {missing_source_count} (Should be 0)")
    
    total_sorted = sum(counts.values())
    print(f"TOTAL FILES SORTED:        {total_sorted}")
    
    if total_sorted == len(xml_files):
        print("\n✅ SUCCESS: Every file was accounted for.")
    else:
        print(f"\n⚠️ WARNING: {len(xml_files) - total_sorted} files were skipped (likely corrupted XMLs or missing txts).")
        
    print(f"Data location: {os.path.abspath(OUTPUT_BASE)}")


if __name__ == '__main__':
    # Entry point for standalone execution
    organize_dataset()
