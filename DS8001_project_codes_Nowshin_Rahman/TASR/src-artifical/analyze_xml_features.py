# src/artificial/analyze_xml_features.py
import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import numpy as np
import config # Loads src/artificial/config.py

def analyze_plagiarism_features():
    print("--- QUANTITATIVE ANALYSIS OF ARTIFICIAL PLAGIARISM ---")
    
    xml_files = glob.glob(os.path.join(config.SUSPICIOUS_DIR, "*.xml"))
    if not xml_files:
        print(f"Error: No XML files found in {config.SUSPICIOUS_DIR}. Check your organize.py run.")
        return

    print(f"Scanning {len(xml_files)} XML Answer Keys...")

    obfuscation_counts = {}
    lengths = []
    source_references = []
    total_segments = 0

    for xml_file in tqdm(xml_files, desc="Analyzing Features"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for feature in root.findall(".//feature[@name='plagiarism']"):
                total_segments += 1
                
                # 1. Obfuscation Level
                obfuscation = feature.get('obfuscation')
                obfuscation_counts[obfuscation] = obfuscation_counts.get(obfuscation, 0) + 1
                
                # 2. Plagiarism Length
                length_str = feature.get('this_length')
                if length_str:
                    lengths.append(int(length_str))
                
                # 3. Source Reference
                source_ref = feature.get('source_reference')
                if source_ref:
                    source_references.append(source_ref)

        except Exception as e:
            # Skip corrupted XMLs
            continue

    # --- 1. Obfuscation Summary ---
    print("\n" + "="*50)
    print(f"ANALYSIS 1: OBFUSCATION LEVEL ({total_segments} Total Segments)")
    print("="*50)
    
    sorted_counts = dict(sorted(obfuscation_counts.items(), key=lambda item: item[1], reverse=True))
    for level, count in sorted_counts.items():
        percentage = (count / total_segments) * 100
        print(f"-> {level.upper()}: {count} segments ({percentage:.2f}%)")

    # --- 2. Length Summary ---
    print("\n" + "="*50)
    print("ANALYSIS 2: PLAGIARISM LENGTHS (in Characters)")
    print("="*50)
    if lengths:
        print(f"Total Number of Plagiarism Segments: {len(lengths)}")
        print(f"Average Segment Length: {np.mean(lengths):.0f} chars")
        print(f"Median Segment Length:  {np.median(lengths):.0f} chars")
        print(f"Maximum Segment Length: {np.max(lengths)} chars")
        print(f"Minimum Segment Length: {np.min(lengths)} chars")
    else:
        print("No length data found.")

    # --- 3. Source Reuse Summary ---
    print("\n" + "="*50)
    print("ANALYSIS 3: SOURCE REUSE (Measuring Dilution)")
    print("="*50)
    if source_references:
        source_counts = pd.Series(source_references).value_counts()
        total_unique_sources = len(source_counts)
        print(f"Total Unique Source Documents Used: {total_unique_sources}")
        print(f"Average Use Per Source: {total_segments / total_unique_sources:.1f} times")
        
        top_5 = source_counts.head(5)
        print("\nTop 5 Most Reused Source Documents:")
        print(top_5.to_markdown(numalign="left", stralign="left"))
    
    print("\n--- END OF ANALYSIS ---")


if __name__ == '__main__':
    analyze_plagiarism_features()