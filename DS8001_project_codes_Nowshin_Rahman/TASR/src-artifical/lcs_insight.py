# src/LLM/6_lcs_insight_test.py (ACADEMIC INSIGHT TEST)
import sys
import os
import random
import xml.etree.ElementTree as ET
import glob

# --- PATH FIX: Allow importing 'config' from parent folder ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from rapidfuzz.distance import LCSseq
from tqdm import tqdm
import config
import re

# --- CONFIGURATION ---
TEST_CASE_COUNT = 10 # Number of ground truth pairs to test
random.seed(42)      # Use the same seed for consistent results

def preprocess_text_char(text):
    """Simple preprocessing for character-based comparison."""
    text = re.sub(r'[^a-z0-9]', '', text.lower())
    return text

def load_ground_truth_pairs(test_count):
    """Loads a random sample of ground truth (suspicious, source) pairs."""
    all_pairs = []
    xml_files = glob.glob(os.path.join(config.SUSPICIOUS_DIR, "*.xml"))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            susp_name = os.path.basename(xml_file).replace('.xml', '.txt')
            
            for feature in root.findall(".//feature[@name='plagiarism']"):
                source_ref = feature.get('source_reference')
                if source_ref and feature.get('type') == "artificial": 
                    # We only need one pair per file for this demonstration
                    all_pairs.append((susp_name, source_ref))
                    break 
        except: pass
        
    if len(all_pairs) < test_count:
        print(f"Warning: Only found {len(all_pairs)} ground truth pairs. Testing all of them.")
        return all_pairs
        
    return random.sample(all_pairs, test_count)


# src/LLM/6_lcs_insight_test.py (Lines 37-54)

def calculate_lcs_score(susp_fname, source_fname):
    """Calculates the precise LCS ratio between the entire document pair."""
    try:
        # 1. Get Clean Suspicious Text
        susp_path = os.path.join(config.SUSPICIOUS_DIR, susp_fname)
        with open(susp_path, 'r', encoding='utf-8', errors='ignore') as f:
            susp_clean = preprocess_text_char(f.read())
            
        # 2. Get Clean Source Text
        source_path = os.path.join(config.SOURCE_DIR, source_fname)
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            src_clean = preprocess_text_char(f.read())
            
        if not susp_clean or not src_clean: return 0.0
        
        len_susp = len(susp_clean)
        len_src = len(src_clean)
        len_sum = len_susp + len_src
        
        # 3. Get the LCS Distance (the number of characters NOT in the LCS)
        lcs_distance = LCSseq.distance(susp_clean, src_clean, processor=None, score_cutoff=None)
        
        # 4. CORRECTLY CALCULATE LCS LENGTH (Length = (Len_A + Len_B - Distance) / 2)
        # This gives the true length of the longest common subsequence.
        lcs_length_true = (len_sum - lcs_distance) / 2
        
        # 5. Calculate LCS Similarity Ratio (Length / Average Length of the two strings)
        lcs_ratio = lcs_length_true / (len_sum / 2) if len_sum > 0 else 0
        
        return lcs_ratio
    except Exception:
        return 0.0


if __name__ == '__main__':
    print("--- STEP 6: LCS ACADEMIC INSIGHT TEST (10 FILES) ---")
    
    # 1. Load 10 random ground truth pairs
    test_pairs = load_ground_truth_pairs(TEST_CASE_COUNT)
    print(f"Testing {len(test_pairs)} known ground-truth pairs.")

    results = []
    
    # 2. Calculate LCS Score for each pair
    for susp_fname, source_fname in tqdm(test_pairs, desc="Calculating LCS Ratios"):
        lcs_score = calculate_lcs_score(susp_fname, source_fname)
        results.append({
            'Suspicious File': susp_fname,
            'Source File': source_fname,
            'LCS Score': f"{lcs_score:.4f}"
        })
        
    # 3. Display Results
    df_results = pd.DataFrame(results)
    
    # Ensure all data necessary for LCS calculation (config.SUSPICIOUS_DIR, etc.) is imported.
    print("\n--- Longest Common Subsequence (LCS) Results ---")
    print("This shows the MAX score even string-matching can achieve on high-obfuscation files.")
    print(df_results.sort_values(by='LCS Score', ascending=False).to_markdown(index=False))
    
    avg_lcs = df_results['LCS Score'].astype(float).mean()
    print(f"\nAverage LCS Score: {avg_lcs:.4f}")
    
 
    
