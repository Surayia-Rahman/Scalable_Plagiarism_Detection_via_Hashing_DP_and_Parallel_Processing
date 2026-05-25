# src/artificial/2_detect_eval_char.py 

import sys
import os

# --- PATH FIX: Allow importing 'config' from parent folder ---
# Dynamically adds the parent directory of this file to sys.path,
# letting Python import config.py and other project modules.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import re
import numpy as np
import pandas as pd
import pickle
import glob
import zlib
import xml.etree.ElementTree as ET
from tqdm import tqdm
from rapidfuzz import fuzz
import config

# --- CONFIG OVERRIDES FOR CHAR-GRAMS ---
# Must match the settings used during indexing (Step 1).
CHAR_K = 8           # Character n-gram size
WINDOW_SIZE = 3000   # Sliding window size (≈500 words)
STEP_SIZE = 1500     # Overlap = 50%
THRESHOLD = 0.45     # Fuzzy-match similarity threshold

# --- GLOBALS ---
# These are used to share large data structures across worker calls
global_source_data = None
global_lsh_buckets = None
global_bands = None
global_rows = None

def init_globals(source_data, lsh_buckets, bands, rows):
    """
    Stores index and hashing parameters as global state.
    Avoids repeated parameter passing across function calls.
    """
    global global_source_data, global_lsh_buckets, global_bands, global_rows
    global_source_data = source_data
    global_lsh_buckets = lsh_buckets
    global_bands = bands
    global_rows = rows


# --- HELPER FUNCTIONS (CHAR MODE) ---
def preprocess_text_char(text):
    """
    Character-level normalization:
    - Lowercase everything
    - Remove all non-alphanumeric characters
    This allows LSH to work consistently on obfuscated text.
    """
    text = re.sub(r'[^a-z0-9]', '', text.lower())
    return text


def get_char_shingles(text, k=CHAR_K):
    """
    Create all k-length character shingles from the text.
    If text is shorter than k, return it as a single shingle.
    """
    if len(text) < k:
        return set([text])
    return set([text[i:i+k] for i in range(len(text) - k + 1)])


def build_minhash_signature(shingles, num_hashes=config.NUM_HASHES):
    """
    Compute MinHash signature:
    - Pre-hash shingles using CRC32
    - XOR each hash against random permutations
    - Take the minimum for each permutation
    """
    if not shingles:
        return np.full(num_hashes, np.inf)

    np.random.seed(42)  # Ensures reproducibility
    permutations = np.random.randint(0, 2147483647, size=num_hashes)

    shingle_hashes = [zlib.crc32(s.encode('utf-8')) & 0xffffffff for s in shingles]
    shingle_arr = np.array(shingle_hashes, dtype=np.int64)

    signature = []
    for p in permutations:
        signature.append(np.min(np.bitwise_xor(shingle_arr, p)))

    return np.array(signature)


# --- LSH DATA STRUCTURE ---
class LSHIndex:
    """
    Basic LSH implementation using banding.
    - buckets[b] maps band-signatures -> list of document fragment IDs
    """
    def __init__(self, bands, rows):
        self.buckets = [{} for _ in range(bands)]  # list of dicts
        self.rows = rows

    def insert(self, doc_id, signature):
        """
        Insert signature into LSH buckets.
        Each band stores the vector slice for that band.
        """
        for b in range(len(self.buckets)):
            band_sig = tuple(signature[b*self.rows : (b+1)*self.rows])
            if band_sig not in self.buckets[b]:
                self.buckets[b][band_sig] = []
            self.buckets[b][band_sig].append(doc_id)


# --- WORKER: WINNOWING WITH CHARACTERS ---
def check_suspicious_char(susp_path):
    """
    Core detection logic for one suspicious document:
    - Clean text into characters
    - Slide a 3000-character window
    - Compute MinHash signature for each window
    - Query LSH buckets for candidate source fragments
    - Verify using RapidFuzz partial_ratio
    """
    try:
        # Read suspicious document
        with open(susp_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()

        clean_text = preprocess_text_char(raw_text)
        if not clean_text:
            return []

        detected_in_file = []

        # Sliding window over the cleaned character stream
        for i in range(0, len(clean_text), STEP_SIZE):
            window_text = clean_text[i : i + WINDOW_SIZE]
            if len(window_text) < CHAR_K:
                continue

            # Hash the current window
            window_sig = build_minhash_signature(get_char_shingles(window_text))

            # LSH query: collect candidates from matching buckets
            candidates = set()
            for b in range(global_bands):
                band_sig = tuple(window_sig[b*global_rows : (b+1)*global_rows])
                if band_sig in global_lsh_buckets[b]:
                    candidates.update(global_lsh_buckets[b][band_sig])

            # Candidate verification using fuzzy matching
            for cand_id in candidates:
                if cand_id not in global_source_data:
                    continue

                fragment_data = global_source_data[cand_id]

                # Load corresponding source document
                with open(fragment_data['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    src_raw = f.read()
                src_clean = preprocess_text_char(src_raw)

                # Extract same-window fragment from source
                start = fragment_data['offset']
                src_frag_text = src_clean[start : start + WINDOW_SIZE]

                # RapidFuzz: partial string similarity (0–100)
                score = fuzz.partial_ratio(window_text, src_frag_text) / 100.0

                if score >= THRESHOLD:
                    original_source_doc = cand_id.split('_')[0]
                    detected_in_file.append({
                        "suspicious": os.path.basename(susp_path),
                        "source": original_source_doc,
                        "score": score
                    })

                    # Early exit: one match per suspicious file is enough for this implementation
                    return detected_in_file

        return detected_in_file

    except Exception:
        # Fail-safe: return empty so evaluation continues
        return []


# --- EVALUATION LOGIC ---
def run_evaluation(report_file):
    """
    Evaluates LSH predictions against ground-truth XML files.
    Computes:
    - True Positives
    - Precision
    - Recall
    - F1 Score
    """
    print("\n--- PHASE 2: EVALUATION ---")

    ground_truth = set()
    xml_files = glob.glob(os.path.join(config.SUSPICIOUS_DIR, "*.xml"))

    # Extract all (suspicious, source) pairs from PAN-PC11 XML
    print(f"Reading {len(xml_files)} Answer Keys...")
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            susp_name = os.path.basename(xml_file).replace('.xml', '.txt')

            for feature in root.findall(".//feature[@name='plagiarism']"):
                source_ref = feature.get('source_reference')
                if source_ref:
                    ground_truth.add((susp_name, source_ref))

        except:
            pass

    if not os.path.exists(report_file):
        print("No matches found to evaluate.")
        return

    # Load predicted results
    df_pred = pd.read_csv(report_file)
    predictions = set(zip(df_pred['suspicious'], df_pred['source']))

    # Compute metrics
    tp = len(predictions.intersection(ground_truth))
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print("\n--- FINAL METRICS (Character N-Gram Algorithm) ---")
    print(f"TRUE POSITIVES: {tp}")
    print(f"PRECISION:      {precision:.4f}")
    print(f"RECALL:         {recall:.4f}")
    print(f"F1 SCORE:       {f1:.4f}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- STEP 2: CHARACTER N-GRAM DETECTION & EVALUATION ---")

    # Load the character-based index file
    index_file = "source_index_artificial_char.pkl"
    if not os.path.exists(index_file):
        print(f"Error: {index_file} not found. Run 1_build_index_char.py first.")
        exit()

    print(f"Loading {index_file}...")
    with open(index_file, 'rb') as f:
        source_data = pickle.load(f)

    # Compute the number of LSH bands
    num_bands = int(config.NUM_HASHES / config.BAND_SIZE)

    # Build the LSH structure and insert all source fragments
    lsh = LSHIndex(bands=num_bands, rows=config.BAND_SIZE)
    for doc_id, data in tqdm(source_data.items(), desc="Bucketing"):
        lsh.insert(doc_id, data['sig'])

    # Locate suspicious files
    susp_files = glob.glob(os.path.join(config.SUSPICIOUS_DIR, "*.txt"))

    # Initialize global shared variables
    init_globals(source_data, lsh.buckets, num_bands, config.BAND_SIZE)

    results = []
    matches_count = 0
    report_file = "final_report_artificial_char.csv"

    print(f"\nScanning {len(susp_files)} files (Char Mode)...")

    # Batching processing (to keep terminal readable)
    BATCH_SIZE = 200
    total_batches = (len(susp_files) // BATCH_SIZE) + 1

    for start in range(0, len(susp_files), BATCH_SIZE):
        batch = susp_files[start : start + BATCH_SIZE]
        batch_num = (start // BATCH_SIZE) + 1

        print(f"\nProcessing Batch {batch_num} of {total_batches}...")

        batch_matches = 0

        # Process all suspicious files in this batch
        for susp_path in tqdm(batch, desc=f"Batch {batch_num}"):
            res = check_suspicious_char(susp_path)
            if res:
                results.extend(res)
                matches_count += 1
                batch_matches += 1

        # Batch summary
        print(f">>> Batch {batch_num} Complete. Found {batch_matches} matches.")
        print(f">>> TOTAL MATCHES SO FAR: {matches_count}")

    # Save and evaluate predictions if any exist
    if results:
        pd.DataFrame(results).to_csv(report_file, index=False)
        print(f"\nDetection complete. Found {matches_count} matches.")
        run_evaluation(report_file)
    else:
        print("\nNo matches found.")
