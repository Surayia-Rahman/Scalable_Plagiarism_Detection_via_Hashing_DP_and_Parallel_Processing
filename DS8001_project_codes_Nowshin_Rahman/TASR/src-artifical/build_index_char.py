# src/artificial/build_index_char.py

import sys
import os

# --- PATH FIXING ---
# Ensures that Python can import modules from the parent directory (e.g., config.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import re
import numpy as np
import pickle
import glob
import zlib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import config


# --- OVERRIDE CONFIG FOR CHAR-GRAMS ---
# We override the shingle size because character-based MinHash
# needs a higher k-value compared to word-based shingles.
CHAR_K = 8  # Use 8-character n-grams (empirically stable for obfuscated text)


def preprocess_text_char(text):
    """
    Preprocess text by:
    - lowering case
    - removing all non-alphanumeric characters
    This simplifies MinHash computation and reduces noise.
    """
    text = re.sub(r'[^a-z0-9]', '', text.lower())
    return text


def get_char_shingles(text, k=CHAR_K):
    """
    Generate character-level n-grams (shingles).
    If the text is shorter than k, return the full string as a single shingle.
    """
    if len(text) < k: 
        return set([text])
    return set([text[i:i+k] for i in range(len(text) - k + 1)])


def build_minhash_signature(shingles, num_hashes=config.NUM_HASHES):
    """
    Build a MinHash signature for a set of shingles.
    - Uses CRC32 hashing for speed
    - Applies XOR with random permutation values
    - Output: NumPy array of minhash values (signature vector)
    """

    # If no shingles exist, return an empty/infinite signature
    if not shingles:
        return np.full(num_hashes, np.inf)

    # Use a fixed seed for reproducibility
    np.random.seed(42)

    # Random permutation values for hashing
    permutations = np.random.randint(0, 2147483647, size=num_hashes)

    # Pre-hash shingles using CRC32
    shingle_hashes = [zlib.crc32(s.encode('utf-8')) & 0xffffffff for s in shingles]
    shingle_arr = np.array(shingle_hashes, dtype=np.int64)

    # Compute MinHash signature: for each permutation, take min XOR
    signature = []
    for p in permutations:
        signature.append(np.min(np.bitwise_xor(shingle_arr, p)))

    return np.array(signature)


def process_source_fragments(filepath):
    """
    Process a single source file and produce:
    - character-cleaned text
    - fragment the text into 3000-character blocks
    - compute shingles + MinHash signatures for each fragment
    - return a list of fragment dictionaries
    """

    try:
        # Read file safely
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            if not raw_text:
                return []

        # Preprocess for character-level hashing
        clean_text = preprocess_text_char(raw_text)

        results = []
        source_fname = os.path.basename(filepath)

        # Fragment size tuned to simulate ~500 words
        chunk_size = 3000

        # Iterate through the cleaned text in fixed-size segments
        for i in range(0, len(clean_text), chunk_size):

            fragment_text = clean_text[i : i + chunk_size]

            # Skip extremely small fragments
            if len(fragment_text) < CHAR_K:
                continue

            shingles = get_char_shingles(fragment_text)
            sig = build_minhash_signature(shingles)

            # Unique fragment ID (filename + character offset)
            fragment_id = f"{source_fname}_{i}"

            # Store fragment information
            results.append({
                'id': fragment_id,
                'sig': sig,
                'path': filepath,
                'offset': i  # Offset in characters
            })

        return results

    except Exception:
        # Fail silently and continue processing other files
        return []


if __name__ == '__main__':
    print("--- STEP 1: INDEXING ARTIFICIAL (CHARACTER N-GRAMS) ---")

    # Path to the Artificial source directory
    artificial_source_dir = os.path.join(config.BASE_DIR, "source")

    # Get all .txt documents in the source folder
    files = glob.glob(os.path.join(artificial_source_dir, "*.txt"))
    print(f"Found {len(files)} source documents.")

    final_index = {}

    # Use 80% of available CPU cores for parallel hashing
    num_workers = max(1, int(cpu_count() * 0.80))

    # Run character-level hashing in parallel across documents
    with Pool(num_workers) as pool:
        results_list = list(tqdm(
            pool.imap_unordered(process_source_fragments, files, chunksize=1),
            total=len(files),
            desc="Hashing Chars"
        ))

    # Flatten and store all fragments into a single dictionary
    for file_fragments in results_list:
        if file_fragments:
            for fragment in file_fragments:
                final_index[fragment['id']] = {
                    'sig': fragment['sig'],
                    'path': fragment['path'],
                    'offset': fragment['offset']
                }

    # Save index to disk (kept separate from the word-based index)
    index_file = "source_index_artificial_char.pkl"
    print(f"Saving index to {index_file}...")
    with open(index_file, 'wb') as f:
        pickle.dump(final_index, f)

    print(f"Total Fragments: {len(final_index)}")
    print("\nDone! Now run the detection script updated for character comparison.")
