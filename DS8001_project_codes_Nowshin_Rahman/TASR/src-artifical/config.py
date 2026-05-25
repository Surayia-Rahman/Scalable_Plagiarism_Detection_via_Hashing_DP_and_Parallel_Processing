# src/artificial/config.py
import os # Import the operating system module for path manipulation

# PATHS: Pointing to the NEW Sorted 'Artificial' Dataset
# Base directory path where the source and suspicious folders reside (CHANGE THIS PATH AS NEEDED)
BASE_DIR = r"C:\Users\Surayia Rahman\Downloads\Algo\algo_project_external\sorted_dataset\artificial"
# Full path to the directory containing source documents
SOURCE_DIR = os.path.join(BASE_DIR, "source")
# Full path to the directory containing suspicious documents
SUSPICIOUS_DIR = os.path.join(BASE_DIR, "suspicious")

# Output file names for saving and reporting results
# File to save the computed LSH index structure
INDEX_FILE = "source_index_artificial.pkl"
# File to save the final evaluation report and metrics
REPORT_FILE = "final_report_artificial.csv"

# TUNING FOR ARTIFICIAL (High Recall) - Parameters optimized for high obfuscation
# Length (k) of the character N-gram (shingle) used for hashing
SHINGLE_SIZE = 3
# Number of hash functions used to generate the MinHash signature (signature length)
NUM_HASHES = 100
# Number of rows (r) grouped together into a single LSH band (b = NUM_HASHES / BAND_SIZE)
BAND_SIZE = 4
# Minimum Jaccard similarity required to confirm a plagiarized candidate pair
EXACT_THRESHOLD = 0.45 
# Documents are split into fragments of this size before comparison to detect localized plagiarism
FRAGMENT_SIZE_WORDS = 500