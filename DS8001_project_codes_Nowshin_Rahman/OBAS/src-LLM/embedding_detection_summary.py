
import csv
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Only source parts 1-10
source_parts = [f'part{i}' for i in range(1, 11)]
source_base = os.path.join('PAN_PC11', 'source_document')

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def read_segment(filepath, offset, length):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(int(offset))
            return f.read(int(length))
    except UnicodeDecodeError:
        print(f"[WARNING] UTF-8 decode failed for {filepath}, trying latin1.")
        encodings = ['utf-8', 'latin1', 'utf-16', 'cp1252']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc, errors='strict') as f:
                    f.seek(int(offset))
                    return f.read(int(length))
            except Exception as e:
                last_error = e
        print(f"[ERROR] Could not decode {filepath} with tried encodings. Last error: {last_error}")
        return ''

threshold = 0.6  # 60% similarity
results = []
debug_count = 0
detected_artificial = 0
total_artificial = 0
obfuscation_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'samples': []})

# Read filtered valid cases
with open('filtered_valid_cases.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    total_rows = 0
    matched_cases = 0
    valid_cases = 0
    processed_pairs = 0
    skipped_encoding = 0
    for row_idx, row in enumerate(reader):
        total_rows += 1
        src_doc = row['source_document']
        susp_doc = row['suspicious_document']

        # Actual matching logic
        found = False
        for part in source_parts:
            src_path = os.path.join(
                'PAN_PC11', 'source_document', part, src_doc)
            for susp_part in ['part3', 'part4', 'part8', 'part9']:
                susp_path = os.path.join(
                    'PAN_PC11', 'suspicious_document', susp_part, susp_doc)
                if os.path.exists(src_path) and os.path.exists(susp_path):
                    matched_cases += 1
                    valid_cases += 1
                    found = True
                    
                    try:
                        susp_text = read_segment(
                            susp_path, row['this_offset'], row['this_length'])
                        src_text = read_segment(
                            src_path, row['source_offset'], row['source_length'])
                        if not susp_text or not src_text:
                            skipped_encoding += 1
                            
                            continue
                        processed_pairs += 1 # Successfully read both segments

                        susp_emb = model.encode([susp_text])[0] # Suspicious embedding
                        src_emb = model.encode([src_text])[0]# Source embedding
                        sim = cosine_similarity(
                            [susp_emb], [src_emb])[0][0] # Cosine similarity

                        detected = sim >= threshold
            
                        # Obfuscation stats
                        obf = row['obfuscation']
                        obfuscation_stats[obf]['total'] += 1 # Total cases
                        if detected:
                            obfuscation_stats[obf]['detected'] += 1 # Detected cases
                            if len(obfuscation_stats[obf]['samples']) < 5:
                                obfuscation_stats[obf]['samples'].append({
                                    'suspicious_document': row['suspicious_document'],
                                    'source_document': src_doc,
                                    'similarity_percent': float(round(sim * 100, 2))
                                })
                    except Exception:
                        skipped_encoding += 1
                        continue
                    break
            if found:
                break
    #

summary = {
   'obfuscation_detection': {} # Summary stats
}    
for obf, stats in obfuscation_stats.items():
    percent = float(
        round(stats['detected'] / stats['total'] * 100, 2)) if stats['total'] else 0.0
    summary['obfuscation_detection'][obf] = {
        'detection_percent': percent,
        'samples': stats['samples'] 
    }

with open('embedding_detection_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

#
