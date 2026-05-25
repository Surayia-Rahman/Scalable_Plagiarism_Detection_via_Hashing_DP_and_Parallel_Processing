import json
import time
from collections import defaultdict


def read_document(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def jaccard_similarity(text1, text2):  # Computing Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def divide_and_conquer_similarity(text1, text2, min_size=100,
                                  depth=0, max_depth=10):
    words1 = text1.split()
    words2 = text2.split()
    
    # Base case: segments small enough or max depth reached
    if (len(words1) <= min_size or len(words2) <= min_size or
            depth >= max_depth):
        return jaccard_similarity(text1, text2)
    
    # Divide: Splitting both documents in half
    mid1 = len(words1) // 2
    mid2 = len(words2) // 2
    
    text1_left = ' '.join(words1[:mid1])
    text1_right = ' '.join(words1[mid1:])
    text2_left = ' '.join(words2[:mid2])
    text2_right = ' '.join(words2[mid2:])
    
    # Conquer: Comparing all 4 combinations (2x2 matrix)
    similarities = [
        divide_and_conquer_similarity(text1_left, text2_left,
                                      min_size, depth+1, max_depth),
        divide_and_conquer_similarity(text1_left, text2_right,
                                      min_size, depth+1, max_depth),
        divide_and_conquer_similarity(text1_right, text2_left,
                                      min_size, depth+1, max_depth),
        divide_and_conquer_similarity(text1_right, text2_right,
                                      min_size, depth+1, max_depth)
    ]
    
    # Combine: Taking maximum similarity (best matching segment pair)
    return max(similarities)


def get_source_part(doc_name):
    try:
        num = int(doc_name.replace('source-document', '').replace('.txt', ''))
        if num < 500:
            return 'part1'
        elif num < 1000:
            return 'part2'
        elif num < 1500:
            return 'part3'
        elif num < 2000:
            return 'part4'
        elif num < 2500:
            return 'part5'
        elif num < 3000:
            return 'part6'
        elif num < 3500:
            return 'part7'
        elif num < 4000:
            return 'part8'
        elif num < 4500:
            return 'part9'
        elif num < 5000:
            return 'part10'
    except Exception:
        pass
    return None


# Configuration
DC_THRESHOLD = 0.30  # 30% similarity threshold
MIN_SEGMENT_SIZE = 100  # Minimum words per segment
MAX_DEPTH = 8  # Maximum recursion depth

# Loading ground truth
with open('minhash_lsh_results.json', 'r') as f:
    data = json.load(f)

all_cases = data['detailed_results']

# Using ALL cases
sampled = all_cases

# Adding suspicious part mapping
for case in sampled:
    susp_doc = case['suspicious_doc']
    doc_num = int(susp_doc.replace('suspicious-document', '')
                  .replace('.txt', ''))
    if doc_num < 2000:
        case['suspicious_part'] = 'part3'
    elif doc_num < 3000:
        case['suspicious_part'] = 'part4'
    elif doc_num < 4000:
        case['suspicious_part'] = 'part8'
    else:
        case['suspicious_part'] = 'part9'

total_cases = len(sampled)

start_time = time.time()
results = []
detected = 0
minhash_detected = 0
valid_tested = 0

for idx, case in enumerate(sampled): # Iterate through cases
    suspicious_doc = case['suspicious_doc']
    suspicious_part = case['suspicious_part']
    source_doc = case['source_doc']
    
    susp_path = (f'PAN_PC11/suspicious_document/'
                 f'{suspicious_part}/{suspicious_doc}')
    susp_text = read_document(susp_path)
    
    source_part = get_source_part(source_doc)
    if not source_part:
        continue
    
    source_path = f'PAN_PC11/source_document/{source_part}/{source_doc}'
    source_text = read_document(source_path)
    
    if not susp_text or not source_text:
        continue
    
    valid_tested += 1
    
    # Divide & Conquer similarity
    case_start = time.time()
    similarity = divide_and_conquer_similarity(
        susp_text,
        source_text,
        min_size=MIN_SEGMENT_SIZE,
        max_depth=MAX_DEPTH
    )
    case_time = time.time() - case_start
    
    is_detected = similarity >= DC_THRESHOLD
    if is_detected:
        detected += 1
    
    if case.get('detected'):
        minhash_detected += 1
    
    results.append({
        'suspicious_doc': suspicious_doc,
        'source_doc': source_doc,
        'obfuscation_type': case['obfuscation_type'],
        'minhash_detected': case.get('detected', False),
        'dc_detected': is_detected,
        'similarity': similarity,
        'time_seconds': case_time
    })
    

elapsed = time.time() - start_time

# Results
total = len(results)
minhash_det = sum(1 for r in results if r['minhash_detected'])
dc_det = sum(1 for r in results if r['dc_detected'])

# By obfuscation
obf_stats = defaultdict(lambda: {'total': 0, 'minhash': 0, 'dc': 0})
for r in results:
    obf = r['obfuscation_type']
    obf_stats[obf]['total'] += 1
    if r['minhash_detected']:
        obf_stats[obf]['minhash'] += 1
    if r['dc_detected']:
        obf_stats[obf]['dc'] += 1

# Overlap analysis
both = sum(1 for r in results
           if r['dc_detected'] and r['minhash_detected'])
dc_only = sum(1 for r in results
              if r['dc_detected'] and not r['minhash_detected'])
mh_only = sum(1 for r in results
              if r['minhash_detected'] and not r['dc_detected'])

# Saving results
output = {
    'configuration': {
        'method': 'Divide & Conquer with Jaccard Similarity',
        'threshold': DC_THRESHOLD,
        'min_segment_size': MIN_SEGMENT_SIZE,
        'max_recursion_depth': MAX_DEPTH,
        'total_cases': total_cases,
        'valid_tested': valid_tested
    },
    'summary': {
        'total_tested': total,
        'minhash_detected': minhash_det,
        'minhash_recall': minhash_det/total*100,
        'dc_detected': dc_det,
        'dc_recall': dc_det/total*100,
        'total_time_minutes': elapsed/60,
        'avg_time_per_case': elapsed/total if total else 0,
        'by_obfuscation': dict(obf_stats),
        'overlap': {
            'both': both,
            'dc_only': dc_only,
            'minhash_only': mh_only
        }
    },
    'detailed_results': results
}

with open('divide_conquer_results.json', 'w') as f:
    json.dump(output, f, indent=2)
