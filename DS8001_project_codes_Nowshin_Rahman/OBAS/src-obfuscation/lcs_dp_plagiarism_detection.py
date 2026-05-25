import json
import time
from nltk.stem import PorterStemmer

LCS_THRESHOLD = 0.15  # 15% threshold
MAX_WORDS = 2000  # Higher limit for better accuracy

stemmer = PorterStemmer()


def read_document(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def stem_text(text):
    # Stemming all words in text
    words = text.lower().split()[:MAX_WORDS]
    return [stemmer.stem(word) for word in words]


def lcs_similarity_stemmed(text1, text2):
    # LCS with stemming
    words1 = stem_text(text1)
    words2 = stem_text(text2)
    
    m, n = len(words1), len(words2)
    
    if m == 0 or n == 0:
        return 0.0
    
    # DP table for LCS - space optimized
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
    
    lcs_length = prev[n]
    
    avg_length = (m + n) / 2
    similarity = lcs_length / avg_length if avg_length > 0 else 0.0
    
    return similarity


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


# Loading all cases
with open('minhash_lsh_results.json', 'r') as f:
    data = json.load(f)

all_cases = data['detailed_results']

# Adding suspicious part
for case in all_cases:
    doc_num = int(case['suspicious_doc']
                  .replace('suspicious-document', '').replace('.txt', ''))
    if doc_num < 2000:
        case['suspicious_part'] = 'part3'
    elif doc_num < 3000:
        case['suspicious_part'] = 'part4'
    elif doc_num < 4000:
        case['suspicious_part'] = 'part8'
    else:
        case['suspicious_part'] = 'part9'

start_time = time.time()
results = []
detected = 0
minhash_detected = 0

for idx, case in enumerate(all_cases):
    suspicious_doc = case['suspicious_doc']
    suspicious_part = case['suspicious_part']
    source_doc = case['source_doc']
    
    # Reading documents
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
    
    # Calculating LCS similarity with stemming
    case_start = time.time()
    similarity = lcs_similarity_stemmed(susp_text, source_text)
    case_time = time.time() - case_start
    
    is_detected = similarity >= LCS_THRESHOLD
    if is_detected:
        detected += 1
    
    if case['detected']:
        minhash_detected += 1
    
    results.append({
        'suspicious_doc': suspicious_doc,
        'source_doc': source_doc,
        'obfuscation_type': case['obfuscation_type'],
        'minhash_detected': case['detected'],
        'lcs_detected': is_detected,
        'similarity': similarity,
        'time_seconds': case_time
    })
    

elapsed = time.time() - start_time

# Results
total = len(results)
minhash_det = sum(1 for r in results if r['minhash_detected'])
lcs_det = sum(1 for r in results if r['lcs_detected'])

# By obfuscation
obf_stats = {}
for r in results:
    obf = r['obfuscation_type']
    if obf not in obf_stats:
        obf_stats[obf] = {'total': 0, 'minhash': 0, 'lcs': 0}
    obf_stats[obf]['total'] += 1
    if r['minhash_detected']:
        obf_stats[obf]['minhash'] += 1
    if r['lcs_detected']:
        obf_stats[obf]['lcs'] += 1

# Overlap analysis
both = sum(1 for r in results
           if r['lcs_detected'] and r['minhash_detected'])
lcs_only = sum(1 for r in results
               if r['lcs_detected'] and not r['minhash_detected'])
mh_only = sum(1 for r in results
              if r['minhash_detected'] and not r['lcs_detected'])

# Save results
output = {
    'configuration': {
        'method': 'LCS with Porter Stemming',
        'max_words': MAX_WORDS,
        'threshold': LCS_THRESHOLD,
        'stemmer': 'PorterStemmer',
        'dataset': 'ALL 7,886 cases'
    },
    'summary': {
        'total_tested': total,
        'minhash_detected': minhash_det,
        'minhash_recall': minhash_det/total*100,
        'lcs_detected': lcs_det,
        'lcs_recall': lcs_det/total*100,
        'total_time_minutes': elapsed/60,
        'total_time_hours': elapsed/3600,
        'avg_time_per_case': elapsed/total if total else 0,
        'by_obfuscation': obf_stats,
        'overlap': {
            'both': both,
            'lcs_only': lcs_only,
            'minhash_only': mh_only
        }
    },
    'detailed_results': results
}

with open('lcs_stemming_all_cases_results.json', 'w') as f:
    json.dump(output, f, indent=2)
