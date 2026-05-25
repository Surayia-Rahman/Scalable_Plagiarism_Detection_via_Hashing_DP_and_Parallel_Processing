import json
import time
import os
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

# MinHash+LSH parameters
NUM_PERM = 128
LSH_THRESHOLD = 0.01


def create_minhash(text, num_perm=NUM_PERM):  # Creating MinHash from text
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for word in words:
        m.update(word.encode('utf-8'))
    return m


def read_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def get_source_part(source_doc):
    # Determining source part from document name
    parts = source_doc.split('-')[1]
    parts = parts.replace('document', '').replace('.txt', '')
    num = int(parts)
    
    if 0 <= num <= 999:
        return 1
    elif 1000 <= num <= 1999:
        return 2
    elif 2000 <= num <= 2999:
        return 3
    elif 3000 <= num <= 3999:
        return 4
    elif 4000 <= num <= 4999:
        return 5
    elif 5000 <= num <= 5999:
        return 6
    elif 6000 <= num <= 6999:
        return 7
    elif 7000 <= num <= 7999:
        return 8
    elif 8000 <= num <= 8999:
        return 9
    elif 9000 <= num <= 9999:
        return 10
    return None


def main():
    print("="*70)
    print("Targated MINHASH+LSH evaluation on GROUND TRUTH")
    print("="*70)
    
    # Loading ground truth
    with open('ground_truth_analysis.json', 'r') as f:
        data = json.load(f)
    
    all_cases = data['ground_truth_cases']
    
    # Filtering to only source parts 1-10
    cases = [c for c in all_cases
             if get_source_part(c['source_doc']) is not None]
    
    print(f"Cases with source parts 1-10: {len(cases)}")
    print(f"Testing on {len(cases)} targeted pairs...")
    
    # Group by obfuscation type
    by_obfuscation = defaultdict(list)
    for case in cases:
        by_obfuscation[case['obfuscation_type']].append(case)
    
    print("\nBreakdown by obfuscation type:")
    for obf_type in ['none', 'low', 'high', 'unknown']:
        count = len(by_obfuscation[obf_type])
        print(f"  {obf_type}: {count} cases")
    
    # Building LSH index with all source documents from parts 1-10
    print("\nBuilding LSH index for source documents...")
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    
    source_minhashes = {}
    source_parts_to_load = range(1, 11)
    
    for src_part in source_parts_to_load:
        src_dir = f'PAN_PC11/source_document/part{src_part}'
        src_files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
        
        for src_file in src_files:
            src_path = f'{src_dir}/{src_file}'
            src_text = read_document(src_path)
            if src_text:
                src_minhash = create_minhash(src_text)
                source_minhashes[src_file] = src_minhash
                lsh.insert(src_file, src_minhash)
        
        print(f"  Loaded part {src_part}: {len(src_files)} documents")
    
    print(f"Total source documents indexed: {len(source_minhashes)}")
    
    # Testing MinHash+LSH on each ground truth case
    print("\nTesting MinHash+LSH")
    start_time = time.time()
    results = []
    detected_by_obf = defaultdict(int)
    total_by_obf = defaultdict(int)
    
    for case in cases:
        suspicious_doc = case['suspicious_doc']
        suspicious_part = case['suspicious_part']
        source_doc = case['source_doc']
        obf_type = case['obfuscation_type']
        
        # Building suspicious document path
        susp_path = (f'PAN_PC11/suspicious_document/'
                     f'{suspicious_part}/{suspicious_doc}')
        
        # Reading suspicious document
        susp_text = read_document(susp_path)
        if not susp_text:
            continue
        
        # Creating MinHash for suspicious document
        susp_minhash = create_minhash(susp_text)
        
        # Query LSH to find candidates
        candidates = lsh.query(susp_minhash)
        
        # Check if the true source is in the candidates
        detected = source_doc in candidates
        
        # Calculate actual similarity for analysis
        similarity = 0.0
        if source_doc in source_minhashes:
            similarity = susp_minhash.jaccard(source_minhashes[source_doc])
        
        total_by_obf[obf_type] += 1
        if detected:
            detected_by_obf[obf_type] += 1
        
        results.append({
            'suspicious_doc': suspicious_doc,
            'source_doc': source_doc,
            'obfuscation_type': obf_type,
            'similarity': similarity,
            'detected': detected,
            'num_candidates': len(candidates)
        })
    
    elapsed_time = time.time() - start_time
    
    # Overall results
    total_detected = sum(1 for r in results if r['detected'])
    overall_recall = (total_detected / len(results)) * 100
    
    # Saving detailed results
    output = {
        'parameters': {
            'num_perm': NUM_PERM,
            'lsh_threshold': LSH_THRESHOLD
        },
        'summary': {
            'total_cases': len(results),
            'total_detected': total_detected,
            'overall_recall': overall_recall,
            'execution_time_seconds': elapsed_time
        },
        'by_obfuscation': {
            obf_type: {
                'total': total_by_obf[obf_type],
                'detected': detected_by_obf[obf_type],
                'recall': (detected_by_obf[obf_type] /
                           total_by_obf[obf_type] * 100
                           if total_by_obf[obf_type] > 0 else 0)
            }
            for obf_type in ['none', 'low', 'high', 'unknown']
        },
        'detailed_results': results
    }
    
    with open('minhash_lsh_results.json', 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
