import os
import csv



# Only use suspicious subfolders part3, part4, part8, part9
suspicious_dir = os.path.join('PAN_PC11', 'suspicious_document')
suspicious_parts = ['part3', 'part4', 'part8', 'part9']
available_suspicious = set()
for subfolder in suspicious_parts:
    subfolder_path = os.path.join(suspicious_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for fname in os.listdir(subfolder_path):
            if fname.endswith('.txt'):
                available_suspicious.add((subfolder, fname))

# Filter CSV to only cases where both files exist
input_csv = 'filtered_plagiarism_cases.csv'
output_csv = 'filtered_valid_cases.csv'

source_parts = [f'part{i}' for i in range(1, 11)]

with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    valid_count = 0
    for row in reader:
        src_doc = row['source_document']
        susp_doc = row['suspicious_document']
        # Check suspicious file exists in any subfolder
        found_suspicious = False
        for subfolder, fname in available_suspicious:
            if fname == susp_doc:
                susp_path = os.path.join(suspicious_dir, subfolder, fname)
                found_suspicious = True
                break
        if not found_suspicious:
            continue
        # Check source file exists in any part1-10
        found_source = False
        for part in source_parts:
            src_path = os.path.join('PAN_PC11', 'source_document', part, src_doc)
            if os.path.exists(src_path):
                found_source = True
                break
        if found_source:
            writer.writerow(row)
            valid_count += 1
    print(f"Filtered valid cases written: {valid_count}")
