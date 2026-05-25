

# Plagiarism Detection Project


## Overview
This project implements and compares three algorithms for plagiarism detection on the PAN-PC11 corpus, which contains 7,886 document pairs with varying levels of text obfuscation (none, low, high). The focus is on evaluating detection accuracy and speed for known plagiarism cases, not exhaustive search.


The project utilized a two-phase methodology: first, Obfuscation-Based Algorithmic Synthesis ($\text{OBAS}$), which employed structural methods (LSH and LCS) to establish a performance baseline, showing that obfuscation challenges sequence integrity; second, Type-Agnostic Semantic Refinement ($\text{TASR}$), this focuesd on the 'artificially plagiarised' datatype. This led to further LLM pipelines as to implementing the $\text{MiniLM}$ semantic baseline, revealing the critical need for higher-resolution encoding (like the planned MPNet refinement) to achieve optimal performance.




## Dataset
- **PAN-PC11**: Standard benchmark for plagiarism detection.
- **Ground Truth**: Provided XML file with known suspicious-source document pairs.


## Evaluation Approach
- Only ground truth pairs are tested (not all possible pairs).
- **Recall** is the main metric: Can the algorithm detect the known plagiarism cases?


## Algorithms


### 1. MinHash + LSH (Probabilistic Hashing)
  - Create compact signatures (128 hash values) for each document.
  - Index documents into buckets using Locality Sensitive Hashing (LSH).
  - Query is O(1): similar documents land in the same buckets.
 - **Goal**: Fast, scalable detection of similar documents.
 - **How it works**: Uses MinHash signatures and LSH to quickly find similar documents without comparing all pairs.
 - **Preprocessing**: Lowercasing, word splitting (minimal for speed).
 - **Strengths**: Extremely fast (0.26 sec/case), highest recall (96.8%), memory efficient, scales to millions of documents.
 - **Weaknesses**: Misses very small copied sections, does not localize where plagiarism occurs, struggles with heavy paraphrasing.
 - **Results**: 96.8% recall, 0.26 sec/case.
  - **Libraries**: [datasketch](https://github.com/ekzhu/datasketch) (MinHash, MinHashLSH)


### 2. LCS + Stemming (Dynamic Programming)
- **Goal**: Detect sequential copying, robust to word variations.
- **How it works**:
  - Apply Porter stemming to normalize words.
  - Use dynamic programming to find the longest common subsequence (LCS) of words.
  - Space-optimized: O(n) memory.
- **Preprocessing**: Lowercasing, Porter stemming.
- **Strengths**: Captures sequential matches, handles word variations, interpretable.
- **Weaknesses**: Lowest recall, order dependent, biased against long documents.
- **Results**: 20.6% recall, 0.15 sec/case (fastest).
 - **Libraries**: [nltk](https://www.nltk.org/) (PorterStemmer)


### 3. Divide & Conquer (Recursive Segmentation)
- **Goal**: Detect partial plagiarism, even if only a segment is copied.
- **How it works**:
  - Recursively split documents into halves (up to depth 8, min 100 words).
  - Compare all segment pairs using Jaccard similarity at the base case.
  - Return the maximum similarity found.
- **Preprocessing**: Lowercasing, word splitting.
- **Strengths**: Detects partial plagiarism, robust to rearrangement.
- **Weaknesses**: Slowest, higher computational cost.
- **Results**: 27.4% recall, 1.2 sec/case.
 - **Libraries**: Standard Python (no external libraries required)


### 4. SBERT Embedding-based Detection
- **Goal**: Detect semantic similarity using sentence embeddings (SBERT).
- **How it works**:
  - Encode documents using a pre-trained SBERT model to obtain dense vector representations.
  - Compute cosine similarity between suspicious and source document embeddings.
  - Detect plagiarism if similarity exceeds a set threshold.
- **Preprocessing**: Lowercasing, sentence splitting, SBERT tokenization.
- **Strengths**: Captures semantic similarity, robust to paraphrasing and word order changes.
- **Weaknesses**: Slower than hashing, requires GPU/CPU for embedding computation, threshold tuning needed.
- **Results**: See below for detection rates by obfuscation type.
 - **Libraries**: [sentence-transformers](https://www.sbert.net/) (SBERT), [torch](https://pytorch.org/) (PyTorch backend)


## Performance Summary
| Metric         | MinHash + LSH | LCS + Stemming | Divide & Conquer |
|---------------|---------------|----------------|------------------|
| **Recall**    | 96.8%         | 20.6%          | 27.4%            |
| **Time/case** | 0.26 sec      | 0.15 sec       | 1.2 sec          |


### By Obfuscation Type
| Obfuscation | MinHash Recall | LCS Recall | D&C Recall |
|-------------|----------------|------------|------------|
| None        | 97.96%         | 60.5%      | 47.2%      |
| Low         | 97.02%         | 23.1%      | 29.9%      |
| High        | 95.81%         | 15.8%      | 22.6%      |
| Unknown     | 97.85%         | 19.2%      | 28.1%      |


### SBERT Embedding-based Detection Results
| Obfuscation | Detection Rate |
|-------------|---------------|
| None        | 88.66%        |
| Low         | 73.56%        |
| High        | 46.11%        |
| Unknown     | 36.89%        |


**Sample SBERT Results:**
- **None**: 98.9%, 99.2%, 94.2%, 98.9%, 97.9% (top 5 sample similarities)
- **Low**: 85.8%, 82.9%, 65.5%, 68.5%, 83.3%
- **High**: 72.9%, 66.4%, 74.8%, 72.9%, 65.6%
- **Unknown**: 69.6%, 80.0%, 86.3%, 86.4%, 92.1%


See `embedding_detection_summary.json` for full details and more samples.


## Usage
- See individual scripts for each algorithm:
  - `minhash_lsh_plagiarism_detection.py`
  - `lcs_dp_plagiarism_detection.py`
  - `divide_conquer_plagiarism_detection.py`
- Results are output as JSON files for further analysis.


## References
- See `COMPARISON_ANALYSIS.md`, `MinHash_LSH_Explanation.md`, `LCS_Stemming_Explanation.md`, and `DivideConquer_Explanation.md` for detailed explanations and code snippets.
—


# Type-Agnostic Semantic Refinement ($\text{TASR}$) for Plagiarism Detection


The TASR repository contains the source code and documentation for the **Semantic Phase** of the project, focusing on the diagnosis and architectural refinement required for **high-obfuscation plagiarism** (Artificial Subset of PAN-PC11).


This approach centered on establishing and analyzing the performance of the semantic baseline ($\text{MiniLM}$), quantitatively proving the need for higher-dimensional encoding.


## 1. Project Structure and File Inventory


Divided into two primary directories, reflecting the structural analysis (Algorithmic Baseline) and the semantic evaluation ($\text{TASR}$).


---


## Prerequisites and Setup


Before running the pipeline, ensure you have Python 3.x installed and install all necessary dependencies:


```bash
# Install core data science and semantic model libraries
pip install numpy pandas nltk scikit-learn sentence-transformers scipy
```


## 2. Algorithmic and Model Approaches (Project Context)


The overall methodology required a comparative analysis across algorithmic and LLM:


### 2.1 Classical Structural Algorithms (Algorithmic Baseline)


These methods were used to provide quantitative proof that lexical structure had been destroyed by obfuscation, justifying the pivot to semantic models.


* **MinHash + LSH:** Used for scalable, character-level near-duplicate detection. The near-zero F1 score proved that high obfuscation effectively defeats hash alignment.
* **LCS (Longest Common Subsequence):** Used to measure preserved sequential structure. An LCS ratio of $\approx 0.35$ provided mathematical confirmation that over $65\%$ of original sequence order was destroyed.


### 2.2 Semantic Baseline and Refinement ($\text{TASR}$)


The semantic phase was structured as a diagnosis-and-refinement cycle.


* **Diagnostic Model (My Focus): MiniLM (384-D)**
  * **Metric:** Cosine Similarity.
  * **Finding:** This model suffered from **Source Dilution**, resulting in a low Precision of **$0.5224$** and $\mathbf{1,066}$ False Positives. This diagnostic failure confirmed the insufficiency of the low-resolution vector space.


* **Optimized Architecture (Future Work): MPNet (768-D)**
  * **Metric:** Euclidean Distance.
  * **Finding:** This architecture is hypothesized to overcome the limitations of the $\text{MiniLM}$ model by utilizing a higher-resolution ($\mathbf{768}$-D) vector space and a proximity-based metric. This refinement is predicted to be the key to achieving high F1 performance.


---


## 3. Detailed File Inventory and Execution Sequence


### 3.1 Structural Analysis Scripts (`src/artificial`)


The script $\texttt{organize.py}$ sorts and restructures the raw dataset into standardized source and suspicious directories to prepare the files for the subsequent indexing and detection pipelines based on the metadata (plagiarism type) found in the XML files for suspicious documents.


These scripts establish the Algorithmic Baseline performance (sequence followed)


| File | Description | Execution Sequence |
|------|-------------|--------------------|
| `config.py` | **Configuration** | Central configuration file for LSH and LCS hyperparameters (e.g., Shingle Size, Band Size, Thresholds). |
| `build_index_char.py` | **Indexing** | Generates MinHash signatures from character $N$-grams of source documents and organizes these signatures into a Locality-Sensitive Hashing (LSH) index for fast candidate retrieval.. |
| `detect_eval_char.py` | **LSH Detection** | Executes the **MinHash + LSH** pipeline using character N-grams. Outputs F1 Score and Confusion Matrix for the LSH baseline. |
| `run_pipeline_aggressive.py` | **LSH Tuning** | Script experimented to liekly see an aggressive, high-recall versions of the LSH pipeline during diagnostic tuning. |
| `lcs_insight.py` | **LCS Analysis** | Calculates the **Longest Common Subsequence (LCS)** ratio for document pairs to quantify the destruction of sequence structure. |
| `analyze_xml_features.py` | **Feature vector analysis** | Calculates the stylometric feature vectors (likely based on Function Word Frequencies or similar metrics) for suspicious documents to perform intrinsic plagiarism analysis. |


source_index_artificial_char.pkl is generated after indexing that contains MinHash signatures, ready for the fast candidate selection via LSH.


A script is produced final_report_artificial_char.csv that stores the details for the candidate matches found by the LSH system


source_index_artificial_aggressive.pkl is generated from the aggressive pipeline (experimental): contains the compressed index of the entire 3,851 source document collection. It is a dictionary mapping fragment IDs to their 100-component MinHash signature vectors.


final_report_artificial_aggressive.csv is also generated after the aggressive piepline,this contains the list of all suspicious document pairs that were identified as plagiarized by the aggressive LSH filter.




### 3.2 Semantic Refinement Scripts (`src/LLM`)


This is the primary execution path for the $\text{TASR}$ methodology, starting with the $\text{MiniLM}$ diagnostic.


| File | Description | Execution Sequence |
|------|-------------|--------------------|
| `ilm_config.py` | **Configuration** | Defines the LLM name ($\text{MiniLM}$ or $\text{MPNet}$), vector dimension, and similarity threshold (Cosine/Euclidean). |
| `1_consolidate_ilm_data.py` | **Data Prep** | **Step 1:** Parses the suspicious documents' XML metadata to generate the definitive ground-truth map for the $\text{Artificial}$ subset. |
| `2_build_semantic_index.py` | **Indexing** | **Step 2:** Generates sentence embeddings (currently $\mathbf{384}$-D $\text{MiniLM}$) and constructs the `.pkl` semantic index file. |
| `3_detect_eval_semantic.py` | **Detection/Eval** | **Step 3:** Uses the semantic index to run the pairwise similarity check, classifying results, and outputting the final $\text{F1}$ Score and confusion matrix. |
| `semantic_index_artificial.pkl` | **Artifact** | The saved index file containing the generated sentence embeddings. |
| `final_report_semantic_artificial.csv` | **Artifact** | The saved file containing the embedding pairs. |






---


## 4.Future Work and MPNet Hyperparameters


The $\text{MPNet}$ phase remains a vital future direction to validate the $\text{TASR}$ hypothesis.


To run the $\mathbf{768}$-D $\text{MPNet}$ analysis (which will require significant computational resources), update `ilm_config.py` with:


* **Model:** `all-MPNet-base-v2`
* **Distance Metric:** **Euclidean Distance**
* **Threshold:** A starting threshold should be tuned in the range of **1.0 to 1.2** for Euclidean distance.


---








## Authors
- [Jakia Nowshin], [Surayia Rahman]


---
_Last updated: December 11, 2025_

