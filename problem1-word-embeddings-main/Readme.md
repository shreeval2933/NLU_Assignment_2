# Problem 1: Learning Word Embeddings from IIT Jodhpur Data

> **Course**: Natural Language Processing — IIT Jodhpur  
> **Objective**: Train Word2Vec models on textual data collected from IIT Jodhpur sources and analyze the semantic structure captured by the learned embeddings.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
  - [Task 1 — Dataset Preparation](#task-1--dataset-preparation)
  - [Task 2 — Model Training](#task-2--model-training)
  - [Task 3 — Semantic Analysis](#task-3--semantic-analysis)
  - [Task 4 — Visualization](#task-4--visualization)
- [Deliverables](#deliverables)
- [Dataset Statistics](#dataset-statistics)
- [Results Summary](#results-summary)

---

## Project Overview

This project builds Word2Vec embeddings from scratch using text scraped from the IIT Jodhpur official website. It covers the full pipeline from raw web scraping to semantic analysis and 2D visualization.

**Four tasks are implemented:**

| Task | Description |
|------|-------------|
| Task 1 | Web scraping, preprocessing, corpus creation |
| Task 2 | CBOW and Skip-gram training across hyperparameter grid |
| Task 3 | Nearest neighbor search and analogy experiments |
| Task 4 | PCA and t-SNE visualizations of word embeddings |

---

## Repository Structure

```
problem1-word-embeddings/
│
├── Dataset_preparation.ipynb      # Web scraping + preprocessing pipeline
├── Model_training.ipynb           # Word2Vec training (CBOW + Skip-gram)
├── Semantic_analysis.ipynb        # Nearest neighbors + analogy experiments
├── Visualization.ipynb            # PCA and t-SNE word embedding plots
│
├── iitj_data_corpus/                 # Generated corpus folder (after running Task 1)
│   ├── corpus.txt
│   ├── vocabulary.csv
│   └── wordcloud.png
│
├── iitj_skipgram.model               # Saved best Skip-gram model (after Task 2)
├── iitj_cbow.model                   # Saved best CBOW model (after Task 2)
├── training_experiments_report.csv   # Hyperparameter experiment results (after Task 2)
├── word_clusters_comparison.png      # Visualization output (after Task 4)
│
└── README.md
```

---

## Requirements

### Python Version
Python 3.8 or higher

### Install Dependencies

```bash
pip install requests beautifulsoup4 nltk wordcloud matplotlib pandas gensim scikit-learn
```

### Download NLTK Data (automatic on first run, or run manually)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Running on Google Colab

All four scripts are designed to run in **Google Colab**. No local setup is needed beyond uploading files when prompted.

Open a new Colab notebook and paste the contents of each script into separate cells, or upload the `.py` files and run them with:

```python
exec(open("Dataset_preparation.py").read())
```

---

## How to Run

> **Important**: Tasks must be run **in order** (1 → 2 → 3 → 4) because each task depends on outputs from the previous one.

---

### Task 1 — Dataset Preparation

**File**: `Dataset_preparation.ipynb`

**What it does**:
- Scrapes text from IIT Jodhpur website pages (homepage, departments, faculty profiles, academic regulations)
- Removes boilerplate HTML (scripts, navbars, footers)
- Extracts meaningful content from `<p>`, `<h1>`, `<h2>`, `<h3>`, `<li>` tags
- Preprocesses text: lowercasing, punctuation removal, tokenization, stopword removal
- Saves four corpus `.txt` files, a `vocabulary.csv`, and a `wordcloud.png`

**How to run**:

```bash
python Dataset_preparation.py
```

**Output folder**: `iitj_data_corpus/`

```
iitj_data_corpus/
├── corpus.txt                  ← All cleaned data 
├── vocabulary.csv              ← Sorted unique word list
└── wordcloud.png               ← Word frequency visualization
```

---

### Task 2 — Model Training

**File**: `Model_training.ipynb`

**What it does**:
-  Uploads `corpus.txt` file (created in Task 1)
- Trains **54 Word2Vec models** across all hyperparameter combinations:
  - Architectures: CBOW (`sg=0`), Skip-gram (`sg=1`)
  - Embedding dimensions: `50`, `100`, `200`
  - Context window sizes: `2`, `5`, `8`
  - Negative samples: `5`, `10`, `15`
- Saves the best models (`dimension=100`, `window=5`, `neg_samples=5`)
- Exports a full experiment report as CSV

**How to run on Colab**:

1. Run the script — it will prompt you to upload the corpus.txt file
2. After training, the following files are auto-downloaded:
   - `training_experiments_report.csv`
   - `iitj_skipgram.model`
   - `iitj_cbow.model`

**Hyperparameter grid (54 runs total)**:

| Parameter | Values Tested |
|-----------|--------------|
| Architecture | CBOW, Skip-gram |
| Embedding Dimension | 50, 100, 200 |
| Context Window | 2, 5, 8 |
| Negative Samples | 5, 10, 15 |
| Epochs | 30 (fixed) |
| Min Count | 1 (fixed) |

---

### Task 3 — Semantic Analysis

**File**: `Semantic_analysis.ipynb`

**What it does**:
- Loads `iitj_skipgram.model` saved from Task 2
- Reports **top 5 nearest neighbors** (by cosine similarity) for:
  - `research`, `student`, `phd`, `examinations`, `faculty`
- Runs **2 analogy experiments** using vector arithmetic:
  - `Undergraduate : BTech :: Postgraduate : ?`
  - `India : Institute :: IIT : ?`

**How to run**:

```bash
python Semantic_analysis.py
```

> Make sure `iitj_skipgram.model` is in the **same directory** as the script.  
> On Colab, upload the `.model` file when prompted.

**Example output**:
```
Top 5 for 'research':
  -> laboratory (0.8821)
  -> projects (0.8714)
  -> publications (0.8603)
  ...

1. [Undergraduate : BTech :: Postgraduate : ?] -> Result: mtech
2. [India : Institute :: IIT : ?]              -> Result: jodhpur
```

---

### Task 4 — Visualization

**File**: `Visualization.ipynb`

**What it does**:
- Loads both `iitj_skipgram.model` and `iitj_cbow.model`
- Projects word vectors to 2D using **PCA** and **t-SNE**
- Plots a **2×2 comparison grid**:
  - Row 1: PCA — CBOW (left) | Skip-gram (right)
  - Row 2: t-SNE — CBOW (left) | Skip-gram (right)
- Words are colour-coded by semantic cluster:
  - 🔴 **Identity**: iit, jodhpur, indian, institute
  - 🔵 **Academic**: btech, mtech, phd, student
  - 🟢 **Domain**: research, technology, engineering
  - 🟠 **Admin**: faculty, professor, department, course
- Saves output as `word_clusters_comparison.png`

**How to run**:

```bash
python Visualization.py
```

> Make sure both `iitj_skipgram.model` and `iitj_cbow.model` are in the same directory.

---

## Deliverables

| File | Description | Generated By |
|------|-------------|--------------|
| `Dataset_preparation.ipynb` | Scraping + preprocessing source code | — |
| `Model_training.ipynb` | Word2Vec training source code | — |
| `Semantic_analysis.ipynb` | Semantic analysis source code | — |
| `Visualization.ipynb` | Visualization source code | — |
| `iitj_data_corpus/` | Cleaned corpus folder with `.txt` file | Task 1 |
| `iitj_data_corpus/vocabulary.csv` | Sorted unique vocabulary list | Task 1 |
| `iitj_data_corpus/wordcloud.png` | Word frequency cloud image | Task 1 |
| `training_experiments_report.csv` | All 54 experiment results | Task 2 |
| `iitj_skipgram.model` | Best trained Skip-gram model | Task 2 |
| `iitj_cbow.model` | Best trained CBOW model | Task 2 |
| `word_clusters_comparison.png` | PCA + t-SNE visualization (2×2 grid) | Task 4 |

---

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total source URLs scraped | 130+ |
| Total `.txt` corpus files | 1 |
| Data sources | Homepage, Academic Regulations, 10 Departments, 100+ Faculty Profiles |
| Preprocessing steps | HTML removal, lowercasing, punctuation stripping, stopword removal, short-token filtering |

> Exact token counts and vocabulary size are printed when Task 1 is run.

---

## Results Summary

### Word2Vec Best Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Skip-gram (best for rare words / semantic tasks) |
| Embedding Dimension | 100 |
| Context Window | 5 |
| Negative Samples | 5 |
| Epochs | 30 |

### Key Observations

- **Skip-gram** captures rare domain-specific terms better than CBOW due to its word-predicts-context training objective
- **CBOW** converges faster and performs better on high-frequency words
- PCA projections show clear linear separation between Identity and Academic clusters
- t-SNE projections reveal tighter, more localized clusters, especially for the Admin group
- Analogy experiments produce semantically meaningful results for degree hierarchy (UG→BTech, PG→MTech)

---

## Notes

- All scripts are written for **Google Colab** but can be adapted for local execution by removing `from google.colab import files` and replacing upload/download calls with local file paths
- If any faculty profile URL returns a connection error, the script skips it and continues — this is expected behaviour
- Model files (`.model`) must always be kept alongside any auxiliary `.npy` files that Gensim generates automatically in the same directory
