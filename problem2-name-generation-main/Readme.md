# Problem 2 — Character-Level Name Generation Using RNN Variants

> **NLP Assignment | Indian Institute of Technology Jodhpur**  
> Course: Natural Language Processing  
> Framework: PyTorch (Google Colab)

---

## Overview

This project implements and compares three recurrent neural network architectures for **character-level Indian name generation**:

| Architecture | Key Feature | Params |
|---|---|---|
| **Vanilla RNN** | Baseline Elman RNN with tanh activation | 223,325 |
| **BiLSTM Encoder-Decoder** | BiLSTM encoder + unidirectional LSTM decoder | 3,364,445 |
| **RNN + Attention** | Causal Bahdanau-style attention over prefix history | 362,333 |

All models are trained on a corpus of **1,000 Indian names** and evaluated on novelty rate, diversity score, and linguistic realism.

---

## Repository Structure

```
problem2-name-generation/
│
├── Model_implementation.ipynb   # Architecture definitions + training
├── Quantitative_evaluation.ipynb # Novelty rate & diversity scoring
├── Qualitative_analysis.ipynb   # Realism scoring & failure mode analysis
│
├── TrainingNames.txt                 # 1,000 Indian names dataset (one per line)
│
├── vanilla_rnn.pt                    # Saved Vanilla RNN weights
├── blstm.pt                          # Saved BiLSTM Encoder-Decoder weights
├── rnn_attention.pt                  # Saved RNN+Attention weights
│
├── evaluation_report.csv             # Task 2 quantitative results
├── qualitative_summary.csv           # Task 3 realism summary
├── failure_mode_comparison.csv       # Task 3 failure mode breakdown
├── all_generated_names.csv           # All generated names with labels
│
└── README.md
```

---

## Results Summary

### Task 2 — Quantitative Evaluation (500 names each, T=0.8)

| Model | Novelty Rate | Diversity | Avg Length |
|---|---|---|---|
| Vanilla RNN | **96.20%** | **91.00%** | 5.79 chars |
| BiLSTM Encoder-Decoder | 38.20% | 83.00% | 6.31 chars |
| RNN + Attention | **97.40%** | 70.80% | 5.25 chars |

### Task 3 — Qualitative Evaluation (300 names each)

| Model | Avg Realism | Valid Names % |
|---|---|---|
| Vanilla RNN | 0.9983 / 1.0 | 99.33% |
| BiLSTM Encoder-Decoder | 0.9917 / 1.0 | 97.00% |
| RNN + Attention | 0.9958 / 1.0 | 98.67% |

### Sample Generated Names

```
Vanilla RNN          BiLSTM Enc-Dec       RNN + Attention
-----------          --------------       ---------------
Pamek                Aseem                Bati
Brinda               Mohan                Aninad
Bandha               Sudhir               Ashaka
Ajanmat              Ayasha               Benivdi
Marek                Manyshareshdar       Suha
```

---

## How to Run

### Prerequisites

All notebooks are designed to run on **Google Colab** (free tier works; GPU recommended for Task 1 training).

No local installation required. If running locally:

```bash
pip install torch numpy pandas
```

---

### Task 1 — Train the Models

1. Open `Model_implementation.ipynb` in Google Colab
2. Run all cells in order
3. When prompted, upload `TrainingNames.txt`
4. Training runs for **100 epochs** across all three models (~15–20 min on GPU)
5. At the end, three weight files are automatically downloaded:
   - `vanilla_rnn.pt`
   - `blstm.pt`
   - `rnn_attention.pt`

**Hyperparameters used:**

```python
EMBEDDING_DIM = 64
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
DROPOUT       = 0.3
EPOCHS        = 100
LR_RNN        = 0.003   # Vanilla RNN
LR_LSTM       = 0.001   # BiLSTM Encoder-Decoder
LR_ATTN       = 0.005   # RNN + Attention
LABEL_SMOOTHING = 0.1   # All models
```

---

### Task 2 — Quantitative Evaluation

1. Open `Quantitative_evaluation.ipynb` in Google Colab
2. Run all cells in order
3. When prompted, upload:
   - `TrainingNames.txt`
   - `vanilla_rnn.pt`
   - `blstm.pt`
   - `rnn_attention.pt`
4. The notebook generates **500 names per model** and computes:
   - **Novelty Rate** — % of names not in the training set
   - **Diversity Score** — % of unique names in the generated batch
   - **Average Name Length** — mean character count
   - **Length Distribution** — histogram of name lengths
5. Results are saved to `evaluation_report.csv` and auto-downloaded

---

### Task 3 — Qualitative Analysis

1. Open `Qualitative_analysis.ipynb` in Google Colab
2. Run all cells in order
3. When prompted, upload the same four files as Task 2
4. The notebook generates **300 names per model** and computes:
   - **Realism Score** (0–1) based on four linguistic heuristics
   - **Failure Mode Classification** (TOO_SHORT, TOO_LONG, REPETITION, NO_VOWEL, CONSONANT_CLUSTER, VALID)
   - **Representative valid names** and **failure examples** per model
5. Three CSV files are downloaded:
   - `qualitative_summary.csv`
   - `failure_mode_comparison.csv`
   - `all_generated_names.csv`

---

## Architecture Details

### Architecture 1 — Vanilla RNN

Standard Elman RNN with teacher forcing:

```
hₜ = tanh(Wᵢₕ · xₜ + Wₕₕ · hₜ₋₁ + b)
P(cₜ₊₁) = softmax(fc(dropout(hₜ)))
```

- 2-layer stacked RNN, tanh activation
- Character embedding → RNN → Dropout → Linear → Softmax
- Autoregressive generation from `<SOS>` token

---

### Architecture 2 — BiLSTM Encoder-Decoder

A correct encoder-decoder design that avoids the train/inference mismatch inherent in any purely bidirectional autoregressive approach:

```
Encoder:  h_enc, c_enc = BiLSTM([SOS])           # always encodes only SOS
h₀ = tanh(h_proj([h_fwd ; h_bwd]))               # project 2H → H
c₀ = tanh(c_proj([c_fwd ; c_bwd]))

Decoder:  hₜ, cₜ = LSTM(xₜ, hₜ₋₁, cₜ₋₁)        # unidirectional
P(cₜ₊₁) = softmax(fc(dropout(hₜ)))
```

**Why encoder-decoder?** A raw BiLSTM cannot be used autoregressively — its backward pass requires future characters unavailable at inference. The encoder-decoder split solves this by always encoding only `[SOS]` (identical at train and inference time), then having a plain unidirectional decoder generate characters step-by-step.

---

### Architecture 3 — RNN with Attention

Bahdanau-style additive attention with **causal masking** to match inference exactly:

```
# At training step t:
ctx_t, weights = Attention(hidden[-1], rnn_out[:, :t+1, :])  # causal
logits_t = fc(dropout([rnn_out_t ; ctx_t]))

# At inference (decode_step):
history = cat([history, rnn_out_t], dim=1)  # grows each step
ctx_t, weights = Attention(hidden[-1], history)
logits_t = fc(dropout([rnn_out_t ; ctx_t]))
```

The causal constraint (attending only to positions ≤ t) is the key design decision — it makes the training loop and inference loop **bit-for-bit identical**, eliminating distribution mismatch.

---

## Realism Scoring Criteria (Task 3)

Each generated name is scored on four binary rules (1 point each, max score = 1.0):

| Rule | Criterion | Rationale |
|---|---|---|
| Length plausibility | 3–12 characters | Authentic Indian names rarely exceed 12 chars |
| No triplet repetition | No 3 identical consecutive chars | Repetition = hidden state collapse artifact |
| Vowel presence | At least one vowel (a,e,i,o,u) | Every pronounceable name has ≥1 vowel |
| Consonant cluster | No run of 4+ consecutive consonants | Indian phonology limit |

**Failure modes:**

| Mode | Condition | Example |
|---|---|---|
| `TOO_SHORT` | ≤ 2 characters | "Am", "Ti" |
| `TOO_LONG` | > 12 characters | "Pandharitajan" |
| `REPETITION` | 3+ identical consecutive chars | "Aaashi" |
| `NO_VOWEL` | No vowel present | "Krshtn" |
| `CONSONANT_CLUSTER` | 4+ consecutive consonants | "Adchrit" |
| `VALID` | Passes all four rules | "Arjunesh" |

---

## Training Notes

- **Scheduler:** `CosineAnnealingWarmRestarts(T_0=25, T_mult=2)` — restarts at epochs 25, 75
- **Label smoothing = 0.1** is critical for the BiLSTM to prevent memorisation (without it, loss collapses to ~0.0001 and the model reproduces training names verbatim)
- **Gradient clipping (max_norm=5.0)** prevents exploding gradients in the Vanilla RNN and RNN+Attention
- The BiLSTM's low novelty rate (38.2%) is expected — its 3.36M parameters learn the 1,000-name training set thoroughly; larger datasets would reduce this effect

---

## Dataset

`TrainingNames.txt` contains 1,000 unique Indian names, one per line, all lowercase:

```
aarav
aditya
akash
arjun
arnav
...
```

**Properties:**
- ~50% male, ~50% female
- Regional coverage: North Indian, South Indian, Bengali, Gujarati, Marathi
- Length range: 3–16 characters
- Vocabulary: 26 lowercase Latin characters (a–z)
- Generated using a large language model and manually curated

---

## Citation / Attribution

```
IIT Jodhpur — NLP Assignment
Problem 2: Character-Level Name Generation Using RNN Variants
```