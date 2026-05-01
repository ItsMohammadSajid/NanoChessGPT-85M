# Nano ChessGPT — 85M

> A GPT-2 Small chess language model trained from scratch on 26 million elite Lichess games.

---

## Overview

**Nano ChessGPT** is a character-level transformer model that learns to play chess purely by predicting the next character in a sequence of moves — no rules, no engine, no tree search.

Trained on the **Lichess Elite Database** (2500+ vs 2300+ ELO games), the model learns to:
- Generate valid chess moves in Standard Algebraic Notation (SAN)
- Understand game context across 512-character windows
- Implicitly learn opening theory, tactical patterns, and endgame technique

---

## Architecture

| Parameter | Value |
|-----------|-------|
| **Model** | GPT-2 Small (from scratch) |
| **Parameters** | ~85 Million |
| **Layers** | 12 |
| **Attention Heads** | 12 |
| **Embedding Dim** | 768 |
| **Context Window** | 512 tokens |
| **Vocabulary** | 29 characters (character-level) |
| **Tokenizer** | Custom chess character-level |

### Vocabulary (29 chars)

```
'\n' ' ' '#' '+' '-' '1'-'8' '=' 'B' 'K' 'N' 'O' 'Q' 'R' 'a'-'h' 'x'
```

Every chess move in Standard Algebraic Notation can be expressed with these 29 characters. No BPE, no subword tokenization — just raw characters.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Lichess Elite Database](https://database.nikonoel.fr/) |
| **Filter** | 2500+ vs 2300+ ELO (elite games only) |
| **Period** | June 2020 – November 2025 |
| **Games** | 26,311,052 |
| **Tokens** | 9.19 Billion |
| **Train split** | 95% (~8.73B tokens) |
| **Val split** | 5% (~460M tokens) |
| **Chinchilla ratio** | 5.4× optimal for 85M model |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 6e-4 (cosine decay) |
| **Warmup Steps** | 2,000 |
| **Total Iterations** | 600,000 |
| **Batch Size** | 16 (× 4 gradient accumulation = 64 effective) |
| **Tokens/Iteration** | 32,768 |
| **Weight Decay** | 0.1 |
| **Dropout** | 0.1 |
| **Precision** | float16 (T4) / bfloat16 (A100+) |
| **Target Hardware** | Google Colab T4 (16GB) |

---

## Project Structure

```
NanoChessGPT-85M/
│
├── model.py                        ← Transformer architecture
├── train.py                        ← Training loop
├── sample.py                       ← Text generation / inference
├── configurator.py                 ← Config override system
│
├── config/
│   └── train_chess_gpt2small.py    ← Chess training config (GPT-2 Small)
│
└── data/chess/
    ├── download.sh                 ← Download Lichess Elite Database
    ├── prepare.py                  ← PGN → train.bin + val.bin + meta.pkl
    └── readme.md                   ← Data preparation guide
```

---

## Quick Start

### Step 1: Download Dataset
```bash
cd data/chess
chmod +x download.sh
bash download.sh          # ~6.4 GB, ~16 min
```

### Step 2: Prepare Data
```bash
python prepare.py --input_dir=./raw_zips    # ~35 min, produces 17 GB
```

### Step 3: Train
```bash
# From project root:
python train.py config/train_chess_gpt2small.py

# Resume after disconnect:
python train.py config/train_chess_gpt2small.py --init_from=resume
```

### Step 4: Generate Moves
```bash
python sample.py --out_dir=out-chess-gpt2small \
                 --start="e4 e5 Nf3 " \
                 --num_samples=5
```

---

## Expected Results

Based on comparable research:

| Metric | Expected |
|--------|---------|
| **ELO** | ~1500–1800 |
| **Legal Move Rate** | >99% |
| **Training Time** | ~166–250 hrs (T4 GPU, float16+compile) |

---

## Research References

- Karvonen, A. (2024). *Chess-GPT's Internal World Model*. arXiv / OpenReview.
  - 50M param character-level model → ~1500 ELO, 99.8% legal moves
- Chinchilla Scaling Laws: Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*.

---

## License

MIT License. The transformer architecture is based on Andrej Karpathy's nanoGPT.
