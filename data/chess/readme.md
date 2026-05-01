# Chess AI — Data Folder

This folder contains data preparation for the chess language model.

## Structure

```
data/chess/
├── download.sh        ← Download all Lichess Elite Database files
├── prepare.py         ← Process PGN → train.bin + val.bin + meta.pkl
├── readme.md          ← This file
└── raw_zips/          ← Downloaded .zip files go here (created by download.sh)
    ├── lichess_elite_2020-06.zip
    ├── lichess_elite_2020-07.zip
    └── ... (66 monthly files)
```

## Step 1: Download Data

```bash
cd data/chess
chmod +x download.sh
bash download.sh
```

**What downloads:**
- Source: https://database.nikonoel.fr
- Filter: 2400+ vs 2200+ ELO (2020-2021), 2500+ vs 2300+ ELO (2022+)
- Format: Monthly `.zip` files containing `.pgn`
- Total: ~66 files, ~6-7 GB compressed

## Step 2: Prepare Training Data

```bash
# Process all files (takes 1-3 hours):
python prepare.py --input_dir=./raw_zips

# Quick test with 10,000 games:
python prepare.py --input_dir=./raw_zips --max_games=10000
```

**Output files:**
- `train.bin` — Training tokens (uint16, ~95% of data)
- `val.bin` — Validation tokens (uint16, ~5% of data)
- `meta.pkl` — Tokenizer: `{vocab_size, stoi, itos}`

## Step 3: Train the Model

From the Nano ChessGPT root directory:

```bash
# Start training (GPT-2 Small architecture):
python train.py config/train_chess_gpt2small.py

# Resume after disconnect:
python train.py config/train_chess_gpt2small.py --init_from=resume
```

## Tokenizer Details

Character-level tokenization. Typical chess PGN vocabulary:

| Token | Meaning |
|-------|---------|
| `a`-`h` | File letters |
| `1`-`8` | Rank numbers |
| `N B R Q K` | Piece names |
| `x` | Capture |
| `+` | Check |
| `#` | Checkmate |
| `=` | Promotion |
| `O` | Castling (O-O or O-O-O) |
| `-` | Castling separator |
| ` ` | Space between moves |
| `\n` | Game separator |

**Vocabulary size: ~30-32 characters**

## PGN Cleaning Process

Input (raw PGN):
```
[Event "Rated Blitz game"]
[White "Magnus"]
[Black "Hikaru"]

1. e4 { [%eval 0.18] } e5 2. Nf3 Nc6 3. Bb5 a6 1-0
```

Output (cleaned moves only):
```
e4 e5 Nf3 Nc6 Bb5 a6
```

Removed: headers, move numbers, comments `{}`, eval annotations `$n`, results.
