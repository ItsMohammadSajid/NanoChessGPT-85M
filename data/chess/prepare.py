"""
Chess PGN Data Preparation for Nano ChessGPT
=======================================
Lichess Elite Database → train.bin + val.bin + meta.pkl

MEMORY-EFFICIENT STREAMING VERSION
─────────────────────────────────────────────────────────────
Previous version: Held ALL 22.5M games in RAM (~17 GB) → OOM Killed
This version:     Processes ONE zip at a time (~300 MB peak) → ✅

Strategy:
- Hardcoded chess vocabulary (29 chars, verified from test data)
- Process each zip file independently: parse → shuffle → encode → write
- Write directly to train.bin / val.bin (no giant in-memory string)
- Random 95/5 train/val assignment per game (statistically equivalent)
- Peak RAM usage: ~400 MB (one zip at a time)

Usage:
    python prepare.py --input_dir=./raw_zips
    python prepare.py --input_dir=./raw_zips --max_games=10000
"""

import os
import re
import sys
import time
import random
import pickle
import zipfile
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# HARDCODED CHESS VOCABULARY
# Verified from actual Lichess Elite data (Oct-Dec 2024 test)
# ─────────────────────────────────────────────────────────────
#
# Chess PGN characters:
#   Files:    a b c d e f g h
#   Ranks:    1 2 3 4 5 6 7 8
#   Pieces:   B K N O Q R  (O = castling: O-O, O-O-O)
#   Special:  x + # = -
#   Space:    ' '
#   Newline:  '\n'  (game separator)
#
CHESS_VOCAB = sorted(list(set('\n #+-12345678=BKNOQRabcdefghx')))
VOCAB_SIZE  = len(CHESS_VOCAB)  # = 29
STOI        = {ch: i for i, ch in enumerate(CHESS_VOCAB)}
ITOS        = {i: ch for i, ch in enumerate(CHESS_VOCAB)}

# ─────────────────────────────────────────
# PGN CLEANING
# ─────────────────────────────────────────

def clean_pgn_game(raw_game: str) -> str:
    """
    Raw PGN game → clean move-only string.

    Input:  '[Event ...]\n...\n1. e4 { eval } e5 2. Nf3 Nc6 1-0'
    Output: 'e4 e5 Nf3 Nc6'
    """
    lines = raw_game.split('\n')
    move_lines = [l for l in lines if not l.strip().startswith('[')]
    moves = ' '.join(move_lines)
    moves = re.sub(r'\{[^}]*\}', '', moves)   # Remove { comments }
    moves = re.sub(r'\$\d+', '', moves)        # Remove $NAG annotations
    moves = re.sub(r'\d+\.+', '', moves)       # Remove move numbers
    moves = re.sub(r'\b(1-0|0-1|1/2-1/2|\*)\b', '', moves)  # Remove results
    moves = re.sub(r'\([^)]*\)', '', moves)    # Remove (variations)
    moves = re.sub(r'\s+', ' ', moves).strip()
    return moves


def parse_pgn_file(pgn_text: str) -> list:
    """Split PGN file text into list of cleaned game strings."""
    raw_games = re.split(r'\n\n(?=\[)', pgn_text)
    games = []
    for raw in raw_games:
        raw = raw.strip()
        if not raw:
            continue
        cleaned = clean_pgn_game(raw)
        if len(cleaned) < 10:   # skip corrupted / empty games
            continue
        # Filter out any characters not in our vocabulary
        cleaned = ''.join(c for c in cleaned if c in STOI)
        if len(cleaned) < 10:
            continue
        games.append(cleaned)
    return games


def encode_game(game: str) -> np.ndarray:
    """Encode a single game string to uint16 numpy array."""
    return np.array([STOI[c] for c in game], dtype=np.uint16)


# ─────────────────────────────────────────
# MAIN PREPARATION (STREAMING)
# ─────────────────────────────────────────

def prepare(input_dir: str, max_games: int = None, val_split: float = 0.05,
            seed: int = 42):
    """
    Memory-efficient pipeline: one zip at a time.
    Peak RAM: ~400 MB (vs 17 GB in the old version).
    """
    input_path = Path(input_dir)
    output_dir = Path(__file__).parent

    print("=" * 60)
    print(" Chess Data Preparation for Nano ChessGPT (Streaming)")
    print(" GPT-2 Small Architecture (Character-Level)")
    print(f" Peak RAM: ~400 MB | Vocab: {VOCAB_SIZE} chars")
    print("=" * 60)

    # ── Find all input files ───────────────────────────────────
    zip_files = sorted(input_path.glob("*.zip"))
    pgn_files = sorted(input_path.glob("*.pgn"))
    all_files = zip_files + pgn_files

    if not all_files:
        print(f"\n❌ ERROR: No .zip or .pgn files found in: {input_dir}")
        sys.exit(1)

    print(f"\n📁 Found {len(zip_files)} zip + {len(pgn_files)} pgn files")
    print(f"🔤 Vocabulary ({VOCAB_SIZE} chars): {repr(''.join(CHESS_VOCAB))}")

    # ── Open output files for streaming write ─────────────────
    train_path = output_dir / 'train.bin'
    val_path   = output_dir / 'val.bin'

    train_f = open(train_path, 'wb')
    val_f   = open(val_path,   'wb')

    random.seed(seed)

    total_games_processed = 0
    total_train_tokens    = 0
    total_val_tokens      = 0
    total_skipped         = 0
    t_start               = time.time()

    # ── Process ONE zip at a time (memory efficient) ───────────
    total_files = len(all_files)

    for file_idx, filepath in enumerate(all_files):
        filename = filepath.name
        print(f"\n[{file_idx+1}/{total_files}] {filename}")
        t0 = time.time()

        # Read the zip/pgn
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    pgn_names = [n for n in zf.namelist() if n.endswith('.pgn')]
                    if not pgn_names:
                        print(f"  ⚠️  No PGN inside, skipping.")
                        continue
                    with zf.open(pgn_names[0]) as f:
                        pgn_text = f.read().decode('utf-8', errors='ignore')
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    pgn_text = f.read()

            # ── Normalize line endings (fix John Hartmann's files 2023-02 to 2024-02)
            # These files use Windows \r\n instead of Unix \n, causing the PGN
            # splitter regex to fail (treats whole file as 1 game).
            pgn_text = pgn_text.lstrip('\ufeff')          # Remove BOM if present
            pgn_text = pgn_text.replace('\r\n', '\n')    # Windows → Unix
            pgn_text = pgn_text.replace('\r', '\n')      # Old Mac → Unix

        except Exception as e:
            print(f"  ⚠️  Failed: {e}")
            continue

        # Parse games
        games = parse_pgn_file(pgn_text)
        parse_time = time.time() - t0
        print(f"  Parsed: {len(games):,} games in {parse_time:.1f}s", end='', flush=True)

        # Shuffle within this zip (important for within-month diversity)
        random.shuffle(games)

        # Early stop if max_games reached
        if max_games is not None:
            remaining = max_games - total_games_processed
            if remaining <= 0:
                break
            if len(games) > remaining:
                games = games[:remaining]

        # Encode and stream-write to train.bin or val.bin
        file_train_tokens = 0
        file_val_tokens   = 0

        for game in games:
            # Encode game + newline separator
            encoded = encode_game(game + '\n')

            # Random 95/5 assignment
            if random.random() < (1.0 - val_split):
                encoded.tofile(train_f)
                file_train_tokens += len(encoded)
            else:
                encoded.tofile(val_f)
                file_val_tokens += len(encoded)

        total_games_processed += len(games)
        total_train_tokens    += file_train_tokens
        total_val_tokens      += file_val_tokens

        write_time = time.time() - t0 - parse_time
        print(f" | train: {file_train_tokens:,} | val: {file_val_tokens:,} toks | {write_time:.1f}s write")

        # Check max_games limit
        if max_games is not None and total_games_processed >= max_games:
            print(f"\n  [max_games={max_games} reached, stopping]")
            break

    # ── Close output files ─────────────────────────────────────
    train_f.close()
    val_f.close()

    # ── Final stats ────────────────────────────────────────────
    total_tokens = total_train_tokens + total_val_tokens
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"✅ Data preparation complete! ({elapsed/60:.1f} min total)")
    print(f"")
    print(f"📌 Summary:")
    print(f"   Files processed:  {min(file_idx+1, total_files)}/{total_files}")
    print(f"   Games processed:  {total_games_processed:,}")
    print(f"   Vocabulary size:  {VOCAB_SIZE} characters")
    print(f"   Total tokens:     {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"   Train tokens:     {total_train_tokens:,}  ({total_train_tokens/total_tokens*100:.1f}%)")
    print(f"   Val tokens:       {total_val_tokens:,}  ({total_val_tokens/total_tokens*100:.1f}%)")
    print(f"")

    # File sizes
    train_gb = os.path.getsize(train_path) / 1024**3
    val_gb   = os.path.getsize(val_path)   / 1024**3
    print(f"   train.bin:  {train_gb:.2f} GB")
    print(f"   val.bin:    {val_gb:.2f} GB")

    # Chinchilla check
    chinchilla_opt = 124_000_000 * 20
    ratio = total_tokens / chinchilla_opt
    print(f"")
    print(f"📊 Chinchilla Scaling Law:")
    print(f"   Optimal tokens: {chinchilla_opt/1e9:.1f}B  (124M params × 20)")
    print(f"   Your tokens:    {total_tokens/1e9:.2f}B")
    if ratio >= 1:
        print(f"   ✅ {ratio:.1f}× Chinchilla optimal — EXCELLENT!")
    else:
        print(f"   ⚠️  {ratio*100:.0f}% of optimal")

    # ── Save meta.pkl ──────────────────────────────────────────
    meta = {
        'vocab_size': VOCAB_SIZE,
        'itos': ITOS,
        'stoi': STOI,
    }
    with open(output_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"")
    print(f"   meta.pkl:   saved (vocab_size={VOCAB_SIZE})")

    # ── Vocabulary table ───────────────────────────────────────
    print(f"")
    print(f"📌 Vocabulary ({VOCAB_SIZE} chars):")
    for i, ch in ITOS.items():
        print(f"   {i:2d} → {repr(ch)}")

    print(f"")
    print(f"📌 Next step:")
    print(f"   python train.py config/train_chess_gpt2small.py")
    print(f"{'='*60}")

    return VOCAB_SIZE, total_tokens


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare Lichess Elite Database for Nano ChessGPT (memory-efficient)'
    )
    parser.add_argument('--input_dir', type=str, default='./raw_zips',
        help='Directory with .zip or .pgn files (default: ./raw_zips)')
    parser.add_argument('--max_games', type=int, default=None,
        help='Max games to process — for testing (default: all)')
    parser.add_argument('--val_split', type=float, default=0.05,
        help='Validation fraction (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed (default: 42)')

    args = parser.parse_args()
    prepare(
        input_dir  = args.input_dir,
        max_games  = args.max_games,
        val_split  = args.val_split,
        seed       = args.seed,
    )
