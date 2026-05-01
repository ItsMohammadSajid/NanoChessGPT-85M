# ================================================================
# Chess AI Training Config — GPT-2 Small Architecture
# ================================================================
# Model: Exactly GPT-2 Small (n_layer=12, n_head=12, n_embd=768)
# Data:  Lichess Elite Database (character-level PGN)
# GPU:   Google Colab T4 (16GB VRAM) or Kaggle T4
#
# To train:
#   python train.py config/train_chess_gpt2small.py
#
# To resume after disconnect:
#   python train.py config/train_chess_gpt2small.py --init_from=resume
#
# Estimated time (T4 GPU, float16 + compile=True):
#   ~1.0-1.5 sec/iter → 600K iters ≈ 166-250 hours (14-21 Colab sessions)
# ================================================================

# ── Output ──────────────────────────────────────────────────────
out_dir = 'out-chess-gpt2small'

# ── Logging & Evaluation ────────────────────────────────────────
eval_interval = 500          # Eval every 500 steps (save checkpoint)
eval_iters = 100             # Use 100 batches to estimate loss
log_interval = 25            # Print loss every 25 steps
always_save_checkpoint = True # Save checkpoint at every eval
eval_only = False

# ── Weights & Biases (optional — set to True if you want graphs) ─
wandb_log = False
wandb_project = 'chess-gpt2small'
wandb_run_name = 'chess-gpt2small'

# ── Dataset ─────────────────────────────────────────────────────
dataset = 'chess'            # → reads from data/chess/train.bin + val.bin

# Gradient accumulation: effective batch = batch_size × grad_accum
# T4 has 16GB VRAM. batch_size=16, block_size=512 is safe.
# effective batch = 16 × 4 = 64 sequences per update (same as before)
# batch=16 + grad_accum=4 → less Python loop overhead vs batch=8 + grad_accum=8
batch_size = 16              # Micro-batch (T4 16GB VRAM safely handles this)
gradient_accumulation_steps = 4   # 4 micro-steps → effective batch = 64

# ── Context Window ──────────────────────────────────────────────
# GPT-2 Small original: 1024
# Chess games average ~200-350 chars, rarely exceed 600.
# 512 saves ~2x compute vs 1024 while covering all games.
block_size = 512

# ── Model Architecture — GPT-2 Small (Exactly) ──────────────────
# Original GPT-2 Small: n_layer=12, n_head=12, n_embd=768 → 124M params
# With chess vocab (~32 chars) instead of 50257:
#   Transformer params: ~85M (same layers, same width)
#   Embedding params:   ~tiny (vocab_size × 768 = 32 × 768 = 24,576)
#   Total: ~85M params (close to "GPT-2 Small" transformer core)
n_layer = 12
n_head = 12
n_embd = 768

# vocab_size is automatically read from data/chess/meta.pkl
# You do NOT need to set it here.

# Regularization
# Note: GPT-2 original used dropout=0.0 for pretraining.
# For chess from scratch, a small dropout helps generalization.
dropout = 0.1

# Bias: GPT-2 original used True.
# Modern research shows False is slightly better.
# We keep False for better performance.
bias = False

# ── Optimizer (AdamW) ───────────────────────────────────────────
# Following GPT-2 / Chinchilla recommendations:
learning_rate = 6e-4         # Peak learning rate
max_iters = 600000           # Total training iterations
weight_decay = 1e-1          # Weight decay (0.1)
beta1 = 0.9
beta2 = 0.95                 # Slightly higher than Adam default (0.999)
grad_clip = 1.0              # Gradient clipping

# ── Learning Rate Schedule (Cosine Decay with Warmup) ───────────
decay_lr = True
warmup_iters = 2000          # Linear warmup for first 2000 steps
lr_decay_iters = 600000      # Should equal max_iters (Chinchilla)
min_lr = 6e-5                # Min LR = learning_rate / 10

# ── Hardware ─────────────────────────────────────────────────────
device = 'cuda'              # 'cuda' for GPU, 'cpu' for testing

# NOTE: T4 (Turing/sm_75) does NOT have bfloat16 Tensor Cores → slow!
# The Colab notebook auto-detects GPU and overrides to float16 on T4.
# A100/H100 (sm_80+) can use bfloat16 natively.
# Default here is bfloat16 for A100 compatibility; T4 notebook overrides it.
dtype = 'bfloat16'

# torch.compile() speeds up training ~20-30% via kernel fusion.
# Requires PyTorch >= 2.0. Safe on Colab T4 (PyTorch 2.x).
# First iter will be slow (~1 min compilation) — that is normal.
compile = True               # Enabled: ~20-30% faster training

# ================================================================
# QUICK REFERENCE — Important Numbers
# ================================================================
#
# Tokens per iteration:
#   batch_size × gradient_accumulation_steps × block_size
#   = 16 × 4 × 512 = 32,768 tokens/iter
#
# Time to see all 8.73B tokens once (1 epoch):
#   8.73B / 32,768 = ~266,000 iterations
#
# Total iters = 600,000 → model sees data ~2.25 times (good!)
#
# Chinchilla optimal for our model (85M params):
#   85M × 20 = 1.7B tokens (optimal)
#   We have 8.73B tokens → 5.1× more than optimal ✅
#
# VRAM usage estimate (batch=16, block=512, 85M params, float16):
#   Static (weights+optimizer states): ~1.3 GB
#   Activations (batch=16, seq=512, 12 layers): ~3-5 GB
#   Total: ~4-6 GB → safely fits in T4's 16GB ✅
#
# Speed (float16 + compile=True on T4):
#   Estimated: ~1.0-1.5 sec/iter
#   Per 12hr Colab session: ~29,000-43,000 iters
# ================================================================
