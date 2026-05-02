import os
import pickle
import torch
import chess
from flask import Flask, request, jsonify, render_template

from model import GPTConfig, GPT

# --- Configuration ---
CHECKPOINT_PATH = 'out-chess-gpt2small/ckpt.pt'
META_PATH = 'data/chess/meta.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load Meta (Tokenizer) ---
print(f"Loading meta from {META_PATH}...")
try:
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except Exception as e:
    print(f"Error loading meta.pkl: {e}")
    print("Make sure you have run the prepare.py script!")
    exit(1)

# --- Load Model ---
print(f"Loading model from {CHECKPOINT_PATH} to {DEVICE}...")
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have a valid checkpoint at out-chess-gpt2small/ckpt.pt")
    # Don't exit here, so the user can still view the UI and put the checkpoint later
    model = None

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def get_move():
    if model is None:
        return jsonify({'error': 'Model not loaded on the server. Please check the terminal for errors.'}), 500

    data = request.json
    pgn_history = data.get('history', '').strip()
    
    if pgn_history != '':
        pgn_history += ' '
    else:
        # If history is empty, we must provide a start token (newline) 
        # so the model knows a new game is starting.
        pgn_history = '\n'
        
    print(f"\n[Request] Current History: '{pgn_history}'")

    start_ids = encode(pgn_history)
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

    # Generate next tokens
    # We generate a chunk of text, then extract the very first word.
    max_new_tokens = 15
    temperature = 0.8
    top_k = 10

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if 'cuda' in DEVICE else 'cpu'):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
    # Decode the full generated string
    full_output = decode(y[0].tolist())
    
    # The generated part is the full output minus the original prompt
    generated_part = full_output[len(pgn_history):]
    
    # The model separates moves by spaces. The next move is the first word.
    predicted_move = generated_part.strip().split(' ')[0]
    
    print(f"[Model Output] Full generation: '{generated_part}'")
    print(f"[Model Output] Selected move: '{predicted_move}'")
    
    return jsonify({'move': predicted_move})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("NanoChessGPT Web Interface")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
