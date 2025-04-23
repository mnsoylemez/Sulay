# Install necessary libraries
#!pip install transformers datasets tqdm sentencepiece nltk torch

# Import required libraries
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import sentencepiece as spm
import nltk
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
import math

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
# Hyperparameters
config = {
    'train_test_split': 0.9,
    'batch_size': 32,
    'block_size': 128,
    'max_iters': 2000,
    'eval_interval': 50,
    'learning_rate': 5e-5,
    'eval_iters': 25,
    'embed_dim': 256,
    'num_heads': 4,
    'num_layers': 4,
    'dropout': 0.5,
    'vocab_size_spm': 5094,
    'num_workers': 0,
    'save_path': "c:/transformer_model.pth" # eğitilmiş modelin kaydedileceği path, tercihen transformer_dec_'vocab_size_spm'_'embed_dim'_'num_heads'_'num_layers'_'block_size'.pth
}

# Set device and seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(148)

# Define Rotary Positional Encoding
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer('emb', emb.unsqueeze(0))

    def forward(self, seq_len, device):
        return self.emb[:, :seq_len, :].to(device)



# Define Transformer Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = RotaryPositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.positional_encoding(T, idx.device)
        x = self.embed(idx) + pos_emb
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.fc(x)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=10):
        for _ in range(max_new_tokens):
            T = idx.shape[1]
            pos_emb = self.positional_encoding(T, idx.device)
            x = self.embed(idx) + pos_emb
            for block in self.transformer_blocks:
                x = block(x)
            x = self.ln(x)
            logits = self.fc(x)[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits = torch.zeros_like(logits).scatter_(1, indices, values)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Define Custom Text Dataset
class CustomTextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

# Load and preprocess dataset
file_path = "c:/dataset.txt"              # eğitim dosyasının path'i
with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

sentences = sent_tokenize(raw_text)
text = ' '.join(sentences)

data_file = "c:/TokDataFile.txt" # geçici dosya            
with open(data_file, "w", encoding="utf-8") as f:
    f.write(text)


# Train SentencePiece model with BPE
spm.SentencePieceTrainer.train(
    input=data_file,
    model_prefix="spm_bpe",  # Changed model prefix as spm_dec_bpe_'vocab_size_spm'_'embed_dim'_'num_heads'_'num_layers'_'block_size'
    vocab_size=config['vocab_size_spm'],
    model_type="bpe",  # Changed model type to BPE
    input_sentence_size=1000000,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    character_coverage=1.0,
)

# Load SentencePiece model with BPE
sp = spm.SentencePieceProcessor(model_file="spm_bpe.model")
tokenized_text = sp.encode(text, out_type=int)  # Tokenize text using BPE
data = torch.tensor(tokenized_text, dtype=torch.long)


# Split data
split_idx = int(config['train_test_split'] * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

# Create datasets and data loaders
train_dataset = CustomTextDataset(train_data, config['block_size'])
val_dataset = CustomTextDataset(val_data, config['block_size'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Initialize model, optimizer, and scheduler
model = TransformerLM(len(sp), config['embed_dim'], config['num_heads'], config['num_layers'], config['dropout']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_iters'])

def train_model(model, train_loader, val_loader, optimizer, scheduler, config):
    def get_batch(loader):
        """Helper function to fetch a single batch."""
        for x, y in loader:
            return x.to(device), y.to(device)

    def evaluate_loss():
        """Evaluate losses for train and validation sets."""
        model.eval()
        losses = {'train': 0, 'val': 0}
        with torch.no_grad():
            for split, loader in [('train', train_loader), ('val', val_loader)]:
                x, y = get_batch(loader)
                _, loss = model(x, y)
                losses[split] = loss.item()
        model.train()
        return losses

    for iter in tqdm(range(config['max_iters']), desc="Training"):
        # Training step
        x, y = get_batch(train_loader)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation and checkpointing
        if iter % config['eval_interval'] == 0:
            losses = evaluate_loss()
            print(f'Iter {iter:5d} | Train Loss: {losses["train"]:6.4f} | Val Loss: {losses["val"]:6.4f}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iter
            }, config['save_path'])

        # Step the scheduler
        scheduler.step()

# Train the model
train_model(model, train_loader, val_loader, optimizer, scheduler, config)


# Generate text
context = torch.tensor([[sp.bos_id()]], dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500, temperature=1.0, top_k=10)
generated_text = sp.decode(generated_ids[0].tolist())
print(generated_text)