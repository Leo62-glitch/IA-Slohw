import torch
import torch.nn as nn
import torch_directml
import re
import os
from torch.utils.data import Dataset, DataLoader

# ====== CONFIG ======
BATCH_SIZE = 64
EPOCHS = 10000
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ_LEN = 5

# ====== DEVICE ======
device = torch_directml.device()

# ====== LECTURE DU FICHIER TEXTE ======
file_path = "datasets.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Fichier non trouvé : {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text[:100000]  # Évite d'exploser la RAM/VRAM

text = text.lower()
text = re.sub(r'[^a-z0-9\s]', '', text)
words = text.split()

print(f"Nombre total de mots dans le texte : {len(words)}")
if len(words) <= SEQ_LEN:
    raise ValueError(f"Pas assez de mots pour entraîner. Il en faut plus de {SEQ_LEN}.")
# Optionnel : limiter la taille si besoin
# text = text[:100_000]

# ====== PRÉTRAITEMENT ======
text = text.lower()
text = re.sub(r'[^a-z0-9\s]', '', text)
words = text.split()

# ====== VOCABULAIRE ======
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# ====== DATASET ======
class TextDataset(Dataset):
    def __init__(self, words, word2idx, seq_len):
        self.data = []
        self.targets = []
        for i in range(len(words) - seq_len):
            seq = words[i:i + seq_len]
            target = words[i + seq_len]
            self.data.append([word2idx[w] for w in seq])
            self.targets.append(word2idx[target])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

dataset = TextDataset(words, word2idx, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====== MODÈLE ======
class TextPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * seq_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = TextPredictor(len(vocab), EMBED_DIM, HIDDEN_DIM, SEQ_LEN).to(device)

# ====== ENTRAÎNEMENT ======
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

# ====== SAUVEGARDE ======
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'word2idx': word2idx,
    'idx2word': idx2word,
    'seq_len': SEQ_LEN
}, "text_predictor_model.pth")

print("✅ Modèle sauvegardé dans 'text_predictor_model.pth'")

# ====== GÉNÉRATION DE TEXTE ======
def generate_text(prompt, length=20):
    model.eval()
    prompt = prompt.lower().split()
    if len(prompt) < SEQ_LEN:
        prompt = [''] * (SEQ_LEN - len(prompt)) + prompt

    current_seq = [word2idx.get(w, 0) for w in prompt[-SEQ_LEN:]]
    generated = prompt.copy()

    for _ in range(length):
        x = torch.tensor([current_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(x)
            next_idx = torch.argmax(pred, dim=1).item()
            next_word = idx2word[next_idx]

        generated.append(next_word)
        current_seq = current_seq[1:] + [next_idx]

    return ' '.join(generated)

# ====== TEST ======
prompt = "artificial intelligence is"
generated = generate_text(prompt)
print("\nTexte généré :")
print(generated)
