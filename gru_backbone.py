import torch
import torch.nn as nn

class BoaConstrictor(nn.Module):
    def __init__(self, d_model=256, num_layers=1, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        return self.output_proj(x)
    
    def get_probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
