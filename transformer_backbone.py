import torch
import torch.nn as nn

class BoaConstrictor(nn.Module):
    def __init__(self, d_model=256, num_layers=1, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_proj(x)
    
    def get_probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
