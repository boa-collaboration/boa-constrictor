import torch
import torch.nn as nn

class MinGRUCell(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.candidate = nn.Linear(d_model, d_model)
    def forward(self, x, h):
        z = torch.sigmoid(self.gate(x))
        c = torch.tanh(self.candidate(x))
        return (1 - z) * h + z * c

class BoaConstrictor(nn.Module):
    def __init__(self, d_model=256, num_layers=1, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cells = nn.ModuleList([MinGRUCell(d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for cell in self.cells:
            x = cell(x, x)  # simplified
        return self.output_proj(x)
    
    def get_probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
