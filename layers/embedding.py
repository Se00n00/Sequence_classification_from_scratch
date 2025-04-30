import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position_embeddings, dropout_prob):
        super(Embeddings, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps = 1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        seq_length = input.size(1)
        batch_size = input.size(0)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0)
        
        token_embeddings = self.token_embeddings(input)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings     # Add Position Embeddings with token embeddings to include positional information

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_prob)
        max_len = config.max_position_embeddings
        embed_dim = config.embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)