import torch
import torch.nn as nn
import math

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
    

# ---
class SinusoidalEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, device):
        super(SinusoidalEmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        self.register_buffer("positional_embedding", self._get_positional_encoding(max_length, embed_size, device))
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
    
    def _get_positional_encoding(self, max_length, embed_size, device):
        pe = torch.zeros(max_length, embed_size, device=device)                              # Create a tensor of zeros of size (max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)              # Create a tensor of size (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))    # Create a tensor of exp values of 0 to embed_size/2
        
        pe[:, 0::2] = torch.sin(position * div_term)                                          # Apply sin function to even indices, start=0 , step=2
        pe[:, 1::2] = torch.cos(position * div_term)                                          # Apply cos function to odd indices, start=1, step=2
        pe = pe.unsqueeze(0)                                                                  # shape: (1, max_length, embed_size)
        return pe

    def forward(self, x):
        word_embedding = self.embedding(x)                                                  # Convert unique word tokens to word embeddings
        
        positional_embeddings = self.positional_embedding[:, :x.size(1), :].to(x.device)   # Get sinosudal indicies information as positional embeddings          Shape: (1, Seqlen, embed_size)
        x = word_embedding + positional_embeddings                                          # Adds word embedding to positional embedding
        x = self.layer_norm(x)                                                              # Apply layer normalization
        return x