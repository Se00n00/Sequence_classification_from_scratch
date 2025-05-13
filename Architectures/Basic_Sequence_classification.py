import torch
import torch.nn as nn

from layers.embedding import Embeddings, PositionalEncoding
from layers.encoderlayer import EncoderLayer, TransformerEncoder
from layers.embedding import SinusoidalEmbeddingLayer

class Sequence_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = SinusoidalEmbeddingLayer(config.vocab_size, config.embed_dim, config.max_length, config.device)
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.final = nn.Linear(config.embed_dim, config.embed_dim)

        self.classification_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, config.num_classess)       # Final linear layer : Outputs Labels > batch, seq_len, labels
        )
        
    def forward(self, input_ids, attention_mask=None):
        # embedding_output = self.token_embedding(input_ids)
        embedding_output = self.position_embedding(input_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)

        first_token_tensor = sequence_output[:, 0]
        
        pooled_output = self.dropout(first_token_tensor)
        logits = self.final(pooled_output)

        logits = self.classification_head(logits)
        
        return logits

# config:
#   voacb_size
#   embed_dim = 128          # Encoder_layer
#   num_layers
#   num_heads
#   ff_dim
#   pre_normallization = True
#   max_position_embeddings     # Embedding
#   dropout_pron
#   num_labels
