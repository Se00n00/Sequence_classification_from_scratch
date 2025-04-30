import torch
import torch.nn as nn

from layers.embedding import Embeddings, PositionalEncoding
from layers.encoderlayer import EncoderLayer, TransformerEncoder

class Transformer_For_Sequence_Classification(nn.Module):
    def __init__(self, config):
        super(Transformer_For_Sequence_Classification, self).__init__()
        
        self.embedding = Embeddings(config.vocab_size, config.embed_dim, config.max_position_embeddings, config.dropout_prob)
        self.layers = nn.ModuleList([EncoderLayer(config.embed_dim, config.num_heads, config.ff_dim, config.pre_normallization) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifer = nn.Linear(config.embed_dim, config.num_labels)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        
        x = x[:, 0]
        x = self.dropout(x)
        x = self.classifer(x)

        return x

class Transformer_For_Sequence_Classification2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = PositionalEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.embed_dim, config.num_labels)  # Fixed spelling from 'classifer' to 'classifier'
        
    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.token_embedding(input_ids)
        embedding_output = self.position_embedding(embedding_output)
        sequence_output = self.encoder(embedding_output, attention_mask)
        first_token_tensor = sequence_output[:, 0]
        
        pooled_output = self.dropout(first_token_tensor)
        logits = self.classifier(pooled_output)
        
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
