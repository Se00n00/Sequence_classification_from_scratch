import torch
import torch.nn as nn

from layers.attention import MultiHead_Attention
from layers.feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, pre_normallization=False):
        super(EncoderLayer, self).__init__()

        self.pre_normallization = pre_normallization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHead_Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForward(embed_dim=embed_dim, hidden_dim=ff_dim)
    
    def forward(self, x):

        if(self.pre_normallization): # Pre-Layer Normallization: Provides stable Training
            x = self.layer_norm1(x)
            x = x + self.attention(x)
            x = x + self.feed_forward(self.layer_norm2(x))
        else:                   # Post-Layer Normallization: It Requires Learning warm-up as gradients may diverge during training 
            x = x + self.attention(x)
            x = self.layer_norm1(x)
            x = x + self.feed_forward(x)
            x = self.layer_norm2(x)

        return x

from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward2
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward2(config)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.pre_norm = config.pre_normalization
        
    def forward(self, x, attention_mask=None):
        if self.pre_norm:
            # Pre-normalization
            norm_x = self.norm1(x)
            attn_output = self.attn(norm_x, attention_mask)
            x = x + self.dropout(attn_output)
            
            norm_x = self.norm2(x)
            ff_output = self.ff(norm_x)
            x = x + self.dropout(ff_output)
        else:
            # Post-normalization
            attn_output = self.attn(x, attention_mask)
            x = self.norm1(x + self.dropout(attn_output))
            
            ff_output = self.ff(x)
            x = self.norm2(x + self.dropout(ff_output))
            
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        
    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x