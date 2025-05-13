import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embed_dim, model_dim):
        super().__init__()

        self.model_dim = model_dim

        self.WQ = nn.Linear(in_features = embed_dim, out_features = model_dim)
        self.WK = nn.Linear(in_features = embed_dim, out_features = model_dim)
        self.WV = nn.Linear(in_features = embed_dim, out_features = model_dim)

    def forward(self, embeddings):
        Q = self.WQ(embeddings)
        K = self.WK(embeddings)
        V = self.WV(embeddings)

        # Scaled Dot-Product Attention
        dot_product = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1 ))
        scaled_dot_product = dot_product / self.model_dim**0.5
        attention_scores = F.softmax(scaled_dot_product, dim=-1)
        attention = torch.matmul(attention_scores, V)

        return attention, attention_scores

class MultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads To Ensure Proper Concatenation"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model_dim = embed_dim// num_heads

        self.WO = nn.Linear(self.embed_dim, self.embed_dim)
        self.multi_heads = nn.ModuleList([
            Head(self.embed_dim, self.model_dim) for _ in range(self.num_heads)
        ])

    def forward(self, hidden_embeddings):
        head_outputs = [head(hidden_embeddings) for head in self.multi_heads]
        multi_head_output = torch.cat([output for output, _ in head_outputs], dim=-1)
        output = self.WO(multi_head_output)
        return output


class Encoder_Decoder_Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_dim = config.embed_dim// config.num_heads

        self.WQ = nn.Linear(in_features = config.embed_dim, out_features = self.model_dim)
        self.WK = nn.Linear(in_features = config.embed_dim, out_features = self.model_dim)
        self.WV = nn.Linear(in_features = config.embed_dim, out_features = self.model_dim)

    def forward(self, embeddings_q, embeddings_k, embeddings_v):
        Q = self.WQ(embeddings_q)
        K = self.WK(embeddings_k)
        V = self.WV(embeddings_v)

        # Scaled Dot-Product Attention
        dot_product = torch.matmul(Q, K.transpose(dim0=-2, dim1=-1 ))
        scaled_dot_product = dot_product / torch.tensor(self.model_dim**0.5)
        attention_scores = F.softmax(scaled_dot_product, dim=-1)
        attention = torch.matmul(attention_scores, V)

        return attention


class Encoder_Decoder_MultiHead_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads To Ensure Proper Concatenation"

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.model_dim = config.embed_dim // config.num_heads

        self.WO = nn.Linear(self.embed_dim, self.embed_dim)
        self.multi_heads = nn.ModuleList(
            [Encoder_Decoder_Head(config) for _ in range(self.num_heads)]
        )

    def forward(self, embeddings_q, embeddings_k, embeddings_v):
        head_outputs = [head(embeddings_q, embeddings_k, embeddings_v) for head in self.multi_heads]
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.WO(multi_head_output)
        return output


# Config:
#   embed_dim
#   num_heads       embed_dim % num_heads == 0

class LinearAttention(nn.Module):
    def __init__(self, model_dim):
        super(LinearAttention, self).__init__()
        self.model_dim = model_dim
        

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        
        assert self.head_dim * config.num_heads == config.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, x, attention_mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to match scores dimensions
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            scores = scores + extended_mask
        
        # Apply softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Calculate output
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.output(context)
        
        return output