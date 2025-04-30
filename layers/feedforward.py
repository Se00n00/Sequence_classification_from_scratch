import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input):
        input = self.gelu(self.linear_1(input))
        input = self.linear_2(input)
        x = self.dropout(input)

        return x

class FeedForward2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x