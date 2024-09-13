import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # Ensure the mask has the right shape: [batch_size, 1, 1, seq_length]
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out

class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(BERT, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.token_type_embedding = nn.Embedding(2, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, 2)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        N, seq_length = input_ids.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(N, seq_length, dtype=torch.long).to(self.device)

        word_embeddings = self.word_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        out = self.dropout(embeddings)

        for layer in self.layers:
            out = layer(out, out, out, attention_mask)

        pooled_output = self.pooler(out[:, 0])
        pooled_output = torch.tanh(pooled_output)
        out = self.fc_out(pooled_output)
        
        return out

"""
Diagramme ASCII de l'architecture BERT

+-----------------------------------------------------+
|                     Input Sequence                  |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|        Word Embedding + Position Embedding          |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                     Dropout Layer                   |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                Transformer Block 1                  |
|  +---------------------------------------------+    |
|  |              Self-Attention Layer           |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
|  |            Feed Forward Layer               |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                Transformer Block 2                  |
|  +---------------------------------------------+    |
|  |              Self-Attention Layer           |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
|  |            Feed Forward Layer               |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                          ...                        |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                Transformer Block N                  |
|  +---------------------------------------------+    |
|  |              Self-Attention Layer           |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
|  |            Feed Forward Layer               |    |
|  +---------------------------------------------+    |
|  |              Add & Norm Layer               |    |
|  +---------------------------------------------+    |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                  [CLS] Token Output                 |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                  Fully Connected Layer               |
+-----------------------------------------------------+
                              |
                              v
+-----------------------------------------------------+
|                   Prediction Output                  |
+-----------------------------------------------------+
"""
