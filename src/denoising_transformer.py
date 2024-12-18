import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pos_range = torch.arange(max_len).unsqueeze(1)
        dim_range = 2 * (torch.arange(embed_dim ) // 2).unsqueeze(0)

        pe = pos_range / torch.pow(10000, dim_range / embed_dim)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        pe =  pe.unsqueeze(0) # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition


        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):

        x = x + self.pe[:, :x.shape[-2], :]
        return x


class PositionalEncodingRandom(nn.Module):
    def __init__(self, d_model, max_len=5000):
        torch.manual_seed(0)
        super(PositionalEncodingRandom, self).__init__()
        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.randn(max_len).unsqueeze(1)*10
        div_term = torch.randn((d_model+1)//2)*10
        pe[:, 0::2] = 2*torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = 2*torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor x
        return x + self.pe[:, :x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super(LearnablePositionalEncoding, self).__init__()

        # max_len is the maximum sequence length, embed_size is the dimension of the embeddings
        self.max_len = max_len
        self.embed_size = embed_size

        # Create learnable embeddings for each position
        self.position_embeddings = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, seq_len, embed_size)
        :return: Tensor with added position encodings of shape (batch_size, seq_len, embed_size)
        """
        batch_size, seq_len, _ = x.size()

        # Create a tensor of positions (e.g., [0, 1, 2, ..., seq_len-1]) for each sample in the batch
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Get position embeddings for each position
        position_embeds = self.position_embeddings(positions)

        # Add the positional embeddings to the input embeddings
        return x + position_embeds

class DenoisingTransformer(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_encoder_layers, num_decoder_layers, max_len=5000, pos_encoding_embedding='default'):
        super().__init__()

        num_embeddings = 4 # bos and eos tokens

        if pos_encoding_embedding == 'learnable':
          self.pos_encoding_enc = LearnablePositionalEncoding(max_len, hidden_dim)
          self.pos_encoding_dec = LearnablePositionalEncoding(max_len, hidden_dim)
        elif pos_encoding_embedding == 'random':
          self.pos_encoding_enc = PositionalEncodingRandom(d_model=hidden_dim)
          self.pos_encoding_dec = PositionalEncodingRandom(d_model=hidden_dim)
        else:
          self.pos_encoding_enc = PositionalEncoding(embed_dim=hidden_dim)
          self.pos_encoding_dec = PositionalEncoding(embed_dim=hidden_dim)

        self.embedding_layer_enc = nn.Embedding(num_embeddings, hidden_dim)
        self.embedding_layer_dec = nn.Embedding(num_embeddings, hidden_dim)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward= 4 * hidden_dim, batch_first=True, dropout=0.0)

        self.fc_out = nn.Linear(hidden_dim, num_embeddings)

    def forward(self, input, tgt):
        input_emb = self.embedding_layer_enc(input)
        input_emb = self.pos_encoding_enc(input_emb)

        tgt_emb = self.embedding_layer_dec(tgt)
        tgt_emb = self.pos_encoding_dec(tgt_emb)

        decoder_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[-1])

        output = self.transformer(input_emb, tgt_emb, tgt_mask=decoder_mask)
        output = self.fc_out(output)

        return output

    def denoise(self, noised_sequence, num_tokens=None):

        device = next(self.transformer.parameters()).device
        noised_sequence = noised_sequence.to(device)

        if num_tokens is None:
          num_tokens = len(noised_sequence) - 1

        cur = torch.tensor([2], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(noised_sequence.unsqueeze(0), cur.unsqueeze(0))

                cur_argmax = torch.argmax(logits[0, -1, :])

                cur = torch.cat((cur, cur_argmax.unsqueeze(0)))

        return cur