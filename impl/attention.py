import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    """
    The **SelfAttentionBlock** implements the core self-attention mechanism within the transformer architecture. 
    It allows the model to weigh the importance of different parts of the input sequence relative to each other.

    one thing to remember: in multi-head, each head get a piece of the original embedding vector. this allow different heads to learn different patterns
    when we use the c_atten layer, we are adding a linear layer that learns the best way to rearagnge the data such that each head will learn meaningful information
    """
    def __init__(self, config):
        """
            * Initializes linear layers: `c_atten` for generating **query (Q)**, **key (K)**, and **value (V)** vectors, and `c_proj` for the output projection.
            * Ensures that the `embed_size` is divisible by `num_heads`.
            * Registers a lower triangular `mask` to ensure **causal attention**, preventing attention to future tokens.
        """
        super(SelfAttentionBlock, self).__init__()
        assert config.embed_size % config.num_heads == 0
        
        # c_atten serves two main purposes:
        # 1. It projects the input embedding (size C) into a larger space (size 3*C) to generate the Query, Key, and Value matrices.
        # 2. As a learnable layer, it learns an optimal projection that organizes the information. This ensures that when Q, K, and V
        #    are later reshaped for the different heads, each head receives a meaningful subspace of the information to work with.
        self.c_atten = nn.Linear(config.embed_size, 3*config.embed_size)
        
        # The final projection layer. This learnable layer takes the concatenated outputs
        # from all attention heads and projects them back into the residual stream's dimension.
        # Its role is to learn how to best combine the specialized information captured by each head
        # into a single, coherent output vector.
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        
        self.embed_size = config.embed_size
        self.num_heads = config.num_heads

        mask = torch.tril(torch.ones(config.block_size, config.block_size)) # Lower triangular mask for causal attention
        mask = mask.view(1, 1, config.block_size, config.block_size) # B, 1, T, T
        self.register_buffer('mask', mask)  # register_buffer allows the mask to be part of the model state but not a parameter to optimize
        
    def forward(self, x):
        #  after each line print the shape of x
        """        
            1.  Transforms the input `x` into **Q**, **K**, and **V** tensors.
            2.  Reshapes and transposes **Q**, **K**, and **V** to prepare for multi-head attention, arranging them by `num_heads`.
            3.  Computes **attention scores** by taking the dot product of **Q** and the transpose of **K**, then scales these scores.
            4.  Applies the causal `mask` to the attention scores, setting future token connections to negative infinity.
            5.  Calculates **attention weights** using a **softmax** function on the attention scores.
            6.  Computes the **output** by multiplying the attention weights with **V**.
            7.  Concatenates the outputs from all attention heads.
            8.  Applies a final linear projection (`c_proj`) to produce the block's output.
        Args:
            x: Input tensor of shape (B, T, C) where B is batch size, T is sequence length, C is embedding size
        Returns:
            out: Output tensor of shape (B, T, C) after self-attention and projection
        """
        B, T, C = x.shape # Batch, sequence length, embed_size
        if C != self.embed_size:
            raise ValueError(f"Input embedding size {C} does not match expected embedding size {self.embed_size}")
        
        k, q, v = self.c_atten(x).split(self.embed_size, dim=2) # split the output into k, q, v.  Each is of shape (B, T, embed_size)
        head_size = self.embed_size // self.num_heads

        # reshape k, q, v to (B, T, num_heads, head_size). As embed_size == num_heads * head_size, we can do this by dividing embed_size by num_heads
        k = k.view(B, T, self.num_heads, head_size)
        q = q.view(B, T, self.num_heads, head_size)
        v = v.view(B, T, self.num_heads, head_size)

        # transpose k, q, v to (B, num_heads, T, head_size) for batch multiplication
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention scores
        att_scores = (q @ k.transpose(-2, -1))
        att_scores = att_scores / math.sqrt(head_size) # scale the attention scores
        att_scores = att_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # mask the attention scores
        
        # calculte the attention weights
        att_weights = F.softmax(att_scores, dim=-1) # shape (B, num_heads, T, T)

        # compute the output
        output = att_weights @ v # shape (B, num_heads, T, head_size)

        # concatenate the output of the heads
        concatenated_output = output.transpose(1, 2).contiguous().view(B, T, C) # shape (B, T, C)

        # project the output to the embed_size
        final_output = self.c_proj(concatenated_output)

        return final_output # shape (B, T, embed_size)


