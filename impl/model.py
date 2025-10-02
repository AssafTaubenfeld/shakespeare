import torch
from torch import nn
import torch.nn.functional as F
from impl.blocks import TransformerBlock

class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # **Token embeddings** Maps each vocabulary word to a learned vector representation.
            # Transforms token IDs into dense embeddings of size `embed_size`.
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            # **Positional embeddings** Adds positional information since attention doesn't inherently understand word order. 
            # Each position gets its own learned embedding.
            wpe = nn.Embedding(config.block_size, config.embed_size), # positional embeddings
            # ModuleList ensures that all the nn.Module elements are properly registrated to pyTourch for param tracking (and optimization)
            blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)]), # transformer blocks
            # Normalize the final output after all transformer blocks
            ln_f = nn.LayerNorm(config.embed_size)
            )
        )

        # language model head: final output layer, that project to the vocabulay size. 
        # hold the logits for the model's prediction for next token in the sequence
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

        # Shares weights between the token embeddings (`wte`) and the output layer (`lm_head`).
        # This ties the input and output representations of tokens.
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, target=None):
        """ Forward pass of the GPT-2 model.
        Args:            
            idx: Input tensor of shape (B, T) where B is batch size and T is sequence length
            target: Optional target tensor of shape (B, T) for computing loss
        Returns:            
            logits: Output tensor of shape (B, T, vocab_size) containing logits for next token
            loss: Optional scalar loss value if target is provided
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Input tokens length {T} exceeds maximum length {self.config.block_size}"
        
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(torch.arange(T, device=idx.device))
        x = token_embeddings + position_embeddings
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        else:
            loss = None
    
        return logits, loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
            