from impl.attention import SelfAttentionBlock
from impl.mlp import MLP
from torch import nn

class TransformerBlock(nn.Module):
    """
    The **Transformer Block** represents a single layer of the transformer architecture, 
    combining the **Self-Attention** and **MLP** components. 
    Multiple blocks can be stacked to construct a deep network.

    """
    def __init__(self, config):
        """
            Initializes the **normalization**, **MLP**, and **Self-Attention** components.
        """
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
        self.mlp = MLP(config)
        self.self_attn = SelfAttentionBlock(config)
        
    def forward(self, x):
        """
            1.  Normalizes the input and passes it to the **Self-Attention** layer.
            2.  Applies a **residual connection** after the Self-Attention output.
            3.  Normalizes the residual output and feeds it into the **MLP** layer.
            4.  Applies another **residual connection** after the MLP layer to form the block's final output.

        """
        normalized_attn_input = self.ln1(x)
        attn_output = self.self_attn(normalized_attn_input)
        x = x + attn_output
        normalized_mlp_input = self.ln2(x)
        mlp_output = self.mlp(normalized_mlp_input)
        x = x + mlp_output
        return x