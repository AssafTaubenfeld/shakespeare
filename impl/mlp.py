from torch import nn
class MLP(nn.Module):
    """
        The **MLP** is the feed-forward component within the transformer block. 
        It receives the output from the **Self-Attention** component and processes
        it through two **Linear** layers with **GeLU** activation between them.
    """
    def __init__(self, config):
        """
            Initializes the **Linear** layers and the **activation function** of the MLP component, based on the **embedding size**.
            The MLp layer streach the dimention of the embedding from `embed_size` to `4*embed_size` and then back to `embed_size`.
        """
        super(MLP, self).__init__()
        self.embed_size = config.embed_size
        self.layer_one = nn.Linear(self.embed_size, 4 * self.embed_size)
        self.layer_two = nn.Linear(4 * self.embed_size, self.embed_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """
            Passes the input data through the initialized layers and returns the result, 
            maintaining the **same dimension** as the input.
        """
        x = self.layer_one(x)
        x = self.gelu(x)
        x = self.layer_two(x)
        return x