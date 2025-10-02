from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Shakespeare text data.
    Generates overlapping sequences for language modeling.
    """
    
    def __init__(self, data, block_size):
        """
        Args:
            data: Tensor of encoded text tokens
            block_size: Length of each sequence (context length)
        """
        self.data = data
        self.block_size = block_size
        
        # Calculate number of possible sequences
        self.num_sequences = len(data) - block_size
        
        if self.num_sequences <= 0:
            raise ValueError(f"Data length ({len(data)}) must be greater than block_size ({block_size})")
    
    def __len__(self):
        """
            Returns the total number of sequences, `num_sequences`. 
            This method allows PyTorch's `DataLoader` to know the dataset's size for iteration.
        """
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
            Retrieves a specific sequence from the dataset. Given an `idx`, it returns the sequence starting at that index. 
            The `DataLoader` ensures that `idx` will always be within valid bounds, preventing out-of-range errors
        
        Args:
            idx: Index of the sequence start position
            
        Returns:
            tuple: (input_sequence, target_sequence)
        """
        # Extract input sequence of length block_size starting at idx
        input_sequence = self.data[idx:idx + self.block_size]
        
        # Extract target sequence (shifted by 1 position)
        target_sequence = self.data[idx + 1:idx + self.block_size + 1]
        
        return input_sequence, target_sequence