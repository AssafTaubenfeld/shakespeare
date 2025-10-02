import torch
from torch.utils.data import DataLoader
from .dataset import ShakespeareDataset

class ShakespeareDataModule:
    """Handles data loading, splitting, and DataLoader creation"""
    def __init__(self, text, tokenizer, block_size, train_ratio=0.9):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        self.train_ratio = train_ratio
        
    def setup(self):
        n = int(self.train_ratio * len(self.data))
        train_data = self.data[:n]
        val_data = self.data[n:]
        
        self.train_dataset = ShakespeareDataset(train_data, self.block_size)
        self.val_dataset = ShakespeareDataset(val_data, self.block_size)
    
    def train_dataloader(self, batch_size, num_workers):
        return DataLoader(self.train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    def val_dataloader(self, batch_size, num_workers):
        return DataLoader(self.val_dataset, batch_size, shuffle=False, num_workers=num_workers)