
class CharacterTokenizer:
    def __init__(self, text):
        """
        Initialize tokenizer by building vocabulary from input text.
        
        Args:
            text: String containing all training text
        """
        # Get all unique characters and sort them
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        mapping_list = list(enumerate(self.chars))
        self.char_to_index = {char: index for index, char in mapping_list}
        self.index_to_char = {index: char for index, char in mapping_list}
 

    def encode(self, text):
        """Convert string to list of integers"""
        return [self.char_to_index[char] for char in text]
    
    def decode(self, indices):
        """Convert list of integers back to string"""
        return ''.join([self.index_to_char[index] for index in indices])
    
    def __len__(self):
        """Return the size of the vocabulary"""
        return self.vocab_size