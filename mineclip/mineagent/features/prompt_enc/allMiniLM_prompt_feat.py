"""
Note that prompt feature is provided by MineCLIP.
"""
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class PromptEmbFeat(nn.Module):
    def __init__(self, device):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed(self, x):
        return self.model.encode(x) # Returns 384 dimensional embedding
