import os
import torch

from .transfomer import Config, Transfomer
from .tokenizer import BpeTokenizer

class Message:
    def __init__(self):
        self.history = []
        

class Khaosz:
    def __init__(self, path=None):
        self.tokenizer : BpeTokenizer = None
        self.config : Config = None
        self.model : Transfomer = None
        
        if path is not None:
            self.load(path)
            
    def load(self, model_dir):
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "model.pt")
        
        self.tokenizer = BpeTokenizer(tokenizer_path)
        self.config = Config(config_path)
        self.model = Transfomer(self.config)
        self.model.load_state_dict(torch.load(weight_path))
    
    def generate():
        pass