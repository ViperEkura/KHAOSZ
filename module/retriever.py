import torch
import sqlite3
import numpy as np
from torch import Tensor
from typing import List, Tuple


class Retriever:
    def __init__(self, db_path=None):
        self.items: List[str] = []
        self.embeddings: List[Tensor] = []
        
        if db_path is not None:
            self.load(db_path)
        
    def add_vector(self, key: str, vector_data: Tensor):
        self.items.append(key)
        self.embeddings.append(vector_data.flatten())
        
    def delete_vector(self, key: str):
        for i in range(len(self.items)-1, -1, -1):
            if self.items[i] == key:
                self.items.pop(i)
                self.embeddings.pop(i)
                
    def similarity(self, input_tensor: Tensor, top_k: int) -> List[Tuple[str, float]]:
        if len(self.items) == 0:
            return []
        
        top_k_clip = min(top_k, len(self.items))
        segment = torch.cat(self.embeddings, dim=0)
        sim_scores = torch.matmul(segment, input_tensor)
        
        top_k_data = [
            (self.items[i], sim_scores[i].item()) 
            for i in sim_scores.topk(top_k_clip).indices.tolist()
        ]
        
        return top_k_data
    
    def save(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        self.__init_db__(cursor)
        cursor.execute('DELETE FROM vectors')
        
        for item, vec in zip(self.items, self.embeddings):
            vec_bytes = vec.numpy().tobytes()
            cursor.execute('INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)', 
                           (item, vec_bytes))
        
        conn.commit()
        conn.close()

    def load(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        self.__init_db__(cursor)
        cursor.execute('SELECT key, vector FROM vectors')
        rows = cursor.fetchall()
        
        self.items = []
        self.embeddings = []
        
        for row in rows:
            key, vec_bytes = row
            vec_numpy = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            vec = torch.from_numpy(vec_numpy)
            
            self.items.append(key)
            self.embeddings.append(vec)
        
        conn.close()
        
    def __init_db__(self,cursor: sqlite3.Cursor):
        # Create table if not exists (in case loading from a new database)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                vector BLOB NOT NULL
            )''')