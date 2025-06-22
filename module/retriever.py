import re
import torch
import sqlite3
import numpy as np
from torch import Tensor
from typing import Callable, Dict, List, Tuple


class Retriever:
    def __init__(self, db_path=None):
        self.data: Dict[str, Tensor] = {} 
        
        if db_path is not None:
            self.load(db_path)
        
    def add_vector(self, key: str, vector_data: Tensor):
        self.data[key] = vector_data.flatten().float()
        
    def delete_vector(self, key: str):
        self.data.pop(key, None)
                
    def similarity(self, query: Tensor, top_k: int) -> List[Tuple[str, float]]:
        if not self.data:
            return []
        
        query = query.flatten().unsqueeze(1)  # [dim, 1]
        embeddings = torch.stack(list(self.data.values()))  # [n_vectors, dim]
        sim_scores = torch.matmul(embeddings, query).squeeze() # [n_vectors]
        
        top_k = min(top_k, len(self.data))
        indices = sim_scores.topk(top_k).indices
        keys = list(self.data.keys())
        
        return [(keys[i], sim_scores[i].item()) for i in indices]
    
    def save(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        self.__init_db__(cursor)
        cursor.execute('DELETE FROM vectors')
        
        for item, vec in self.data.items():
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
        self.data = {}
        
        for row in rows:
            key, vec_bytes = row
            vec_numpy = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            vec = torch.from_numpy(vec_numpy)
            self.data[key] = vec
        
        conn.close()
        
    def __init_db__(self,cursor: sqlite3.Cursor):
        # Create table if not exists (in case loading from a new database)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                vector BLOB NOT NULL
            )''')
        
        
class TextSplitter:
    def __init__(self, emb_func: Callable[[str], Tensor]):
        self.emb_func = emb_func
    
    def chunk(self, text: str, threshold: int = 0.5) -> List[str]:
        pattern = r'(?<=[。！？!?]|\.(?!\w))\s*'
        sentences = re.split(pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_embs = [self.emb_func(s) for s in sentences]

        chunks = [sentences[0]]
        current_emb = sentence_embs[0]

        for i in range(1, len(sentences)):
            cos_sim = torch.dot(current_emb, sentence_embs[i])
            
            if cos_sim.item() >= threshold:
                chunks[-1] += " " + sentences[i]
                current_emb = (current_emb + sentence_embs[i]) / 2
                current_emb = current_emb / current_emb.norm()
            else:
                chunks.append(sentences[i])
                current_emb = sentence_embs[i]

        return chunks