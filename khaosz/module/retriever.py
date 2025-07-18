import re
import torch
import sqlite3
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Dict, List, Tuple


class Retriever:
    def __init__(self, db_path=None):
        self.data: Dict[str, Tensor] = {}
        self.embedding_cache: Tensor = None
        self.is_caculated: bool = False
        
        if db_path is not None:
            self.load(db_path)
                
    def retrieve(self, query: Tensor, top_k: int) -> List[Tuple[str, float]]:
        if not self.data:
            return []
        
        query = query.flatten().unsqueeze(1)                            # [dim, 1]
        norm_embeddings = self._embeddings.to(
            device=query.device,
            dtype=query.dtype
        )                                                               # [n_vectors, dim]
        sim_scores = torch.matmul(norm_embeddings, query).squeeze()     # [n_vectors]
        
        top_k = min(top_k, len(self.data))
        indices = sim_scores.topk(top_k).indices
        keys = list(self.data.keys())
        
        return [(keys[i], sim_scores[i].item()) for i in indices]
    
    def add_vector(self, key: str, vector_data: Tensor):
        self.is_caculated = False
        self.data[key] = vector_data.flatten().float().cpu()
        
    def delete_vector(self, key: str):
        self.is_caculated = False
        self.data.pop(key, None)
    
    def save(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        self._init_db(cursor)
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
        self._init_db(cursor)
        cursor.execute('SELECT key, vector FROM vectors')
        rows = cursor.fetchall()
        self.data = {}
        
        for row in rows:
            key, vec_bytes = row
            vec_numpy = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            vec = torch.from_numpy(vec_numpy)
            self.data[key] = vec
        
        conn.close()
        
    def _init_db(self,cursor: sqlite3.Cursor):
        # Create table if not exists (in case loading from a new database)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                vector BLOB NOT NULL
            )''')
    
    @property
    def _embeddings(self) -> Tensor:
        if not self.is_caculated:
            embeddings = torch.stack(list(self.data.values())) 
            norm_embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
            self.embedding_cache = norm_embeddings
        
        return self.embedding_cache
        
        
class TextSplitter:
    def __init__(
        self, 
        emb_func: Callable[[List[str]], List[Tensor]],
        pattern: str = None
    ):
        base_pattern =  r'(?<=[。！？!?])(?=(?:[^"\'‘’“”]*["\'‘’“”][^"\'‘’“”]*["\'‘’“”])*[^"\'‘’“”]*$)'
        self.emb_func = emb_func
        self.pattern = pattern
        if pattern is None:
            self.pattern = base_pattern
    
    def chunk(self, text: str, threshold: float = 0.5, window_size: int = 1) -> List[str]:
        sentences = [s.strip() for s in re.split(self.pattern, text.strip()) if s.strip()]
        
        if len(sentences) <= 1:
            return sentences
        
        chunks = []
        sentence_embs = self.emb_func(sentences)     
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size)
            
            prev_window = torch.mean(torch.stack(sentence_embs[start:i]), dim=0)
            next_window = torch.mean(torch.stack(sentence_embs[i:end]), dim=0)
            window_sim = F.cosine_similarity(prev_window,next_window, dim=0).item()
            dynamic_threshold = max(threshold * (1 - 0.03 * (end - start)), 0.2)
            
            if window_sim >= dynamic_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks