import json
from typing import Dict, List, Any, Optional

class JsonProcessor:
    def __init__(self, db_path: str = None):
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        if db_path is not None:
            self.load_from_file(db_path)
    
    def create_table(self, table_name: str, primary_key: str = "id") -> bool:
        if table_name in self.tables:
            return False
        self.tables[table_name] = []
        return True
    
    def insert(self, table_name: str, data: Dict[str, Any]) -> bool:
        if table_name not in self.tables:
            return False
        self.tables[table_name].append(data)
        return True
    
    def batch_insert(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        if table_name not in self.tables:
            return 0
        self.tables[table_name].extend(data_list)
        return len(data_list)
    
    def select(self, table_name: str, condition: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if table_name not in self.tables:
            return []
        
        if condition is None:
            return self.tables[table_name].copy()
        
        results = []
        for record in self.tables[table_name]:
            match = True
            for key, value in condition.items():
                if key not in record or record[key] != value:
                    match = False
                    break
            if match:
                results.append(record.copy())
        return results
    
    def update(self, table_name: str, condition: Dict[str, Any], new_data: Dict[str, Any]) -> int:

        if table_name not in self.tables:
            return 0
        
        count = 0
        for record in self.tables[table_name]:
            match = True
            for key, value in condition.items():
                if key not in record or record[key] != value:
                    match = False
                    break
            if match:
                record.update(new_data)
                count += 1
        return count
    
    def delete(self, table_name: str, condition: Dict[str, Any]) -> int:
        if table_name not in self.tables:
            return 0
        
        initial_length = len(self.tables[table_name])
        self.tables[table_name] = [
            record for record in self.tables[table_name]
            if not all(key in record and record[key] == value for key, value in condition.items())
        ]
        return initial_length - len(self.tables[table_name])
    
    def load_from_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r') as file:
                self.tables = json.load(file)
            return True
        except (IOError, json.JSONDecodeError):
            return False
    
    def save_to_file(self, file_path: str) -> bool:

        try:
            with open(file_path, 'w') as file:
                json.dump(self.tables, file, indent=4)
            return True
        except IOError:
            return False
    
    def get_table_names(self) -> List[str]:
        return list(self.tables.keys())
    
    def table_size(self, table_name: str) -> int:
        if table_name not in self.tables:
            return -1
        return len(self.tables[table_name])
        


class VectorAssets:
    def __init__(self, data_path):
        self.json_processor = JsonProcessor()
        
        
    def add_vector(self, key, vector_data):
        pass
        
    def delete_vector(self, key):
        pass
    
    
    def simliarity(self, input_info):
        
        pass