import os
import time
import json
import google.generativeai as genai
import psycopg2
from typing import List, Dict, Any, Union
import pandas as pd

def safe_api_call(func, max_retries=3, delay=1):
    """Execute API calls with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
                return None
            print(f"⚠️ Attempt {attempt + 1} failed, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff 

def process_csv_file(db_store, file_path: str) -> bool:
    """Process a CSV file and insert records into the database"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"✅ Read {len(df)} records from {file_path}")
        
        # Process each row
        for _, row in df.iterrows():
            # Convert row to string format for embedding
            row_dict = dict(row)
            transaction_str = f"Transaction details: {json.dumps(row_dict)}"
            
            # Insert into database
            success = db_store.insert_temp_transaction(transaction_str, {
                "source": "csv_import",
                "original_data": row_dict
            })
            
            if not success:
                print(f"⚠️ Failed to insert transaction: {transaction_str[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process CSV file: {e}")
        return False

class DatabaseVectorStore:
    """Vector store with PostgreSQL database (requires pgvector extension)"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.embedding_dimensions = 1536  # Gemini embedding dimensions
        
    def insert_temp_transaction(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Insert document into temp_transactions table"""
        # Convert dictionary to string if content is a dictionary
        if isinstance(content, dict):
            content = json.dumps(content)
            
        embedding = self.get_embedding(content)
        if not embedding or len(embedding) != self.embedding_dimensions:
            print(f"❌ Invalid embedding: expected {self.embedding_dimensions} dimensions, got {len(embedding) if embedding else 0}")
            return False
        
        # Normalize the embedding before insertion
        embedding = self._normalize_embedding(embedding)
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            insert_sql = """
            INSERT INTO temp_transactions (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            """
            
            cursor.execute(insert_sql, (content, str(embedding), json.dumps(metadata or {})))
            conn.commit()
            
            print(f"✅ Temp transaction inserted: '{str(content)[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert temp transaction: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False
            
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini API"""
        try:
            # Your embedding generation code here
            # This is a placeholder - implement actual embedding generation
            import numpy as np
            return list(np.random.rand(self.embedding_dimensions))
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return []
            
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector"""
        try:
            import numpy as np
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm == 0:
                return embedding
            return (embedding_array / norm).tolist()
        except Exception as e:
            print(f"❌ Failed to normalize embedding: {e}")
            return embedding 