import os
import json
import google.generativeai as genai
import psycopg2
from typing import List, Dict, Any

class DatabaseVectorStore:
    """Vector store with PostgreSQL database (requires pgvector extension)"""
    
    def __init__(self, database_url: str = None):
        """Initialize with database connection"""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.embedding_model = "models/embedding-001"
        self.vector_dimension = 768  # Changed from 1536 to match Gemini's output
        
        if not self.database_url:
            raise ValueError("Database URL not provided. Set DATABASE_URL environment variable or pass it to constructor.")
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # Test database connection and create tables
        if not self._test_connection():
            raise ConnectionError("Failed to connect to database")
            
        # Create tables with correct dimension
        self._create_vector_tables()
        
        print("✅ DatabaseVectorStore initialized")
    
    def _test_connection(self) -> bool:
        """Test database connection and check for required extensions"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Check PostgreSQL version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to: {version[:50]}...")
            
            # Enable vector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            print("✅ Vector extension enabled")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def _create_vector_tables(self) -> bool:
        """Create all required vector tables"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Drop existing tables if they exist
            cursor.execute("DROP TABLE IF EXISTS temp_transactions;")
            cursor.execute("DROP TABLE IF EXISTS master_vector;")
            cursor.execute("DROP TABLE IF EXISTS search_history;")
            
            # Create temp_transactions table with correct dimension
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS temp_transactions (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({self.vector_dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Create master_vector table with correct dimension
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS master_vector (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({self.vector_dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Create search_history table with correct dimension
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS search_history (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({self.vector_dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            conn.commit()
            print("✅ All vector tables created successfully")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        if not text or not text.strip():
            return []
        
        text = text.replace("\n", " ").strip()
        
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return []
    
    def insert_document(self, text: str, metadata: Dict[str, Any] = None, table_name: str = "temp_transactions") -> bool:
        """Insert document with vector embedding"""
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            insert_sql = f"""
            INSERT INTO {table_name} (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            """
            
            # Convert embedding list to string format for vector type
            embedding_str = str(embedding)
            
            cursor.execute(insert_sql, (text, embedding_str, json.dumps(metadata or {})))
            conn.commit()
            
            print(f"✅ Document inserted: '{text[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert document: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def search_similar(self, query: str, limit: int = 3, table_name: str = "temp_transactions") -> List[Dict]:
        """Search for similar documents using proper vector casting"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Convert query embedding to string format for vector type
            query_embedding_str = str(query_embedding)
            
            search_sql = f"""
            SELECT id, content, metadata, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            cursor.execute(search_sql, (query_embedding_str, query_embedding_str, limit))
            results = cursor.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "similarity": float(row[3])
                })
            
            print(f"✅ Found {len(formatted_results)} similar documents")
            cursor.close()
            conn.close()
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            if conn:
                cursor.close()
                conn.close()
            return [] 