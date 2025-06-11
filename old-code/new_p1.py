import os
import getpass
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta
import sqlite3  # Added for SQLite fallback
import psycopg2
from urllib.parse import urlparse

import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
SETUP INSTRUCTIONS:

1. Install required packages:
   pip install langchain-google-genai pandas psycopg2-binary google-generativeai

2. For vector database (choose one option):
   
   OPTION A - PostgreSQL with pgvector (Recommended):
   - Install PostgreSQL: https://www.postgresql.org/download/
   - Install pgvector extension: https://github.com/pgvector/pgvector
   - Create database: createdb your_db_name
   - Set DATABASE_URL: export DATABASE_URL="postgresql://username:password@localhost:5432/your_db_name"
   
   OPTION B - SQLite (Simple, no setup required):
   - Uses sqlite-vss extension (will be handled automatically)
   - No additional setup needed
   
   OPTION C - Docker PostgreSQL with pgvector:
   docker run --name postgres-vector -e POSTGRES_PASSWORD=password -e POSTGRES_DB=vectordb -p 5432:5432 -d ankane/pgvector
   export DATABASE_URL="postgresql://postgres:password@localhost:5432/vectordb"

3. Set environment variables:
   export GOOGLE_API_KEY="your_gemini_api_key"
   export DATABASE_URL="postgresql://username:password@localhost:5432/dbname"  # optional
"""

# Set your Google API key (replace with your actual key or use environment variable)
if "GOOGLE_API_KEY" not in os.environ:
    # SECURITY WARNING: Don't hardcode API keys in production!
    # Use environment variables or secure credential management
    api_key = input("Enter your Google API Key (or set GOOGLE_API_KEY env var): ").strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        # Fallback to hardcoded key (remove in production)
        os.environ["GOOGLE_API_KEY"] = "AIzaSyACjg7EaWTAB3-lXMmbmK-3sIbuY08erKQ"

# Set your LangSmith key (optional, for tracing)
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"

def setup_database_url():
    """Interactive setup for database URL if not set"""
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    
    print("\n🔧 Database Setup Required")
    print("Choose your database option:")
    print("1. PostgreSQL with vector_user (recommended)")
    print("2. PostgreSQL with custom credentials")
    print("3. SQLite (simple, no setup required)")
    print("4. Skip database tests")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Use predefined vector_user credentials
        print("\nUsing vector_user credentials...")
        host = input("Host (default: 127.0.0.1): ").strip() or "127.0.0.1"
        port = input("Port (default: 5432): ").strip() or "5432"
        database = input("Database name (default: vectordb): ").strip() or "vectordb"
        username = "vector_user"
        password = "SecurePassword123!"
        
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        os.environ["DATABASE_URL"] = database_url
        print(f"✅ Using: postgresql://vector_user:***@{host}:{port}/{database}")
        return database_url
    
    elif choice == "2":
        print("\nCustom PostgreSQL Setup:")
        host = input("Host (default: 127.0.0.1): ").strip() or "127.0.0.1"
        port = input("Port (default: 5432): ").strip() or "5432"
        database = input("Database name (default: vectordb): ").strip() or "vectordb"
        username = input("Username (default: postgres): ").strip() or "postgres"
        password = getpass.getpass("Password: ")
        
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        os.environ["DATABASE_URL"] = database_url
        return database_url
    
    elif choice == "3":
        # Use SQLite as fallback
        database_url = "sqlite:///vector_store.db"
        os.environ["DATABASE_URL"] = database_url
        return database_url
    
    else:
        return None

def test_basic_gemini():
    """Test basic Gemini functionality"""
    print("🔄 Testing basic Gemini functionality...")
    
    try:
        # Create the ChatGoogleGenerativeAI instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Make a test call
        response = llm.invoke("Explain few-shot learning in one sentence.")
        print(f"✅ Gemini response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def process_bank_transactions():
    """Process bank transactions with improved error handling"""
    print("\n🔄 Processing bank transactions...")
    
    try:
        # Check if file exists
        filename = "bank_transactions_with_vendor_100.xlsx"
        if not os.path.exists(filename):
            print(f"⚠️  File {filename} not found. Skipping transaction processing.")
            return
            
        df = pd.read_excel(filename)
        print(f"✅ Loaded {len(df)} transactions")
        print(f"Columns: {list(df.columns)}")
        
        # Create subset with required columns
        required_columns = ['Vendor', 'Description', 'Amount', 'Transaction Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return
            
        new_df = pd.DataFrame(df, columns=required_columns)
        print(f"✅ Processing {len(new_df)} transactions")
        
        # Create LLM instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_retries=2,
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial transaction categorization expert.
            Categorize transactions based on their description and vendor information.
            
            Categories include:
            - Education (school, college, university fees)
            - Health (medical, operations, surgery)
            - Groceries (food, supermarket)
            - Transportation (fuel, taxi, public transport)
            - Utilities (electricity, water, internet)
            - Entertainment (movies, restaurants, recreation)
            - Shopping (retail, clothing, electronics)
            - Other
            
            Respond with valid JSON only:
            {
                "category": "string",
                "type": "string", 
                "confidence": float,
                "explanation": "string"
            }"""),
            ("human", """Categorize this transaction:
            Description: {description}
            Vendor: {vendor}
            Transaction Type: {transaction_type}
            Amount: {amount}""")
        ])
        
        chain = prompt | llm
        
        # Process transactions with rate limiting
        results = []
        for i, row in new_df.iterrows():
            if i >= 5:  # Limit to first 5 for demo
                break
                
            input_data = {
                "description": str(row["Description"]),
                "vendor": str(row["Vendor"]),
                "transaction_type": str(row["Transaction Type"]),
                "amount": str(row["Amount"])
            }
            
            try:
                print(f"🔄 Processing transaction {i+1}: {row['Description'][:30]}...")
                result = chain.invoke(input_data)
                
                # Try to parse JSON response
                try:
                    parsed_result = json.loads(result.content)
                    results.append(parsed_result)
                    print(f"✅ Category: {parsed_result.get('category', 'Unknown')}")
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON response: {result.content[:100]}...")
                    results.append({"category": "Unknown", "error": "Invalid JSON"})
                    
            except Exception as e:
                print(f"❌ Error processing transaction {i}: {e}")
                results.append({"category": "Error", "error": str(e)})
            
            # Rate limiting
            time.sleep(2)
        
        print(f"✅ Processed {len(results)} transactions")
        return results
        
    except Exception as e:
        print(f"❌ Transaction processing failed: {e}")
        return None

class SimpleVectorStore:
    """Simplified vector store using only Gemini embeddings (no external DB required)"""
    
    def __init__(self):
        """Initialize with Gemini embeddings only"""
        self.embedding_model = "models/embedding-001"
        self.vectors = []  # Store vectors in memory for this demo
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        print("✅ Simple VectorStore initialized with Gemini embeddings")
    
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
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to in-memory vector store"""
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        doc = {
            "id": len(self.vectors),
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        
        self.vectors.append(doc)
        print(f"✅ Added document {doc['id']}: '{text[:50]}...'")
        return True
    
    def search_similar(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        if not self.vectors:
            print("⚠️  No documents in vector store")
            return []
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for doc in self.vectors:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((similarity, doc))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [{"similarity": sim, **doc} for sim, doc in similarities[:limit]]
        
        print(f"✅ Found {len(results)} similar documents")
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0
    
    def test_functionality(self):
        """Test the vector store functionality"""
        print("\n🔬 Testing SimpleVectorStore...")
        
        # Add some test documents
        test_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Vector databases store and search high-dimensional vectors.",
            "Natural language processing helps computers understand human language."
        ]
        
        print("1. Adding test documents...")
        for i, doc in enumerate(test_docs):
            success = self.add_document(doc, {"source": "test", "index": i})
            if not success:
                print(f"❌ Failed to add document {i}")
                return False
        
        print("2. Testing similarity search...")
        query = "What is AI and machine learning?"
        results = self.search_similar(query, limit=2)
        
        if results:
            print("✅ Search results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"      Text: {result['text'][:60]}...")
        else:
            print("❌ No search results")
            return False
        
        print("✅ SimpleVectorStore test completed successfully!")
        return True

class SQLiteVectorStore:
    """SQLite-based vector store for simple setup"""
    
    def __init__(self, db_path: str = "vector_store.db"):
        """Initialize SQLite vector store"""
        self.db_path = db_path
        self.embedding_model = "models/embedding-001"
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # Initialize database
        self._init_database()
        print(f"✅ SQLite VectorStore initialized at {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding TEXT,  -- Store as JSON string
                metadata TEXT,   -- Store as JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
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
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to SQLite database"""
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO documents (content, embedding, metadata)
                VALUES (?, ?, ?)
            """, (text, json.dumps(embedding), json.dumps(metadata or {})))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Document added: '{text[:50]}...'")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents
            cursor.execute("SELECT id, content, embedding, metadata FROM documents")
            documents = cursor.fetchall()
            
            if not documents:
                print("⚠️  No documents in database")
                return []
            
            # Calculate similarities
            similarities = []
            for doc_id, content, embedding_str, metadata_str in documents:
                try:
                    doc_embedding = json.loads(embedding_str)
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    similarities.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": json.loads(metadata_str),
                        "similarity": similarity
                    })
                except Exception as e:
                    print(f"⚠️  Error processing document {doc_id}: {e}")
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:limit]
            
            conn.close()
            print(f"✅ Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0
    
    def test_functionality(self):
        """Test the SQLite vector store functionality"""
        print("\n🔬 Testing SQLiteVectorStore...")
        
        # Add some test documents
        test_docs = [
            "Artificial intelligence is transforming industries.",
            "Machine learning algorithms learn from data patterns.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
        
        print("1. Adding test documents...")
        for i, doc in enumerate(test_docs):
            success = self.add_document(doc, {"source": "test", "index": i})
            if not success:
                print(f"❌ Failed to add document {i}")
                return False
        
        print("2. Testing similarity search...")
        query = "What is AI and machine learning?"
        results = self.search_similar(query, limit=2)
        
        if results:
            print("✅ Search results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"      Text: {result['content'][:60]}...")
        else:
            print("❌ No search results")
            return False
        
        print("✅ SQLiteVectorStore test completed successfully!")
        return True

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
    
    def create_table(self, table_name: str = "documents") -> bool:
        """Create vector table using standard PostgreSQL + vector extension"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Create table with vector column
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            cursor.execute(create_table_sql)
            
            # Create index for vector similarity search
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
            ON {table_name} USING ivfflat (embedding vector_cosine_ops);
            """
            
            cursor.execute(create_index_sql)
            conn.commit()
            
            print(f"✅ Table '{table_name}' created successfully")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to create table: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def insert_document(self, text: str, metadata: Dict[str, Any] = None, table_name: str = "documents") -> bool:
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
    
    def search_similar(self, query: str, limit: int = 3, table_name: str = "documents") -> List[Dict]:
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

def main():
    """Main function to demonstrate functionality"""
    print("🚀 Starting GenAI Vector Store Demo...")
    
    # Test basic Gemini functionality
    if not test_basic_gemini():
        print("❌ Basic Gemini test failed. Check your API key.")
        return
    
    # Process bank transactions (if file exists)
    transaction_results = process_bank_transactions()
    
    # Test simple vector store (always works)
    print("\n" + "="*50)
    print("TESTING SIMPLE VECTOR STORE (In-Memory)")
    print("="*50)
    
    try:
        simple_store = SimpleVectorStore()
        simple_store.test_functionality()
    except Exception as e:
        print(f"❌ SimpleVectorStore failed: {e}")
    
    # Setup database URL interactively if not set
    database_url = setup_database_url()
    
    if database_url and database_url.startswith("sqlite"):
        # Test SQLite vector store
        print("\n" + "="*50)
        print("TESTING SQLITE VECTOR STORE")
        print("="*50)
        
        try:
            sqlite_store = SQLiteVectorStore()
            sqlite_store.test_functionality()
        except Exception as e:
            print(f"❌ SQLiteVectorStore failed: {e}")
    
    elif database_url and database_url.startswith("postgresql"):
        # Test PostgreSQL vector store
        print("\n" + "="*50)
        print("TESTING POSTGRESQL VECTOR STORE")
        print("="*50)
        
        try:
            db_store = DatabaseVectorStore(database_url)
            
            # Create table
            if db_store.create_table():
                # Add test documents
                test_docs = [
                    "Artificial intelligence is transforming industries.",
                    "Machine learning algorithms learn from data patterns.",
                    "Deep learning uses neural networks with multiple layers."
                ]
                
                for doc in test_docs:
                    db_store.insert_document(doc, {"source": "demo"})
                
                # Search
                results = db_store.search_similar("What is AI?", limit=2)
                if results:
                    print("✅ Database vector store test successful!")
                    for result in results:
                        print(f"   Similarity: {result['similarity']:.3f} - {result['content'][:50]}...")
                
        except Exception as e:
            print(f"❌ DatabaseVectorStore failed: {e}")
            print("💡 Try using SQLite option or check PostgreSQL setup")
    
    else:
        print("⚠️  Database tests skipped")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()