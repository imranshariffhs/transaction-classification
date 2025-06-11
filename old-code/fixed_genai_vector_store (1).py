import os
import json
import time
import pandas as pd
import psycopg2
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def safe_api_call(func, max_retries=3, delay=1):
    """Safely call API with retries and rate limiting"""
    for attempt in range(max_retries):
        try:
            time.sleep(delay)  # Rate limiting
            return func()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = delay * (2 ** attempt)
                print(f"⚠️  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif attempt == max_retries - 1:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(delay)
    return None

class DatabaseVectorStore:
    """Vector store with PostgreSQL database (requires pgvector extension)"""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize with database connection"""
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db")
        self.embedding_model = "models/embedding-001"
        self.embedding_dimensions = 1536  # Required by your schema
        self.model_dimensions = 768  # Actual Gemini model dimensions
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # Test database connection
        if not self._test_connection():
            raise ConnectionError("Failed to connect to database")
        
        print("✅ DatabaseVectorStore initialized")
    
    def get_embedding(self, text: Union[str, dict]) -> List[float]:
        """Generate embedding using Gemini with rate limiting and dimension padding"""
        # Handle different input types
        if isinstance(text, dict):
            # Convert dict to string representation
            text_content = str(text)
        elif isinstance(text, str):
            text_content = text
        else:
            print(f"❌ Unsupported input type: {type(text)}")
            return []
        
        if not text_content or not text_content.strip():
            return []
        
        text_content = text_content.replace("\n", " ").strip()
        
        def _generate_embedding():
            return genai.embed_content(
                model=self.embedding_model,
                content=text_content,
                task_type="retrieval_document"
            )
        
        response = safe_api_call(_generate_embedding)
        if response:
            base_embedding = response["embedding"]
            
            # Pad the embedding to match required dimensions
            if len(base_embedding) < self.embedding_dimensions:
                # Calculate how many times to repeat the base embedding
                repeat_times = self.embedding_dimensions // len(base_embedding)
                remainder = self.embedding_dimensions % len(base_embedding)
                
                # Create padded embedding
                padded_embedding = base_embedding * repeat_times
                if remainder:
                    padded_embedding.extend(base_embedding[:remainder])
                
                print(f"✅ Padded embedding from {len(base_embedding)} to {len(padded_embedding)} dimensions")
                return padded_embedding
            
            return base_embedding
        else:
            print(f"❌ Embedding generation failed for text: {str(text_content)[:50]}...")
            return []

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        import math
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            return [x / magnitude for x in embedding]
        return embedding

    def _test_connection(self) -> bool:
        """Test database connection and check for required tables"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Check PostgreSQL version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to: {version[:50]}...")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def create_table(self) -> bool:
        """Create required tables if they don't exist"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Enable vector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create temp_transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temp_transactions (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create master_vector table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS master_vector (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create search_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS temp_transactions_embedding_idx 
                ON temp_transactions USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS master_vector_embedding_idx 
                ON master_vector USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS search_history_embedding_idx 
                ON search_history USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("✅ All tables and indexes created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False

    def add_document(self, text: Union[str, dict], metadata: Optional[Dict[str, Any]] = None, table_name: str = "temp_transactions") -> bool:
        """Add document to specified table (convenience method)"""
        if table_name == "temp_transactions":
            return self.insert_temp_transaction(text, metadata)
        elif table_name == "master_vector":
            return self.insert_master_vector(text, metadata)
        elif table_name == "search_history":
            return self.insert_search_history(text, metadata)
        else:
            print(f"❌ Invalid table name: {table_name}")
            return False
    
    def insert_temp_transaction(self, content: Union[str, dict], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Insert document into temp_transactions table - now handles both strings and dicts"""
        # Convert content to string if it's a dict
        if isinstance(content, dict):
            # Create a meaningful string representation
            content_str = f"Transaction: {content.get('Description', 'N/A')} at {content.get('Vendor', 'N/A')} for ${content.get('Amount', 'N/A')}"
            # Add the original dict to metadata
            if metadata is None:
                metadata = {}
            metadata.update(content)
        else:
            content_str = str(content)
        
        embedding = self.get_embedding(content_str)
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
            
            cursor.execute(insert_sql, (content_str, str(embedding), json.dumps(metadata or {})))
            conn.commit()
            
            print(f"✅ Temp transaction inserted: '{content_str[:50]}...'")
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
    
    def insert_master_vector(self, content: Union[str, dict], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Insert document into master_vector table"""
        # Convert content to string if it's a dict
        if isinstance(content, dict):
            content_str = f"Transaction: {content.get('Description', 'N/A')} at {content.get('Vendor', 'N/A')} for ${content.get('Amount', 'N/A')}"
            if metadata is None:
                metadata = {}
            metadata.update(content)
        else:
            content_str = str(content)
            
        embedding = self.get_embedding(content_str)
        if not embedding or len(embedding) != self.embedding_dimensions:
            print(f"❌ Invalid embedding: expected {self.embedding_dimensions} dimensions, got {len(embedding) if embedding else 0}")
            return False
            
        # Normalize the embedding before insertion
        embedding = self._normalize_embedding(embedding)
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            insert_sql = """
            INSERT INTO master_vector (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            """
            
            cursor.execute(insert_sql, (content_str, str(embedding), json.dumps(metadata or {})))
            conn.commit()
            
            print(f"✅ Master vector inserted: '{content_str[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert master vector: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def insert_search_history(self, query: Union[str, dict], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Insert search query into search_history table"""
        # Convert query to string if it's a dict
        if isinstance(query, dict):
            query_str = str(query)
            if metadata is None:
                metadata = {}
            metadata.update(query)
        else:
            query_str = str(query)
            
        embedding = self.get_embedding(query_str)
        if not embedding or len(embedding) != self.embedding_dimensions:
            print(f"❌ Invalid embedding: expected {self.embedding_dimensions} dimensions, got {len(embedding) if embedding else 0}")
            return False
            
        # Normalize the embedding before insertion
        embedding = self._normalize_embedding(embedding)
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Add timestamp to metadata if not present
            if metadata is None:
                metadata = {}
            if 'query_time' not in metadata:
                metadata['query_time'] = datetime.now().isoformat()
            
            insert_sql = """
            INSERT INTO search_history (content, embedding, metadata)
            VALUES (%s, %s::vector, %s)
            """
            
            cursor.execute(insert_sql, (query_str, str(embedding), json.dumps(metadata)))
            conn.commit()
            
            print(f"✅ Search history inserted: '{query_str[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert search history: {e}")
            if 'conn' in locals():
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def search_similar(self, query: str, table_name: str = "temp_transactions", limit: int = 3) -> List[Dict]:
        """Search for similar documents in specified table"""
        if table_name not in ['master_vector', 'search_history', 'temp_transactions']:
            print(f"❌ Invalid table name: {table_name}")
            return []
            
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize_embedding(query_embedding)
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            search_sql = f"""
            SELECT content, metadata, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            cursor.execute(search_sql, (str(query_embedding), str(query_embedding), limit))
            results = cursor.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "content": row[0],
                    "metadata": row[1],
                    "similarity": float(row[2])
                })
            
            print(f"✅ Found {len(formatted_results)} similar documents in {table_name}")
            cursor.close()
            conn.close()
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            if 'conn' in locals():
                cursor.close()
                conn.close()
            return []

    def process_file(self, file_path: str, table_name: str = "temp_transactions") -> bool:
        """Process and insert data from a CSV file"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} records from {file_path}")
            
            # Convert to dictionary records
            records = df.to_dict(orient='records')
            
            success_count = 0
            for i, record in enumerate(records):
                print(f"🔄 Processing record {i+1}/{len(records)}...")
                
                if self.add_document(record, {"source": file_path, "row_id": i}, table_name):
                    success_count += 1
                
                # Rate limiting
                time.sleep(0.5)
            
            print(f"✅ Successfully processed {success_count}/{len(records)} records")
            return success_count == len(records)
            
        except Exception as e:
            print(f"❌ Failed to process file {file_path}: {e}")
            return False
    
    def test_functionality(self):
        """Test the complete functionality of DatabaseVectorStore"""
        print("\n🔬 Testing DatabaseVectorStore functionality...")
        
        # Test 1: Create tables
        print("1. Creating tables...")
        if not self.create_table():
            print("❌ Table creation failed")
            return False
            
        # Test 2: Insert test documents
        print("2. Testing document insertion...")
        test_docs = [
            "Machine learning algorithms learn from data patterns to make predictions.",
            "Python is widely used for data science and artificial intelligence projects.",
            "Vector databases efficiently store and search high-dimensional embeddings.",
            "Natural language processing enables computers to understand human language."
        ]
        
        for i, doc in enumerate(test_docs):
            success = self.add_document(doc, {"source": "test", "index": i}, "temp_transactions")
            if not success:
                print(f"❌ Failed to insert document {i}")
                return False
        
        # Test 3: Search functionality
        print("3. Testing similarity search...")
        query = "What is machine learning and AI?"
        results = self.search_similar(query, "temp_transactions", limit=2)
        
        if results:
            print("✅ Search results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"      Content: {result['content'][:60]}...")
        else:
            print("❌ No search results found")
            return False
        
        # Test 4: Test different tables
        print("4. Testing master_vector table...")
        self.add_document("This is a master document for testing", {"type": "master"}, "master_vector")
        
        print("5. Testing search_history table...")
        self.insert_search_history("test search query", {"user": "test_user"})
        
        print("✅ DatabaseVectorStore test completed successfully!")
        return True

class SimpleVectorStore:
    """Simple in-memory vector store for testing"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.embedding_model = "models/embedding-001"
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        print("✅ SimpleVectorStore initialized")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        def _generate_embedding():
            return genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
        
        response = safe_api_call(_generate_embedding)
        return response["embedding"] if response else []
    
    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """Add document to the store"""
        embedding = self.get_embedding(text)
        if embedding:
            self.documents.append({"text": text, "metadata": metadata or {}})
            self.embeddings.append(embedding)
            print(f"✅ Added document: '{text[:50]}...'")
            return True
        return False
    
    def search_similar(self, query: str, limit: int = 3):
        """Search for similar documents"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Simple cosine similarity
        import math
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, similarity in similarities[:limit]:
            results.append({
                "content": self.documents[i]["text"],
                "metadata": self.documents[i]["metadata"],
                "similarity": similarity
            })
        
        return results
    
    def test_functionality(self):
        """Test the simple vector store"""
        print("\n🔬 Testing SimpleVectorStore functionality...")
        
        # Add test documents
        test_docs = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning models require large datasets to train effectively.",
            "Deep learning uses neural networks with multiple hidden layers."
        ]
        
        for doc in test_docs:
            self.add_document(doc, {"source": "test"})
        
        # Test search
        results = self.search_similar("What is AI and machine learning?", limit=2)
        
        if results:
            print("✅ Search results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"      Content: {result['content'][:60]}...")
            return True
        else:
            print("❌ No search results found")
            return False

def process_bank_transactions(limit: int = 10):
    """Process bank transactions with explicit prompting"""
    print(f"\n🔄 Processing bank transactions (limit: {limit})...")
    try:
        filename = "bank_transactions_with_vendor_100.csv"
        if not os.path.exists(filename):
            print(f"⚠️  File {filename} not found.")
            return []
        
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df)} transactions")
        
        # Create LLM instance
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            convert_system_message_to_human=True
        )
        
        results = []
        
        for i, row in df.iterrows():
            if i >= limit:
                break
                
            try:
                print(f"🔄 Processing transaction {i+1}: {row['Description'][:30]}...")
                
                # Single message prompt
                message = (
                    f"Categorize this financial transaction into exactly one of these categories: "
                    f"Education, Health, Groceries, Transportation, Utilities, Entertainment, Shopping, Other\n\n"
                    f"Transaction: {row['Description']} at {row['Vendor']} for ${row['Amount']}\n\n"
                    f"Category:"
                )
                
                response = llm.invoke(message)
                category = response.content.strip().replace("Category:", "").strip()
                
                # Ensure it's a valid category
                valid_categories = ['Education', 'Health', 'Groceries', 'Transportation', 
                                 'Utilities', 'Entertainment', 'Shopping', 'Other']
                
                if category not in valid_categories:
                    for valid_cat in valid_categories:
                        if valid_cat.lower() in category.lower():
                            category = valid_cat
                            break
                    else:
                        category = 'Other'
                
                results.append({
                    "transaction_id": i + 1,
                    "description": row["Description"],
                    "vendor": row["Vendor"],
                    "amount": row["Amount"],
                    "category": category
                })
                
                print(f"✅ Category: {category}")
                
            except Exception as e:
                print(f"❌ Error processing transaction {i+1}: {e}")
                results.append({
                    "transaction_id": i + 1,
                    "description": row["Description"],
                    "vendor": row["Vendor"],
                    "amount": row["Amount"],
                    "category": "Error"
                })
            
            time.sleep(0.5)
        
        print(f"✅ Processed {len(results)} transactions")
        return results
        
    except Exception as e:
        print(f"❌ Transaction processing failed: {e}")
        return []

def setup_database_url():
    """Interactive setup for database URL"""
    print("\n🔧 Database Configuration")
    print("Choose your database option:")
    print("1. PostgreSQL (recommended for production)")
    print("2. SQLite (simple file-based)")
    print("3. Skip database tests")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # PostgreSQL setup
        default_url = "postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db"
        print(f"\nDefault PostgreSQL URL: {default_url}")
        custom_url = input("Enter custom URL (or press Enter for default): ").strip()
        
        return custom_url if custom_url else default_url
    
    elif choice == "2":
        # SQLite setup
        return "sqlite:///vector_store.db"
    
    else:
        return None

def main():
    """Main function to demonstrate functionality"""
    print("🚀 Starting GenAI Vector Store Demo...")
    
    # Process bank transactions (if file exists)
    transaction_results = process_bank_transactions(limit=5)
    
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
    
    if database_url and database_url.startswith("postgresql"):
        # Test PostgreSQL vector store
        print("\n" + "="*50)
        print("TESTING POSTGRESQL VECTOR STORE")
        print("="*50)
        
        try:
            db_store = DatabaseVectorStore(database_url)
            
            # Create table
            if db_store.create_table():
                # Test with sample file
                csv_file = "bank_transactions_with_vendor_100.csv"
                if os.path.exists(csv_file):
                    print(f"🔄 Processing file: {csv_file}")
                    db_store.process_file(csv_file, "temp_transactions")
                else:
                    # Add test documents
                    test_docs = [
                        "Artificial intelligence is transforming industries.",
                        "Machine learning algorithms learn from data patterns.",
                        "Deep learning uses neural networks with multiple layers."
                    ]
                    
                    for doc in test_docs:
                        db_store.add_document(doc, {"source": "demo"})
                
                # Search
                results = db_store.search_similar("What is AI?", limit=2)
                if results:
                    print("✅ Database vector store test successful!")
                    for result in results:
                        print(f"   Similarity: {result['similarity']:.3f} - {result['content'][:50]}...")
                
        except Exception as e:
            print(f"❌ DatabaseVectorStore failed: {e}")
            print("💡 Check PostgreSQL setup and ensure database exists")
    
    else:
        print("⚠️  Database tests skipped")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()
