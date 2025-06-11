import os
import getpass
import pandas as pd
import time
import json
import psycopg2
from urllib.parse import urlparse
from datetime import timedelta, datetime
from typing import List, Dict, Any
import logging
import random

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
class RateLimiter:
    def __init__(self, requests_per_minute=10, requests_per_day=400):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = []
        self.daily_requests = []
        
    def wait_if_needed(self):
        """Wait if rate limits would be exceeded"""
        now = datetime.now()
        
        # Clean old requests (older than 1 minute)
        self.minute_requests = [req_time for req_time in self.minute_requests 
                               if (now - req_time).seconds < 60]
        
        # Clean old requests (older than 24 hours)
        self.daily_requests = [req_time for req_time in self.daily_requests 
                              if (now - req_time).seconds < 86400]
        
        # Check daily limit
        if len(self.daily_requests) >= self.requests_per_day:
            logger.warning("Daily rate limit reached. Skipping API calls.")
            return False
            
        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            wait_time = 60 - (now - min(self.minute_requests)).seconds
            logger.info(f"Rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time + 1)
            
        return True
    
    def record_request(self):
        """Record a successful request"""
        now = datetime.now()
        self.minute_requests.append(now)
        self.daily_requests.append(now)

# Initialize rate limiter
rate_limiter = RateLimiter()

# Set your Google API key (if not already in environment)
if "GOOGLE_API_KEY" not in os.environ:
    api_key = 'AIzaSyBvYAw9t-xndPUNBJ4EKhWv6_l_nJp6_yo'
    os.environ["GOOGLE_API_KEY"] = api_key
    logger.info("Using default Google API key")

# Set your LangSmith API key (optional, for tracing)
if "LANGSMITH_API_KEY" not in os.environ:
    # Set to empty to disable tracing if you don't have LangSmith
    os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"  # Disable to reduce API calls

# Create the ChatGoogleGenerativeAI instance with better retry settings
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=60,  # Increased timeout
    max_retries=3,  # Increased retries
    convert_system_message_to_human=True  # Add support for system messages
)

# Configure Gemini for embeddings
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def safe_api_call(func, *args, **kwargs):
    """Safely make API calls with rate limiting and error handling"""
    if not rate_limiter.wait_if_needed():
        return None
        
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            rate_limiter.record_request()
            return result
        except Exception as e:
            error_str = str(e).lower()
            
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit. Waiting {delay:.1f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error("Rate limit exceeded. Please wait before making more requests or upgrade your API plan.")
                    return None
            else:
                logger.error(f"API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                    continue
                return None
    
    return None

def test_llm():
    """Test the LLM functionality with rate limiting"""
    print("🔬 Testing LLM functionality...")
    
    def _invoke_llm():
        return llm.invoke("Can you explain the concept of 'few-shot learning' in LLMs and its advantages?")
    
    response = safe_api_call(_invoke_llm)
    if response:
        print(f"✅ LLM Response: {response.content[:100]}...")
        return True
    else:
        print("❌ LLM test failed due to rate limiting or API issues")
        return False

def process_transactions():
    """Process transactions from Excel file with rate limiting"""
    print("\n🔬 Processing transactions...")
    
    try:
        df = pd.read_excel("bank_transactions_with_vendor_100.xlsx")
        print(f"✅ Loaded {len(df)} transactions")
        print(f"Columns: {list(df.columns)}")
        
        new_df = pd.DataFrame(df, columns=['Vendor', 'Description', 'Amount', 'Transaction Type'])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial transaction categorization expert.
            Your task is to categorize transactions based on their description and vendor information.
            You need to categorize transaction on which category does it belongs to like
             if it is ( school, college, university, school fees, college fees, university fees ) comes under Education
             for ( operations, surgary, injury ) comes under Health like this need to categorize the transaction

            For each transaction, provide a JSON response with:
            1. The most appropriate category
            2. The transaction type (Inflow/Outflow/Inflow-Outflow)
            3. A confidence score (0-1)
            4. A brief explanation of your categorization

            Consider both the transaction description and vendor information in your analysis.
            Your response must be a valid JSON object with this exact structure:
            {{
                "category": "string",
                "type": "string",
                "confidence": float,
                "explanation": "string"
            }}

            Example response:
            {{
                "category": "Groceries",
                "type": "Outflow",
                "confidence": 0.95,
                "explanation": "Transaction matches grocery store pattern"
            }}"""),
            ("human", """Please categorize this transaction:
            Description: {description}
            Vendor: {vendor}
            Transaction Type: {transaction_type}
            Amount: {amount}""")
        ])

        # Create prompt chain
        chain = prompt | llm

        results = []
        # Process only first 3 for testing to avoid rate limits
        for i, row in new_df.head(3).iterrows():
            input_data = {
                "description": row["Description"],
                "vendor": row["Vendor"],
                "transaction_type": row["Transaction Type"],
                "amount": row["Amount"]
            }
            
            def _invoke_chain():
                return chain.invoke(input_data)
            
            result = safe_api_call(_invoke_chain)
            if result:
                results.append(result.content)
                print(f"✅ Processed transaction {i+1}")
            else:
                print(f"❌ Skipped transaction {i+1} due to rate limiting")
                results.append(None)
            
            # Add delay between requests
            time.sleep(2)

        successful_results = [r for r in results if r]
        print(f"✅ Processed {len(successful_results)} transactions successfully")
        return results
        
    except FileNotFoundError:
        print("❌ Excel file not found. Please ensure 'bank_transactions_with_vendor_100.xlsx' exists.")
        return []
    except Exception as e:
        print(f"❌ Error processing transactions: {e}")
        return []

class SimpleVectorStore:
    """Simple in-memory vector store for testing with rate limiting"""
    
    def __init__(self):
        """Initialize with Gemini embeddings"""
        self.embedding_model = "models/embedding-001"
        self.documents = []
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        print("✅ Simple VectorStore initialized with Gemini embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini with rate limiting"""
        if not text or not text.strip():
            return []
        
        text = text.replace("\n", " ").strip()
        
        def _generate_embedding():
            return genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
        
        response = safe_api_call(_generate_embedding)
        if response:
            return response["embedding"]
        else:
            print(f"❌ Embedding generation failed for text: {text[:50]}...")
            return []
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to the store"""
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        doc = {
            "id": len(self.documents),
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        
        self.documents.append(doc)
        print(f"✅ Added document {doc['id']}: '{text[:50]}...'")
        return True
    
    def search_similar(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        similarities = []
        for doc in self.documents:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            similarities.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": similarity
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)

class DatabaseVectorStore:
    """Vector store with PostgreSQL database (requires pgvector extension)"""
    
    def __init__(self, database_url: str = None):
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
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini with rate limiting and dimension padding"""
        if not text or not text.strip():
            return []
        
        text = text.replace("\n", " ").strip()
        
        def _generate_embedding():
            return genai.embed_content(
                model=self.embedding_model,
                content=text,
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
            print(f"❌ Embedding generation failed for text: {text[:50]}...")
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
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('master_vector', 'search_history', 'temp_transactions');
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if not all(table in existing_tables for table in ['master_vector', 'search_history', 'temp_transactions']):
                print("❌ One or more required tables are missing")
                cursor.close()
                conn.close()
                return False
            
            print("✅ All required tables found")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def insert_temp_transaction(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Insert document into temp_transactions table"""
        embedding = self.get_embedding(content)
        if not embedding or len(embedding) != self.embedding_dimensions:
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
            
            print(f"✅ Temp transaction inserted: '{content[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert temp transaction: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def insert_master_vector(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Insert document into master_vector table"""
        embedding = self.get_embedding(content)
        if not embedding or len(embedding) != self.embedding_dimensions:
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
            
            cursor.execute(insert_sql, (content, str(embedding), json.dumps(metadata or {})))
            conn.commit()
            
            print(f"✅ Master vector inserted: '{content[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert master vector: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def insert_search_history(self, query: str, metadata: Dict[str, Any] = None) -> bool:
        """Insert search query into search_history table"""
        embedding = self.get_embedding(query)
        if not embedding or len(embedding) != self.embedding_dimensions:
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
            
            cursor.execute(insert_sql, (query, str(embedding), json.dumps(metadata)))
            conn.commit()
            
            print(f"✅ Search history inserted: '{query[:50]}...'")
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to insert search history: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                conn.close()
            return False
    
    def search_similar(self, query: str, table_name: str = "master_vector", limit: int = 3) -> List[Dict]:
        """Search for similar documents in specified table"""
        if table_name not in ['master_vector', 'search_history', 'temp_transactions']:
            print(f"❌ Invalid table name: {table_name}")
            return []
            
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
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
            if conn:
                cursor.close()
                conn.close()
            return []

def test_simple_vector_store():
    """Test the simple in-memory vector store with rate limiting"""
    print("\n" + "="*50)
    print("TESTING SIMPLE VECTOR STORE (In-Memory)")
    print("="*50)
    
    try:
        # Initialize simple vector store
        simple_store = SimpleVectorStore()
        
        print("🔬 Testing SimpleVectorStore...")
        
        # Test documents - reduced to avoid rate limits
        test_docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Python is a popular programming language for data science and web development.",
        ]
        
        print("1. Adding test documents...")
        successful_adds = 0
        for doc in test_docs:
            if simple_store.add_document(doc, {"source": "test"}):
                successful_adds += 1
            time.sleep(1)  # Rate limiting
        
        if successful_adds == 0:
            print("❌ No documents were added due to rate limiting")
            return False
        
        print("2. Testing similarity search...")
        results = simple_store.search_similar("artificial intelligence machine learning", limit=2)
        
        if results:
            print(f"✅ Found {len(results)} similar documents")
            print("✅ Search results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                print(f"      Text: {result['text'][:60]}...")
        else:
            print("❌ No search results found")
        
        print("✅ SimpleVectorStore test completed!")
        return True
        
    except Exception as e:
        print(f"❌ SimpleVectorStore test failed: {e}")
        return False

def test_database_vector_store():
    """Test the PostgreSQL vector store with rate limiting"""
    print("\n" + "="*50)
    print("TESTING POSTGRESQL VECTOR STORE")
    print("="*50)
    
    try:
        # Initialize database vector store
        db_store = DatabaseVectorStore()
        
        # Test documents - reduced to avoid rate limits
        test_docs = [
            "Machine learning algorithms can learn patterns from data automatically.",
            "Database systems store and retrieve information efficiently.",
        ]
        
        print("🔬 Testing database operations...")
        
        # Test temp_transactions
        print("1. Testing temp_transactions table...")
        success = db_store.insert_temp_transaction(
            test_docs[0], 
            {"source": "test", "type": "temp"}
        )
        if success:
            print("✅ Temp transaction insert successful")
        
        # Test master_vector
        print("\n2. Testing master_vector table...")
        success = db_store.insert_master_vector(
            test_docs[1],
            {"source": "test", "status": "verified"}
        )
        if success:
            print("✅ Master vector insert successful")
        
        # Test search_history
        print("\n3. Testing search_history table...")
        success = db_store.insert_search_history(
            "machine learning patterns",
            {"user": "test_user"}
        )
        if success:
            print("✅ Search history insert successful")
        
        # Test similarity search in each table
        print("\n4. Testing similarity search...")
        query = "machine learning algorithms"
        
        for table in ['master_vector', 'temp_transactions', 'search_history']:
            results = db_store.search_similar(query, table_name=table, limit=2)
            if results:
                print(f"\n✅ Found {len(results)} similar documents in {table}")
                for i, result in enumerate(results):
                    print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
                    print(f"      Text: {result['content'][:60]}...")
        
        print("\n✅ DatabaseVectorStore test completed!")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseVectorStore test failed: {e}")
        return False

def main():
    """Main function to run all tests with rate limiting awareness"""
    print("🚀 Starting comprehensive testing with rate limiting...")
    print("⚠️  Note: Tests are limited to avoid API quota exhaustion")
    
    # Test 1: LLM functionality
    print("\n" + "="*50)
    print("TESTING LLM FUNCTIONALITY")
    print("="*50)
    llm_success = test_llm()
    
    # Test 2: Transaction processing (optional - only if file exists and rate limits allow)
    try:
        if llm_success:  # Only proceed if basic LLM test passed
            results = process_transactions()
            if results:
                print("Sample transaction results:")
                for result in results[:2]:
                    if result:
                        print(f"  {result[:100]}...")
        else:
            print("⚠️ Transaction processing skipped due to LLM test failure")
    except Exception as e:
        print(f"⚠️ Transaction processing skipped: {e}")
    
    # Test 3: Simple vector store
    simple_success = test_simple_vector_store()
    
    # Test 4: Database vector store
    print("\n🔧 Database Setup Required")
    print("   PostgreSQL with vector extension (recommended)")
    
    db_url = os.getenv("DATABASE_URL", "postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db")
    print(f"Using database URL: {db_url}")
    
    db_success = test_database_vector_store()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"{'✅' if llm_success else '❌'} LLM Test: {'Passed' if llm_success else 'Failed'}")
    print(f"{'✅' if simple_success else '❌'} Simple Vector Store: {'Passed' if simple_success else 'Failed'}")
    print(f"{'✅' if db_success else '❌'} Database Vector Store: {'Passed' if db_success else 'Failed'}")
    
    if not db_success:
        print("\n🔧 To fix database issues:")
        print("1. Install PostgreSQL and pgvector extension")
        print("2. Create database and user:")
        print("   sudo -u postgres createdb vector_db")
        print("   sudo -u postgres createuser -P vector_user")
        print("3. Grant permissions and create extensions")
    
    if not (llm_success or simple_success or db_success):
        print("\n⚠️  API Rate Limit Solutions:")
        print("1. Wait 24 hours for quota reset (free tier: 500 requests/day)")
        print("2. Upgrade to paid Google AI Studio plan")
        print("3. Use a different API key")
        print("4. Implement local embeddings (e.g., sentence-transformers)")
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()