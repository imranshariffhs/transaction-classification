

import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

"""# **Set your Google API key** *local we using .env*"""

# Set your Google API key (if not already in environment)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyACjg7EaWTAB3-lXMmbmK-3sIbuY08erKQ"

"""# **Set your LangSmith key** *local we using .env*"""

# Set your LangSmith API key (optional, for tracing)
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"

"""### **Create the ChatGoogleGenerativeAI instance**  I using gemini-2.0-flash"""

# Create the ChatGoogleGenerativeAI instance

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

"""#  **Calling Our Models or LLM**"""

# Make a call to the Gemini model
response = llm.invoke("Can you explain the concept of 'few-shot learning' in LLMs and its advantages?")
# Access the generated text
print(response.content)

"""## **Chain calls with Prompt Template**

"""

import pandas as pd
import time
from langchain_core.prompts import ChatPromptTemplate

df=pd.read_excel("bank_transactions_with_vendor_100.xlsx")

print(df.head(5))
print(df.columns)
new_df=pd.DataFrame(df,columns=['Vendor','Description','Amount','Transaction Type'])


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

# # Create prompt chain
# chain = prompt | llm

# # Run prompt for each transaction and collect results
# results = []
# for _, row in new_df.iterrows():
#     input_data = {
#         "description": row["Description"],
#         "vendor": row["Vendor"],
#         "transaction_type": row["Transaction Type"],
#         "amount": row["Amount"]
#     }
#     result = chain.invoke(input_data)
#     results.append(result.content)

# # Print first 5 results
# for res in results[:5]:
#     print(res)


# Create prompt chain
chain = prompt | llm

results = []
for i, row in new_df.iterrows():
    input_data = {
        "description": row["Description"],
        "vendor": row["Vendor"],
        "transaction_type": row["Transaction Type"],
        "amount": row["Amount"]
    }
    try:
        result = chain.invoke(input_data)
        results.append(result.content)
    except Exception as e:
        print(f"Error at row {i}: {e}")
        results.append(None)
    time.sleep(5)  # Sleep 5 seconds to stay within quota

def get_settings():
    return {
        "vector_store": {
            "host": "localhost",
            "port": 6333,
            "table_name": "my_vector_table",
            "embedding_dimensions": 768,
            "time_partition_interval": "day",
        },
        "database": {
            "service_url": "postgresql://postgres:root123@localhost:5432/mydb"
        }
    }

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import time
import os
from tqdm import tqdm

# Try to import the vector database client
try:
    from timescale_vector import client
    VECTOR_CLIENT_AVAILABLE = True
    print("✅ Timescale Vector client imported successfully")
except ImportError as e:
    VECTOR_CLIENT_AVAILABLE = False
    print(f"❌ Timescale Vector not available: {e}")
    print("Install with: pip install timescale-vector")

def get_settings():
    """Default settings - you should replace this with your actual settings"""
    return {
        "database": {
            "service_url": "postgresql://username:password@localhost:5432/your_database"
        },
        "vector_store": {
            "table_name": "vector_embeddings",
            "embedding_dimensions": 768,
            "time_partition_interval": "7 days"
        }
    }

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import time
import os
from tqdm import tqdm

# Try to import the vector database client
try:
    from timescale_vector import client
    VECTOR_CLIENT_AVAILABLE = True
    print("✅ Timescale Vector client imported successfully")
except ImportError as e:
    VECTOR_CLIENT_AVAILABLE = False
    print(f"❌ Timescale Vector not available: {e}")
    print("Install with: pip install timescale-vector")


def get_settings():
    """Default settings - you should replace this with your actual settings"""
    return {
        "database": {
            "service_url": "postgresql://username:password@localhost:5432/your_database"
        },
        "vector_store": {
            "table_name": "vector_embeddings",
            "embedding_dimensions": 768,
            "time_partition_interval": "7 days"
        }
    }


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize settings and Gemini embedding client."""
        self.settings = get_settings()
        self.vector_settings = self.settings["vector_store"]

        # Gemini embedding model
        self.embedding_model = "models/embedding-001"

        # Set Gemini API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)

        # Initialize vector database client
        if VECTOR_CLIENT_AVAILABLE:
            try:
                print(f"🔄 Connecting to database: {self.settings['database']['service_url']}")
                print(f"🔄 Table name: {self.vector_settings['table_name']}")
                print(f"🔄 Embedding dimensions: {self.vector_settings['embedding_dimensions']}")

                self.vec_client = client.Sync(
                    self.settings["database"]["service_url"],
                    self.vector_settings["table_name"],
                    self.vector_settings["embedding_dimensions"],
                    time_partition_interval=self.vector_settings["time_partition_interval"],
                )
                print("✅ Vector database client initialized successfully")

                # Test the connection
                self._test_connection()

            except Exception as e:
                print(f"❌ Failed to initialize vector database client: {e}")
                print("Make sure your PostgreSQL database is running and accessible")
                print("Also ensure the database has the required extensions installed:")
                print("  - CREATE EXTENSION IF NOT EXISTS vector;")
                print("  - CREATE EXTENSION IF NOT EXISTS timescaledb;")
                self.vec_client = None
        else:
            self.vec_client = None
            print("⚠️  Vector database client not available. Install timescale-vector to enable database operations.")

    def _test_connection(self):
        """Test the database connection"""
        try:
            # Try to get some basic info about the table
            print("🔄 Testing database connection...")
            # This is a simple test - adjust based on timescale-vector API
            print("✅ Database connection test passed")
        except Exception as e:
            print(f"⚠️  Database connection test failed: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using Gemini (Google Generative AI).

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()

        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response["embedding"]

            elapsed_time = time.time() - start_time
            logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")

            return embedding
        except Exception as e:
            logging.error(f"Failed to get embedding: {e}")
            print(f"❌ Embedding generation failed: {e}")
            return []

    def insert_vector(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> bool:
        """
        Insert a vector into the database.

        Args:
            text: The text to generate embedding for
            metadata: Optional metadata dictionary
            doc_id: Optional document ID

        Returns:
            True if successful, False otherwise
        """
        if not self.vec_client:
            print("❌ Vector database client not available. Cannot insert vector.")
            return False

        print(f"🔄 Generating embedding for text: '{text[:50]}...'")
        embedding = self.get_embedding(text)
        if not embedding:
            print("❌ Failed to generate embedding")
            return False

        print(f"✅ Generated embedding with {len(embedding)} dimensions")

        try:
            # Prepare the record
            record_id = doc_id or str(int(time.time() * 1000))
            record = {
                "id": record_id,
                "embedding": embedding,
                "metadata": metadata or {},
                "contents": text  # Store the original text
            }

            print(f"🔄 Inserting record with ID: {record_id}")
            print(f"🔄 Record keys: {list(record.keys())}")
            print(f"🔄 Metadata: {record['metadata']}")

            # Insert into the vector database
            result = self.vec_client.upsert([record])
            print(f"🔄 Upsert result: {result}")
            print(f"✅ Successfully inserted vector for text: '{text[:50]}...'")
            return True

        except Exception as e:
            print(f"❌ Failed to insert vector: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Error details: {str(e)}")

            # Additional debugging info
            if hasattr(e, 'args') and e.args:
                print(f"❌ Error args: {e.args}")

            return False

    def search_similar(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar vectors in the database.

        Args:
            query_text: The query text to search for
            limit: Maximum number of results to return

        Returns:
            List of similar documents with metadata
        """
        if not self.vec_client:
            print("❌ Vector database client not available. Cannot search.")
            return []

        print(f"🔄 Generating query embedding for: '{query_text}'")
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []

        try:
            print(f"🔄 Searching for {limit} similar documents...")
            results = self.vec_client.search(
                query_embedding,
                limit=limit
            )
            print(f"✅ Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            return []

    def create_table_if_not_exists(self):
        """Create the vector table if it doesn't exist"""
        if not self.vec_client:
            print("❌ Vector database client not available. Cannot create table.")
            return False

        try:
            print("🔄 Attempting to create table if it doesn't exist...")
            # This depends on the timescale-vector API
            # You might need to adjust this based on the actual API
            self.vec_client.create_tables()
            print("✅ Table creation completed")
            return True
        except Exception as e:
            print(f"❌ Failed to create table: {e}")
            return False

    def test_full_functionality(self):
        """Test both embedding generation and database operations."""
        print("\n🔬 Testing VectorStore functionality...")

        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        print(f"\n1. Testing embedding generation for: '{test_text}'")
        embedding = self.get_embedding(test_text)

        if embedding:
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
            print(f"   First 5 values: {embedding[:5]}")
        else:
            print("❌ Failed to generate embedding")
            return

        # Test database operations if client is available
        if self.vec_client:
            print("\n2. Ensuring table exists...")
            self.create_table_if_not_exists()

            print("\n3. Testing database insertion...")
            success = self.insert_vector(
                text=test_text,
                metadata={"source": "test", "type": "example", "timestamp": time.time()},
                doc_id=f"test_doc_{int(time.time())}"
            )

            if success:
                print("\n4. Testing similarity search...")
                results = self.search_similar("test sentence", limit=3)
                if results:
                    print(f"✅ Search returned {len(results)} results")
                    for i, result in enumerate(results):
                        print(f"   Result {i+1}: {result}")
                else:
                    print("⚠️  No search results returned")
            else:
                print("❌ Database insertion failed, skipping search test")
        else:
            print("\n2. Database operations skipped (client not available)")

    def debug_connection(self):
        """Debug connection and configuration issues"""
        print("\n🔍 Debugging VectorStore connection...")
        print(f"🔍 Database URL: {self.settings['database']['service_url']}")
        print(f"🔍 Table name: {self.vector_settings['table_name']}")
        print(f"🔍 Embedding dimensions: {self.vector_settings['embedding_dimensions']}")
        print(f"🔍 Time partition: {self.vector_settings['time_partition_interval']}")
        print(f"🔍 Vector client available: {VECTOR_CLIENT_AVAILABLE}")
        print(f"🔍 Vector client initialized: {self.vec_client is not None}")

        if self.vec_client:
            print("🔍 Vector client object:", type(self.vec_client))

# Example usage and testing
if __name__ == "__main__":
    # Make sure to set your environment variables:
    # export GOOGLE_API_KEY="your_gemini_api_key"

    try:
        vector_store = VectorStore()
        vector_store.debug_connection()
        vector_store.test_full_functionality()
    except Exception as e:
        print(f"❌ Failed to initialize VectorStore: {e}")
        print("Please check your configuration and environment variables")

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import time
import os
from datetime import timedelta
import psycopg2
from urllib.parse import urlparse

# Try to import the vector database client
try:
    from timescale_vector import client
    VECTOR_CLIENT_AVAILABLE = True
    print("✅ Timescale Vector client imported successfully")
except ImportError as e:
    VECTOR_CLIENT_AVAILABLE = False
    print(f"❌ Timescale Vector not available: {e}")
    print("Install with: pip install timescale-vector")


def get_settings():
    """Get database settings from environment or use defaults"""

    # Get from environment variable or use default
    database_url = os.getenv("DATABASE_URL", "postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db")

    return {
        "database": {
            "service_url": database_url
        },
        "vector_store": {
            "table_name": "vector_embeddings",
            "embedding_dimensions": 768,
            "time_partition_interval": timedelta(days=7)  # Use timedelta object instead of string
        }
    }


def test_database_connection_comprehensive(database_url: str) -> bool:
    """Test database connection with multiple approaches"""

    print(f"🔄 Testing database connection...")
    print(f"   Database URL: {database_url}")

    # Parse the URL
    try:
        parsed = urlparse(database_url)
        print(f"   Host: {parsed.hostname}")
        print(f"   Port: {parsed.port or 5432}")
        print(f"   User: {parsed.username}")
        print(f"   Database: {parsed.path[1:] if parsed.path else 'None'}")
    except Exception as e:
        print(f"❌ Failed to parse database URL: {e}")
        return False

    # Method 1: Try with parsed components
    connection_methods = [
        {
            "name": "localhost with parsed components",
            "params": {
                "host": parsed.hostname or 'localhost',
                "port": parsed.port or 5432,
                "user": parsed.username,
                "password": 'SecurePassword123!',
                "database": parsed.path[1:] if parsed.path and len(parsed.path) > 1 else 'postgres'
            }
        },
        {
            "name": "127.0.0.1 with parsed components",
            "params": {
                "host": '127.0.0.1',
                "port": parsed.port or 5432,
                "user": parsed.username,
                "password": 'SecurePassword123!',
                "database": parsed.path[1:] if parsed.path and len(parsed.path) > 1 else 'postgres'
            }
        },
        {
            "name": "direct URL string",
            "params": {"dsn": database_url}
        }
    ]

    for i, method in enumerate(connection_methods, 1):
        print(f"\n🔄 Method {i}: Trying {method['name']}...")

        try:
            if "dsn" in method["params"]:
                conn = psycopg2.connect(method["params"]["dsn"])
            else:
                conn = psycopg2.connect(**method["params"])

            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"✅ Connection successful: {version[0][:50]}...")

            # Check if we're connected to the right database
            cursor.execute("SELECT current_database(), current_user;")
            db_info = cursor.fetchone()
            print(f"✅ Connected to database: {db_info[0]} as user: {db_info[1]}")

            # Check if required extensions exist
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'timescaledb');")
            extensions = cursor.fetchall()
            ext_names = [ext[0] for ext in extensions]

            if 'vector' in ext_names:
                print("✅ Vector extension is installed")
            else:
                print("❌ Vector extension is NOT installed")
                print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")

            if 'timescaledb' in ext_names:
                print("✅ TimescaleDB extension is installed")
            else:
                print("❌ TimescaleDB extension is NOT installed")
                print("   Run: CREATE EXTENSION IF NOT EXISTS timescaledb;")

            cursor.close()
            conn.close()
            return True

        except psycopg2.OperationalError as e:
            if "password authentication failed" in str(e):
                print(f"❌ Password authentication failed")
                print("   Solutions:")
                print("   1. Check if user exists: sudo -u postgres psql -c '\\du'")
                print("   2. Reset password: sudo -u postgres psql -c \"ALTER USER vector_user PASSWORD 'SecurePassword123!';\"")
                print("   3. Check pg_hba.conf authentication method")
            elif "database" in str(e) and "does not exist" in str(e):
                print(f"❌ Database does not exist")
                print("   Create it with: sudo -u postgres createdb vector_db")
            else:
                print(f"❌ Connection failed: {e}")

        except Exception as e:
            print(f"❌ Connection failed: {e}")

    print(f"\n❌ All connection methods failed")
    return False


def create_database_and_user():
    """Helper function to create database and user"""
    print("🔧 Attempting to create database and user...")

    try:
        # Connect as postgres superuser
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            database='postgres'
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create database
        try:
            cursor.execute("CREATE DATABASE vector_db;")
            print("✅ Database 'vector_db' created")
        except psycopg2.errors.DuplicateDatabase:
            print("ℹ️  Database 'vector_db' already exists")

        # Create user
        try:
            cursor.execute("CREATE USER vector_user WITH PASSWORD 'SecurePassword123!';")
            print("✅ User 'vector_user' created")
        except psycopg2.errors.DuplicateObject:
            print("ℹ️  User 'vector_user' already exists")
            cursor.execute("ALTER USER vector_user PASSWORD 'SecurePassword123!';")
            print("✅ Password updated for 'vector_user'")

        # Grant privileges
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE vector_db TO vector_user;")
        print("✅ Privileges granted")

        cursor.close()
        conn.close()

        # Now connect to vector_db and create extensions
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            database='vector_db'
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create extensions
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ Vector extension created")
        except Exception as e:
            print(f"⚠️  Could not create vector extension: {e}")

        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            print("✅ TimescaleDB extension created")
        except Exception as e:
            print(f"⚠️  Could not create timescaledb extension: {e}")

        # Grant schema privileges
        cursor.execute("GRANT USAGE ON SCHEMA public TO vector_user;")
        cursor.execute("GRANT CREATE ON SCHEMA public TO vector_user;")
        cursor.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vector_user;")
        cursor.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vector_user;")

        cursor.close()
        conn.close()

        print("✅ Database setup completed")
        return True

    except Exception as e:
        print(f"❌ Failed to create database and user: {e}")
        print("Please run the setup script manually or check PostgreSQL installation")
        return False


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize settings and Gemini embedding client."""
        self.settings = get_settings()
        self.vector_settings = self.settings["vector_store"]

        # Gemini embedding model
        self.embedding_model = "models/embedding-001"

        # Set Gemini API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")

        # Test database connection first
        db_url = self.settings["database"]["service_url"]
        if not test_database_connection_comprehensive(db_url):
            print("\n🔧 Attempting to create database setup...")
            if create_database_and_user():
                print("🔄 Retrying database connection...")
                if not test_database_connection_comprehensive(db_url):
                    print("❌ Database connection still failed after setup")
                    self.vec_client = None
                    return
            else:
                print("❌ Database setup failed")
                self.vec_client = None
                return

        # Initialize vector database client
        if VECTOR_CLIENT_AVAILABLE:
            try:
                print(f"\n🔄 Initializing vector client...")
                print(f"🔄 Database URL: {db_url}")
                print(f"🔄 Table name: {self.vector_settings['table_name']}")
                print(f"🔄 Embedding dimensions: {self.vector_settings['embedding_dimensions']}")
                print(f"🔄 Time partition: {self.vector_settings['time_partition_interval']}")

                self.vec_client = client.Sync(
                    service_url=db_url,
                    table_name=self.vector_settings["table_name"],
                    num_dimensions=self.vector_settings["embedding_dimensions"],
                    time_partition_interval=self.vector_settings["time_partition_interval"],
                )
                print("✅ Vector database client initialized successfully")

            except Exception as e:
                print(f"❌ Failed to initialize vector database client: {e}")
                print(f"❌ Error type: {type(e).__name__}")
                print(f"❌ Error details: {str(e)}")
                self.vec_client = None
        else:
            self.vec_client = None
            print("⚠️  Vector database client not available.")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using Gemini."""
        text = text.replace("\n", " ").strip()
        if not text:
            return []

        start_time = time.time()

        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response["embedding"]

            elapsed_time = time.time() - start_time
            print(f"✅ Embedding generated in {elapsed_time:.3f} seconds")

            return embedding
        except Exception as e:
            print(f"❌ Failed to get embedding: {e}")
            return []

    def create_table(self) -> bool:
        """Create the vector table"""
        if not self.vec_client:
            print("❌ Vector database client not available")
            return False

        try:
            print("🔄 Creating vector table...")
            self.vec_client.create_tables()
            print("✅ Vector table created successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to create table: {e}")
            return False

    def insert_vector(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> bool:
        """Insert a vector into the database."""
        if not self.vec_client:
            print("❌ Vector database client not available")
            return False

        print(f"🔄 Processing text: '{text[:50]}...'")
        embedding = self.get_embedding(text)
        if not embedding:
            print("❌ Failed to generate embedding")
            return False

        try:
            # Prepare the record
            record_id = doc_id or f"doc_{int(time.time() * 1000)}"

            # Create record in the format expected by timescale-vector
            record = {
                "id": record_id,
                "metadata": metadata or {},
                "contents": text,
                "embedding": embedding
            }

            print(f"🔄 Inserting record with ID: {record_id}")

            # Insert into the vector database
            self.vec_client.upsert([record])
            print(f"✅ Successfully inserted vector")
            return True

        except Exception as e:
            print(f"❌ Failed to insert vector: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            return False

    def search_similar(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Search for similar vectors in the database."""
        if not self.vec_client:
            print("❌ Vector database client not available")
            return []

        print(f"🔄 Searching for: '{query_text}'")
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []

        try:
            results = self.vec_client.search(
                query_embedding,
                limit=limit
            )
            print(f"✅ Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def test_functionality(self):
        """Test the complete functionality"""
        print("\n🔬 Testing VectorStore functionality...")

        # Test 1: Embedding generation
        test_text = "Machine learning is transforming how we process data."
        print(f"\n1. Testing embedding generation...")
        embedding = self.get_embedding(test_text)

        if not embedding:
            print("❌ Embedding test failed")
            return

        print(f"✅ Generated {len(embedding)} dimensional embedding")

        # Test 2: Database operations
        if not self.vec_client:
            print("\n❌ Skipping database tests - client not available")
            return

        print(f"\n2. Creating table...")
        if not self.create_table():
            print("❌ Table creation failed")
            return

        print(f"\n3. Testing vector insertion...")
        success = self.insert_vector(
            text=test_text,
            metadata={"source": "test", "category": "ml"},
            doc_id="test_doc_1"
        )

        if not success:
            print("❌ Vector insertion failed")
            return

        print(f"\n4. Testing similarity search...")
        results = self.search_similar("data processing", limit=3)

        if results:
            print("✅ All tests passed!")
            for i, result in enumerate(results[:2]):
                print(f"   Match {i+1}: {result}")
        else:
            print("⚠️  No search results found")


def main():
    """Main function to test the VectorStore"""
    print("🚀 Starting VectorStore setup and testing...")

    try:
        # Check environment variables
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ GOOGLE_API_KEY environment variable not set")
            print("Set it with: export GOOGLE_API_KEY='your_api_key'")
            return

        # Initialize and test VectorStore
        vector_store = VectorStore()

        if vector_store.vec_client:
            vector_store.test_functionality()
        else:
            print("❌ VectorStore initialization failed")
            print("\n🔧 Manual setup required:")
            print("1. Run the PostgreSQL setup script")
            print("2. Ensure PostgreSQL is running: sudo systemctl start postgresql")
            print("3. Create user manually: sudo -u postgres createuser -P vector_user")
            print("4. Create database: sudo -u postgres createdb -O vector_user vector_db")

    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()

