import pandas as pd
import json
from typing import Dict, Any, Optional, Union

def test_database_vector_store(db_store):
    """Test the PostgreSQL vector store functionality"""
    print("\n" + "="*50)
    print("TESTING POSTGRESQL VECTOR STORE")
    print("="*50)
    
    try:
        # Load and process bank transactions
        print("🔄 Loading bank transactions...")
        test_docs = pd.read_csv('bank_transactions_with_vendor_100.csv')
        test_docs = test_docs.to_dict(orient='records')
        
        print(f"✅ Loaded {len(test_docs)} transactions")
        
        # Process each transaction
        for i, doc in enumerate(test_docs, 1):
            # Create a meaningful string representation
            description = doc.get('Description', 'N/A')
            print(f"🔄 Processing transaction {i}: {description[:30]}...")
            
            # Create transaction string and metadata
            transaction_str = f"Transaction: {description} at {doc.get('Vendor', 'N/A')} for ${doc.get('Amount', 'N/A')}"
            metadata = {
                "source": "demo",
                "original_data": doc,
                "transaction_type": doc.get('Type', 'Unknown')
            }
            
            # Insert into database
            success = db_store.insert_temp_transaction(transaction_str, metadata)
            if success:
                print(f"✅ Category: {doc.get('Category', 'Other')}")
            
        print(f"✅ Processed {len(test_docs)} transactions")
        
        # Load and process master data
        print("\n🔄 Loading master data...")
        try:
            test_docs_master = pd.read_csv('Master_data.csv')
            test_docs_master = test_docs_master.to_dict(orient='records')
            
            for i, doc_master in enumerate(test_docs_master, 1):
                description = doc_master.get('Description', 'N/A')
                print(f"🔄 Processing master record {i}: {description[:30]}...")
                
                # Create master data string and metadata
                master_str = f"Transaction: {description} at {doc_master.get('Vendor', 'N/A')} for ${doc_master.get('Amount', 'N/A')}"
                metadata = {
                    "source": "demo",
                    "original_data": doc_master,
                    "record_type": "master"
                }
                
                # Insert into database
                success = db_store.insert_master_vector(master_str, metadata)
                if success:
                    print(f"✅ Master record processed")
                    
            print(f"✅ Processed {len(test_docs_master)} master records")
            
        except FileNotFoundError:
            print("⚠️ Master_data.csv not found, skipping master data processing")
        
        # Test search functionality
        print("\n🔄 Testing search functionality...")
        query = "I want to retrieve all transaction details related to Zomato, including a clear summary of the total inflow and outflow amounts"
        results = db_store.search_similar(query, limit=2)
        
        if results:
            print("✅ Search results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Similarity: {result['similarity']:.3f}")
                print(f"      Content: {result['content'][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ DatabaseVectorStore failed: {e}")
        print("💡 Try using SQLite option or check PostgreSQL setup")
        return False 