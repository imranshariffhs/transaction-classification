import pandas as pd

def process_vector_store_data(db_store):
    try:
        # Process transactions
        print("🔄 Processing bank transactions...")
        test_docs = pd.read_csv('bank_transactions_with_vendor_100.csv')
        test_docs = test_docs.to_dict(orient='records')
        for doc in test_docs:
            db_store.insert_temp_transaction(doc, {"source": "demo"})
        
        # Process master data
        print("🔄 Processing master data...")                
        test_docs_master = pd.read_csv('Master_data.csv')
        test_docs_master = test_docs_master.to_dict(orient='records')
        for doc_master in test_docs_master:
            db_store.insert_master_vector(doc_master, {"source": "demo"})
        
        # Search
        print("🔄 Testing search functionality...")
        query = "I want to retrieve all transaction details related to Zomato, including a clear summary of the total inflow and outflow amounts"
        results = db_store.search_similar(query, limit=2)
        
        if results:
            print("✅ Search results:")
            for result in results:
                print(f"   Similarity: {result['similarity']:.3f} - {result['content'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False 