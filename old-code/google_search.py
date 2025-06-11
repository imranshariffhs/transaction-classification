import requests
import json
from typing import Dict, List, Optional

class GoogleCustomSearch:
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10, start_index: int = 1, 
               site_search: Optional[str] = None, exact_terms: Optional[str] = None,
               exclude_terms: Optional[str] = None, file_type: Optional[str] = None,
               date_restrict: Optional[str] = None) -> Dict:
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10),  # API limits to 10 results per request
            'start': start_index
        }
        
        # Add optional parameters
        if site_search:
            params['siteSearch'] = site_search
        if exact_terms:
            params['exactTerms'] = exact_terms
        if exclude_terms:
            params['excludeTerms'] = exclude_terms
        if file_type:
            params['fileType'] = file_type
        if date_restrict:
            params['dateRestrict'] = date_restrict
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return {}
    
    def get_search_results(self, query: str, max_results: int = 10) -> List[Dict]:
        results = []
        start_index = 1
        
        while len(results) < max_results:
            # Calculate how many results to request in this batch
            batch_size = min(10, max_results - len(results))
            
            response = self.search(query, num_results=batch_size, start_index=start_index)
            
            if 'items' not in response:
                break
            
            for item in response['items']:
                if len(results) >= max_results:
                    break
                
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'display_link': item.get('displayLink', ''),
                    'formatted_url': item.get('formattedUrl', '')
                }
                results.append(result)
            
            start_index += batch_size
            
            # Check if there are more results available
            if 'queries' not in response or 'nextPage' not in response['queries']:
                break
        
        return results
    
    def print_results(self, query: str, max_results: int = 10):
        """
        Print search results in a readable format
        """
        results = self.get_search_results(query, max_results)
        
        print(f"\nSearch Results for: '{query}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['link']}")
            print(f"   {result['snippet']}")
            print("-" * 40)
        
        if not results:
            print("No results found.")

    def has_transaction_mention(self, query: str) -> bool:
        """
        Check if any search results exist for the given query.
        Returns True if at least one result is found, False otherwise.
        """
        results = self.get_search_results(query, max_results=1)
        return bool(results)

if __name__ == "__main__":
    # Replace these with your actual API key and Search Engine ID
    API_KEY = "YOUR_API_KEY"
    SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"
    
    # Initialize the search client
    search_client = GoogleCustomSearch(API_KEY, SEARCH_ENGINE_ID)
    
    # Example query to check for Amazon transactions
    query = "How can I determine whether a company or organization has received transactions related to Amazon?"
    
    # Print detailed search results
    search_client.print_results(query, max_results=1)
    
    # Check if transactions are mentioned and print boolean result
    has_transaction = search_client.has_transaction_mention(query)
    print("\nTransaction Check Result:", "TRUE" if has_transaction else "FALSE") 