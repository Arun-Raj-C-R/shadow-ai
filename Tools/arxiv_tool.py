# arxiv_tool.py
import requests
import feedparser
import time
import logging
from urllib.parse import urlencode
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArxivAPIError(Exception):
    """Custom exception for API-level errors."""
    pass

class ArxivAPI:
    """
    A comprehensive Python client for the arXiv API.
    
    This client implements all features from the documentation, including
    querying, ID-based retrieval, filtering, sorting, and automatic paging.
    """
    
    BASE_URL = 'http://export.arxiv.org/api/query?'
    
    def __init__(self, rate_limit_delay: float = 3.0, default_page_size: int = 100):
        """
        Initializes the ArxivAPI client.
        
        Args:
            rate_limit_delay (float): Seconds to wait between paged API calls,
                                      as requested by the arXiv documentation.
            default_page_size (int): Number of results to fetch per page in
                                     the 'search' generator. Max is 2000.
        """
        self.delay = rate_limit_delay
        self.page_size = min(default_page_size, 2000) # Max efficient slice is 2000
        logging.info(f"ArxivAPI client initialized. Rate limit delay: {self.delay}s")

    def _make_request(self, params: dict) -> feedparser.FeedParserDict:
        """
        Internal method to perform the HTTP GET request and parse the response.
        """
        try:
            url = self.BASE_URL + urlencode(params)
            response = requests.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx, 5xx)
            
            feed = feedparser.parse(response.content)
            
            # Check for API-level errors (returned as an Atom feed with an error entry)
            if 'entry' in feed and 'title' in feed.entry and feed.entry.title == 'Error':
                error_summary = feed.entry.get('summary', 'Unknown API Error')
                logging.error(f"ArXiv API Error: {error_summary}")
                raise ArxivAPIError(error_summary)
                
            return feed
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Request failed: {e}")
            raise ArxivAPIError(f"HTTP Request failed: {e}")

    def _parse_entry(self, entry: feedparser.FeedParserDict) -> dict:
        """
        Internal method to parse a single Atom <entry> into a clean Python dict.
        """
        abs_url = entry.get('id')
        arxiv_id = abs_url.split('/abs/')[-1]
        
        pdf_url = ''
        for link in entry.get('links', []):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        
        return {
            'id': arxiv_id,
            'title': entry.get('title'),
            'summary': entry.get('summary'),
            'authors': [author.get('name') for author in entry.get('authors', [])],
            'primary_category': entry.get('arxiv_primary_category', {}).get('term'),
            'categories': [tag.get('term') for tag in entry.get('tags', [])],
            'published': entry.get('published'),
            'updated': entry.get('updated'),
            'abs_url': abs_url,
            'pdf_url': pdf_url,
            'comment': entry.get('arxiv_comment'),
            'journal_ref': entry.get('arxiv_journal_ref'),
            'doi': entry.get('arxiv_doi'),
        }

    def query(self, 
              search_query: str = None, 
              id_list: list[str] = None, 
              start: int = 0, 
              max_results: int = 10, 
              sortBy: str = 'relevance', 
              sortOrder: str = 'descending') -> dict:
        """
        Performs a single, low-level query to the arXiv API.
        """
        params = {
            'start': start,
            'max_results': max_results,
            'sortBy': sortBy,
            'sortOrder': sortOrder
        }
        
        if search_query:
            params['search_query'] = search_query
        
        if id_list:
            params['id_list'] = ','.join(id_list)
            
        # Ensure at least one query parameter is present
        if 'search_query' not in params and 'id_list' not in params:
            raise ValueError("You must provide 'search_query' or 'id_list'.")

        logging.info(f"Performing low-level query: {params}")
        feed = self._make_request(params)
        
        results = [self._parse_entry(entry) for entry in feed.entries]
        
        meta = {
            'total_results': int(feed.feed.get('opensearch_totalresults', 0)),
            'start_index': int(feed.feed.get('opensearch_startindex', 0)),
            'items_per_page': int(feed.feed.get('opensearch_itemsperpage', 0)),
            'query_url': feed.feed.get('link')
        }
        
        return {'meta': meta, 'results': results}

    def search(self, 
               search_query: str = None, 
               id_list: list[str] = None,
               sortBy: str = 'relevance', 
               sortOrder: str = 'descending',
               max_total_results: int = 1000):
        """
        High-level search generator that handles automatic paging and rate limiting.
        """
        start = 0
        results_fetched = 0
        
        # Clamp max_total_results to the API's hard limit of 30,000
        api_hard_limit = 30000
        max_total_results = min(max_total_results, api_hard_limit)

        logging.info(f"Starting paged search. Max results: {max_total_results}. Page size: {self.page_size}.")
        
        while results_fetched < max_total_results:
            # Calculate remaining results needed
            remaining = max_total_results - results_fetched
            current_page_size = min(self.page_size, remaining)
            
            if current_page_size <= 0:
                break
                
            try:
                page_data = self.query(
                    search_query=search_query,
                    id_list=id_list,
                    start=start,
                    max_results=current_page_size,
                    sortBy=sortBy,
                    sortOrder=sortOrder
                )
            except ArxivAPIError as e:
                logging.error(f"Failed to fetch page at start={start}: {e}")
                break

            page_results = page_data.get('results', [])
            num_on_page = len(page_results)
            
            if num_on_page == 0:
                logging.info("No more results found. Stopping search.")
                break # No more results

            # In the first iteration, update max_total_results to the *actual*
            # number of results found, if it's smaller.
            if start == 0:
                total_api_results = page_data['meta'].get('total_results', 0)
                if total_api_results < max_total_results:
                    logging.info(f"API found only {total_api_results} results. Adjusting max.")
                    max_total_results = total_api_results

            for entry in page_results:
                if results_fetched < max_total_results:
                    yield entry
                    results_fetched += 1
                else:
                    break # Reached max_total_results mid-page
            
            start += num_on_page
            
            # If we are not done, sleep to respect rate limit
            if results_fetched < max_total_results and results_fetched > 0:
                logging.info(f"Fetched {results_fetched}/{max_total_results}. Sleeping for {self.delay}s...")
                time.sleep(self.delay)
                
        logging.info(f"Search complete. Total results fetched: {results_fetched}")

    def search_by_ids(self, id_list: list[str], search_query: str = None) -> list[dict]:
        """
        A helper method to retrieve articles by IDs, optionally filtering them.
        """
        if not id_list:
            return []
        
        logging.info(f"Fetching {len(id_list)} articles by ID.")
        data = self.query(id_list=id_list, search_query=search_query, max_results=len(id_list))
        return data.get('results', [])

# ==============================================================================
# --- 1. GLOBAL API CLIENT INSTANCE ---
# ==============================================================================
# We create one instance to be shared by all tool calls
api_client = ArxivAPI(rate_limit_delay=3.0)


# ==============================================================================
# --- 2. TOOL WRAPPER FUNCTIONS ---
# ==============================================================================
# These are the functions the AI will actually call.
# They handle the generator and return a clean, JSON-friendly result.

def search_arxiv(search_query: str, max_results: int = 5, sort_by: str = 'submittedDate', sort_order: str = 'descending') -> str:
    """
    Searches arXiv for papers matching a query.
    Returns a JSON string of the results.
    """
    try:
        logging.info(f"Tool call: search_arxiv with query='{search_query}', max_results={max_results}")
        
        # The 'search' method is a generator. We must consume it.
        search_gen = api_client.search(
            search_query=search_query,
            max_total_results=max_results,
            sortBy=sort_by,
            sortOrder=sort_order
        )
        
        # Convert the generator to a list
        results = list(search_gen)
        
        if not results:
            return "No papers found matching that query."
            
        # Return a JSON string, as tool outputs must be simple types.
        return json.dumps(results)
        
    except Exception as e:
        logging.error(f"Error in search_arxiv tool: {e}")
        return f"Error: {e}"

def get_arxiv_papers_by_id(id_list: list[str]) -> str:
    """
    Retrieves specific arXiv papers based on a list of their IDs.
    Returns a JSON string of the results.
    """
    try:
        logging.info(f"Tool call: get_arxiv_papers_by_id with IDs='{id_list}'")
        if not id_list or not isinstance(id_list, list):
            return "Error: You must provide a list of paper IDs."
            
        results = api_client.search_by_ids(id_list=id_list)
        
        if not results:
            return "No papers found for the given IDs."
            
        # Return a JSON string
        return json.dumps(results)
        
    except Exception as e:
        logging.error(f"Error in get_arxiv_papers_by_id tool: {e}")
        return f"Error: {e}"

# ==============================================================================
# --- 3. TOOL DEFINITIONS FOR THE AGENT ---
# ==============================================================================
# This is the schema the main agent will see.

ARXIV_TOOL_DEFINITIONS = [
    {
        "name": "search_arxiv",
        "description": "Search the arXiv academic paper database for papers matching a query string.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "search_query": {
                    "type": "STRING",
                    "description": "The search query (e.g., 'all:quantum computing' or 'au:Einstein AND ti:relativity')."
                },
                "max_results": {
                    "type": "INTEGER",
                    "description": "The maximum number of papers to return. Default is 5."
                },
                "sort_by": {
                    "type": "STRING",
                    "description": "Sort metric. 'relevance', 'lastUpdatedDate', or 'submittedDate'. Default is 'submittedDate'."
                },
                "sort_order": {
                    "type": "STRING",
                    "description": "Sort order. 'ascending' or 'descending'. Default is 'descending'."
                }
            },
            "required": ["search_query"]
        }
    },
    {
        "name": "get_arxiv_papers_by_id",
        "description": "Retrieve specific arXiv papers by their unique IDs (e.g., '2307.08654v1' or 'hep-ex/0307015v1').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "id_list": {
                    "type": "ARRAY",
                    "description": "A list of one or more arXiv paper IDs.",
                    "items": { "type": "STRING" }
                }
            },
            "required": ["id_list"]
        }
    }
]