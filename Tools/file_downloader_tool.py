# file_downloader_tool.py
import os
import requests
import pathlib
import logging

# --- Configuration ---
# Set the target download directory using a raw string for the Windows path
DOWNLOAD_DIR = pathlib.Path(r"D:\Project File\Shadow\Shadow\Brain\Shadow2\arxiv_download")

# ==============================================================================
# --- 1. TOOL WRAPPER FUNCTION ---
# ==============================================================================

def download_file_from_url(url: str, filename: str) -> str:
    """
    Downloads a file from a URL and saves it to a predefined local directory.
    """
    try:
        # Ensure the target directory exists
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sanitize the filename to prevent path traversal attacks (e.g., "../malicious.exe")
        # os.path.basename will extract just the filename from any path
        sanitized_filename = os.path.basename(filename)
        
        # Ensure the filename ends with .pdf
        if not sanitized_filename.lower().endswith('.pdf'):
            sanitized_filename += ".pdf"
            
        # Create the full, safe path to save the file
        full_save_path = DOWNLOAD_DIR / sanitized_filename
        
        logging.info(f"Tool call: download_file_from_url. URL: {url}, Save Path: {full_save_path}")

        # Make the request to download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Will raise an HTTPError for bad responses
            
            # Write the file to disk in chunks
            with open(full_save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        success_message = f"SUCCESS: File '{sanitized_filename}' downloaded to {DOWNLOAD_DIR}"
        logging.info(success_message)
        return success_message

    except requests.exceptions.RequestException as e:
        error_message = f"FAILURE: Could not download file. HTTP Error: {e}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"FAILURE: An unexpected error occurred: {e}"
        logging.error(error_message)
        return error_message

# ==============================================================================
# --- 2. TOOL DEFINITION FOR THE AGENT ---
# ==============================================================================

DOWNLOAD_TOOL_DEFINITIONS = [
    {
        "name": "download_file_from_url",
        "description": "Downloads a file from a given URL (e.g., a PDF from an arXiv search) and saves it to the user's local 'arxiv_download' folder.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "url": {
                    "type": "STRING",
                    "description": "The direct URL of the file to download (e.g., the 'pdf_url' from an arXiv search result)."
                },
                "filename": {
                    "type": "STRING",
                    "description": "The desired filename for the saved file. **Crucially, you MUST use the paper's arXiv ID as the filename** and end it with '.pdf' (e.g., '2307.08654v1.pdf')."
                }
            },
            "required": ["url", "filename"]
        }
    }
]