import requests
import io
import xml.etree.ElementTree as ET

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def _parse_grobid_tei(tei_xml: str) -> str:
    """Parses GROBID's TEI XML output into a clean, LLM-friendly text format."""
    try:
        # Simple string-based extraction for robustness against namespace issues
        import re
        
        def extract_tag(tag, text):
            match = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return re.sub(r'<[^>]+>', '', match.group(1)).strip() if match else ""
            
        title = extract_tag("title", tei_xml)
        abstract = extract_tag("abstract", tei_xml)
        
        # Extract body divs
        body = ""
        divs = re.findall(r"<div[^>]*>(.*?)</div>", tei_xml, re.DOTALL | re.IGNORECASE)
        for d in divs:
            head = extract_tag("head", d)
            p_tags = re.findall(r"<p[^>]*>(.*?)</p>", d, re.DOTALL | re.IGNORECASE)
            text = "\n".join([re.sub(r'<[^>]+>', '', p).strip() for p in p_tags])
            if head:
                body += f"\n\n--- {head.upper()} ---\n{text}"
            else:
                body += f"\n{text}"
                
        output = f"[PARSED PAPER] GROBID PARSED RESEARCH PAPER\n"
        output += f"TITLE: {title}\n\n"
        if abstract: output += f"ABSTRACT:\n{abstract}\n\n"
        output += f"BODY:{body}\n"
        return output
    except Exception as e:
        return f"GROBID TEI Parsing error: {e}"

def extract_text_from_pdf(pdf_path_or_url: str) -> str:
    """
    Extracts structured text from a PDF (Local or ArXiv link).
    Tries GROBID first for deep academic parsing. Falls back to PyPDF2.
    """
    try:
        pdf_bytes = None
        if pdf_path_or_url.startswith("http"):
            if "arxiv.org/abs/" in pdf_path_or_url:
                pdf_path_or_url = pdf_path_or_url.replace("/abs/", "/pdf/") + ".pdf"
            print(f"[DOWNLOAD] Downloading PDF from {pdf_path_or_url}...")
            response = requests.get(pdf_path_or_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            pdf_bytes = response.content
        else:
            with open(pdf_path_or_url, 'rb') as f:
                pdf_bytes = f.read()
                
        # 1. TRY GROBID
        try:
            print("[GROBID] Attempting to parse paper with GROBID structural intelligence...")
            files = {'input': ('paper.pdf', pdf_bytes, 'application/pdf')}
            grobid_res = requests.post(GROBID_URL, files=files, timeout=30)
            if grobid_res.status_code == 200:
                print("[OK] GROBID parsing successful!")
                text = _parse_grobid_tei(grobid_res.text)
                return text[:40000] # Safe limit
        except Exception as e:
            print("[WARNING] GROBID server not reachable (localhost:8070). Falling back to PyPDF2...")
            
        # 2. FALLBACK TO PYPDF2
        if not PyPDF2:
            return "Error: PyPDF2 is not installed, and GROBID is unavailable."
            
        file_obj = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(file_obj)
        text = f"--- PDF EXTRACTION (PyPDF2 Fallback): {pdf_path_or_url} ---\n\n"
        
        for i, page in enumerate(reader.pages):
            text += f"\n[Page {i+1}]\n"
            page_text = page.extract_text()
            if page_text: text += page_text
            
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... [TRUNCATED due to length.]"
            
        return text
        
    except Exception as e:
        return f"Failed to extract PDF: {str(e)}"
