"""
Document ingestion module for the Course Planning Assistant.

Handles loading PDF and HTML files, cleaning extracted text,
and chunking into overlapping segments with metadata for
vector store indexing.
"""

import os
import re
import uuid
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


# ── Tokenizer helper ──────────────────────────────────────────
def _get_tokenizer():
    """Return a tiktoken tokenizer for counting tokens."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    enc = _get_tokenizer()
    return len(enc.encode(text))


def _decode_tokens(tokens: list) -> str:
    """Decode a list of token IDs back to a string."""
    enc = _get_tokenizer()
    return enc.decode(tokens)


def _encode_tokens(text: str) -> list:
    """Encode a text string into token IDs."""
    enc = _get_tokenizer()
    return enc.encode(text)


# ── Text Cleaning ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove noise from extracted text.

    Strips headers, footers, page numbers, excessive whitespace,
    and other common PDF/HTML artifacts.

    Args:
        text: Raw extracted text from a document.

    Returns:
        Cleaned text string.
    """
    # Remove page numbers (standalone numbers on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove common header/footer patterns
    text = re.sub(r'(?i)page\s+\d+\s*(of\s+\d+)?', '', text)
    text = re.sub(r'(?i)^\s*(confidential|draft|©.*)\s*$', '', text, flags=re.MULTILINE)

    # Collapse multiple newlines into double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces into single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Strip overall
    return text.strip()


# ── Section Detection ─────────────────────────────────────────
def _detect_section(text: str) -> str:
    """
    Attempt to detect the section heading from a chunk of text.

    Looks for common heading patterns at the start of the text.

    Args:
        text: A chunk of text to analyze.

    Returns:
        Detected section heading, or 'General' if none found.
    """
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        # All-caps headings
        if line and len(line) < 100 and line.isupper():
            return line.title()
        # Lines that look like headings (short, no period at end)
        if line and len(line) < 80 and not line.endswith('.') and not line.endswith(','):
            # Check if it looks like a section heading
            if re.match(r'^[A-Z][A-Za-z\s:&/\-]+$', line):
                return line
            # Numbered heading
            if re.match(r'^\d+[\.\)]\s+\w', line):
                return line
    return "General"


# ── Chunking ──────────────────────────────────────────────────
def chunk_text(
    text: str,
    source: str,
    page: Optional[int] = None,
    section: str = "General",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """
    Split text into overlapping chunks with metadata.

    Uses token-based chunking to produce chunks of approximately
    `chunk_size` tokens with `chunk_overlap` token overlap.

    Args:
        text: The cleaned text to split.
        source: Source filename or URL.
        page: Page number (for PDFs).
        section: Detected section heading.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of chunk dictionaries with keys:
        content, source, page, chunk_id, section.
    """
    if not text or not text.strip():
        return []

    tokens = _encode_tokens(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_content = _decode_tokens(chunk_tokens).strip()

        if chunk_content:
            detected_section = _detect_section(chunk_content) if section == "General" else section
            chunks.append({
                "content": chunk_content,
                "source": source,
                "page": page,
                "chunk_id": str(uuid.uuid4()),
                "section": detected_section,
            })

        # Move forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap
        if start >= len(tokens):
            break

    return chunks


# ── PDF Loading ───────────────────────────────────────────────
def load_pdf(file_path: str) -> List[dict]:
    """
    Load and chunk a PDF file into pieces with metadata.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of chunk dictionaries from the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If PDF parsing fails.
    """
    from pypdf import PdfReader

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    print(f"  [Ingestion] Loading PDF: {file_path}")

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise Exception(f"Failed to read PDF '{file_path}': {e}")

    source = os.path.basename(file_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        page_chunks = chunk_text(
            text=cleaned,
            source=source,
            page=page_num,
            section="General",
        )
        all_chunks.extend(page_chunks)

    print(f"  [Ingestion] Extracted {len(all_chunks)} chunks from {source}")
    return all_chunks


# ── HTML Loading ──────────────────────────────────────────────
def load_html(file_path: str) -> List[dict]:
    """
    Load and chunk an HTML file into pieces with metadata.

    Args:
        file_path: Absolute or relative path to the HTML file.

    Returns:
        List of chunk dictionaries from the HTML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If HTML parsing fails.
    """
    from bs4 import BeautifulSoup

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HTML file not found: {file_path}")

    print(f"  [Ingestion] Loading HTML: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
    except Exception as e:
        raise Exception(f"Failed to read HTML '{file_path}': {e}")

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Extract text from meaningful elements
    source = os.path.basename(file_path)
    all_chunks = []

    # Try to extract by sections (headings)
    sections = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if sections:
        for heading in sections:
            section_name = heading.get_text(strip=True)
            # Collect text until next heading
            content_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    break
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    content_parts.append(text)

            section_text = "\n".join(content_parts)
            cleaned = clean_text(section_text)
            if cleaned:
                section_chunks = chunk_text(
                    text=cleaned,
                    source=source,
                    section=section_name,
                )
                all_chunks.extend(section_chunks)
    else:
        # Fallback: extract all text
        full_text = soup.get_text(separator="\n", strip=True)
        cleaned = clean_text(full_text)
        if cleaned:
            all_chunks = chunk_text(
                text=cleaned,
                source=source,
                section="General",
            )

    print(f"  [Ingestion] Extracted {len(all_chunks)} chunks from {source}")
    return all_chunks


# ── Multi-document Loading ────────────────────────────────────
def load_documents(file_paths: List[str]) -> List[dict]:
    """
    Load multiple PDF/HTML files and return all chunks.

    Args:
        file_paths: List of file paths (PDF or HTML).

    Returns:
        Combined list of chunk dictionaries from all files.
    """
    all_chunks = []
    print(f"\n{'='*50}")
    print(f"[Ingestion] Processing {len(file_paths)} document(s)...")
    print(f"{'='*50}")

    for file_path in file_paths:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                chunks = load_pdf(file_path)
            elif ext in (".html", ".htm"):
                chunks = load_html(file_path)
            else:
                print(f"  [Ingestion] ⚠ Skipping unsupported file type: {file_path}")
                continue
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [Ingestion] ✗ Error processing {file_path}: {e}")

    print(f"\n[Ingestion] Total chunks generated: {len(all_chunks)}")
    return all_chunks
