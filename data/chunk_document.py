"""
Semantic Chunking with LangChain
Uses AI-based text splitting that understands context and topic boundaries
"""

import json
import re
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def extract_pdf_with_metadata(pdf_path: str):
    """
    Extract PDF content with page numbers and section detection
    """
    print(f"\n[1/4] Extracting PDF: {pdf_path}")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")

    # Extract all text with page mapping
    pages_content = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            pages_content.append({
                'page_number': page_num,
                'text': text
            })

    # Combine all text
    full_text = "\n\n".join([p['text'] for p in pages_content])

    print(f"Extracted {len(pages_content)} pages")
    print(f"Total characters: {len(full_text):,}")

    return full_text, pages_content


def detect_sections(full_text: str):
    """
    Detect section headers and create section metadata
    """
    print("\n[2/4] Detecting document sections...")

    # Pattern for "SECTION X: Title"
    section_pattern = re.compile(
        r'SECTION\s+(\d+|[A-Z]+)[:\s]+([A-Z][A-Za-z\s,\-&]+?)(?=\n|$)',
        re.MULTILINE
    )

    sections = []
    for match in section_pattern.finditer(full_text):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start_pos = match.start()

        sections.append({
            'section_number': section_num,
            'section_title': section_title,
            'start_position': start_pos
        })
        print(f"  Found: Section {section_num} - {section_title}")

    # Add appendix
    appendix_pattern = re.compile(r'APPENDIX\s+([A-Z])', re.MULTILINE)
    for match in appendix_pattern.finditer(full_text):
        appendix_id = match.group(1)
        sections.append({
            'section_number': f'APPENDIX_{appendix_id}',
            'section_title': f'Appendix {appendix_id}',
            'start_position': match.start()
        })
        print(f"  Found: Appendix {appendix_id}")

    print(f"Detected {len(sections)} sections")
    return sections


def get_section_for_position(position: int, sections: list, full_text: str):
    """
    Determine which section a text position belongs to
    """
    current_section = {'section_number': '0', 'section_title': 'Introduction'}

    for section in sections:
        if position >= section['start_position']:
            current_section = section
        else:
            break

    return current_section


def get_page_numbers(chunk_text: str, pages_content: list, full_text: str):
    """
    Determine which pages a chunk spans
    """
    # Find chunk position in full text
    position = full_text.find(chunk_text)
    if position == -1:
        return None, None

    chunk_end = position + len(chunk_text)

    # Build cumulative text positions
    cumulative_pos = 0
    page_ranges = []

    for page in pages_content:
        page_text = page['text']
        page_start = cumulative_pos
        page_end = cumulative_pos + len(page_text) + 2  # +2 for "\n\n" separator

        page_ranges.append({
            'page_number': page['page_number'],
            'start': page_start,
            'end': page_end
        })

        cumulative_pos = page_end

    # Find which pages the chunk spans
    pages_touched = []
    for page_range in page_ranges:
        # Check if chunk overlaps with this page
        if (position < page_range['end'] and chunk_end > page_range['start']):
            pages_touched.append(page_range['page_number'])

    if pages_touched:
        return min(pages_touched), max(pages_touched)
    return None, None


def extract_key_terms(text: str):
    """
    Extract potential key terms from chunk text
    """
    # Simple keyword extraction - look for capitalized terms and common HR terms
    keywords = []

    # Look for capitalized phrases (proper nouns, titles)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    keywords.extend(capitalized[:5])  # Top 5

    # Common HR policy terms
    hr_terms = ['vacation', 'holiday', 'leave', 'salary', 'compensation', 'benefits',
                'overtime', 'termination', 'probation', 'policy', 'procedure', 'employee',
                'employer', 'workplace', 'safety', 'harassment', 'discrimination']

    text_lower = text.lower()
    for term in hr_terms:
        if term in text_lower:
            keywords.append(term)

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for k in keywords:
        k_lower = k.lower()
        if k_lower not in seen:
            seen.add(k_lower)
            unique_keywords.append(k)

    return unique_keywords[:10]  # Max 10 keywords


def semantic_chunking(full_text: str, sections: list, pages_content: list):
    """
    Use LangChain's RecursiveCharacterTextSplitter for semantic chunking
    with rich metadata extraction

    This splitter:
    - Tries to keep paragraphs together
    - Splits on sentence boundaries
    - Maintains semantic coherence
    - Adds overlap for context continuity
    """
    print("\n[3/4] Performing semantic chunking with LangChain...")

    # Configure semantic splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,        # ~400 words (average 4 chars per word)
        chunk_overlap=200,      # ~50 words overlap for context
        length_function=len,
        separators=[
            "\n\n",   # Paragraph breaks (highest priority)
            "\n",     # Line breaks
            ". ",     # Sentence endings
            "! ",     # Exclamation sentences
            "? ",     # Question sentences
            "; ",     # Semicolons
            ", ",     # Commas
            " ",      # Words
            ""        # Characters (last resort)
        ],
        is_separator_regex=False,
    )

    # Split text into chunks
    chunks = text_splitter.split_text(full_text)

    print(f"Created {len(chunks)} semantic chunks")
    print(f"Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} characters")

    # Add rich metadata to each chunk
    chunks_with_metadata = []

    for idx, chunk_text in enumerate(chunks):
        # Find position in original text
        position = full_text.find(chunk_text)

        # Determine which section this chunk belongs to
        section_info = get_section_for_position(position, sections, full_text)

        # Get page numbers
        page_start, page_end = get_page_numbers(chunk_text, pages_content, full_text)

        # Extract key terms
        key_terms = extract_key_terms(chunk_text)

        # Create chunk with RICH metadata
        chunk_obj = {
            'id': f"chunk_{idx}",
            'text': chunk_text,

            # Section information
            'section_number': section_info['section_number'],
            'section_title': section_info['section_title'],

            # Position information
            'chunk_index': idx,
            'page_start': page_start,
            'page_end': page_end,

            # Content statistics
            'char_count': len(chunk_text),
            'word_count': len(chunk_text.split()),

            # Semantic metadata
            'key_terms': key_terms,

            # Source document
            'source_document': 'policy_example.pdf',

            # Timestamp
            'chunking_method': 'langchain_recursive_semantic'
        }

        chunks_with_metadata.append(chunk_obj)

    # Display stats
    print("\nChunking Statistics:")
    print(f"  Total chunks: {len(chunks_with_metadata)}")
    print(f"  Avg words per chunk: {sum(c['word_count'] for c in chunks_with_metadata) // len(chunks_with_metadata)}")
    print(f"  Min words: {min(c['word_count'] for c in chunks_with_metadata)}")
    print(f"  Max words: {max(c['word_count'] for c in chunks_with_metadata)}")

    return chunks_with_metadata


def save_chunks(chunks: list, output_path: str):
    """
    Save chunks to JSON file
    """
    print(f"\n[4/4] Saving chunks to: {output_path}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Saved {len(chunks)} chunks")


def main():
    """
    Main semantic chunking pipeline
    """
    print("=" * 80)
    print("  SEMANTIC CHUNKING WITH LANGCHAIN")
    print("=" * 80)

    # Configuration
    PDF_PATH = "policy_example.pdf"
    OUTPUT_PATH = "data/chunks.json"

    # Step 1: Extract PDF
    full_text, pages_content = extract_pdf_with_metadata(PDF_PATH)

    # Step 2: Detect sections
    sections = detect_sections(full_text)

    # Step 3: Semantic chunking with rich metadata
    chunks = semantic_chunking(full_text, sections, pages_content)

    # Step 4: Save chunks
    save_chunks(chunks, OUTPUT_PATH)

    # Show sample chunk with rich metadata
    print("\n" + "=" * 80)
    print("SAMPLE CHUNK WITH RICH METADATA")
    print("=" * 80)
    sample = chunks[10]  # Show middle chunk
    print(f"ID: {sample['id']}")
    print(f"Section: {sample['section_number']} - {sample['section_title']}")
    print(f"Pages: {sample['page_start']}-{sample['page_end']}")
    print(f"Words: {sample['word_count']} | Characters: {sample['char_count']}")
    print(f"Key Terms: {', '.join(sample['key_terms'][:5])}")
    print(f"Source: {sample['source_document']}")
    print(f"Method: {sample['chunking_method']}")
    print(f"\nText (first 300 chars):")
    print(sample['text'][:300].encode('ascii', 'ignore').decode('ascii') + "...")
    print("=" * 80)

    print(f"\n[NEXT STEP] Run: python generate_embeddings.py")


if __name__ == "__main__":
    main()
