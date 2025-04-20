import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
from datetime import datetime
from pathlib import Path
import os
from typing import Any, Dict, List, Union

from . import BaseVectorDB


def _sanitize_metadata_value(value: Any) -> Union[str, int, float, bool]:
    """
    Convert metadata values to ChromaDB supported types.

    Args:
        value: The value to sanitize

    Returns:
        Union[str, int, float, bool]: The sanitized value
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return " | ".join(str(item) for item in value)
    elif isinstance(value, dict):
        return str(value)
    else:
        return str(value)


def _sanitize_metadata(
    metadata: Dict[str, Any],
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Sanitize all metadata values to ensure they are ChromaDB compatible.

    Args:
        metadata: The metadata dictionary to sanitize

    Returns:
        Dict[str, Union[str, int, float, bool]]: The sanitized metadata
    """
    return {key: _sanitize_metadata_value(value) for key, value in metadata.items()}


class BOCYLVectorDB(BaseVectorDB):
    """Class for managing BOCYL documents in a ChromaDB collection."""

    def __init__(
        self,
        db_path=None,
        model_name=None,
        embedding_function=None,
        persist_directory=True,
        max_section_length=2000,  # Maximum length for a single section
    ):
        """
        Initialize the BOCYL vector database.

        Args:
            db_path (str, optional): Path to the ChromaDB database. If not provided,
                defaults to /workspace/data/vectordb/chromadb
            model_name (str, optional): Name of the HuggingFace model to use for embeddings
            embedding_function: Custom embedding function to use
            persist_directory (bool): Whether to persist the database to disk
            max_section_length (int): Maximum length for a single section before splitting
        """
        if db_path is None:
            db_path = Path("/workspace/data/vectordb/chromadb")

        super().__init__(
            db_path=db_path,
            collection_name="bocyl",
            model_name=model_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )
        self.max_section_length = max_section_length

    def _get_heading_level(self, line):
        """
        Get the level of a markdown heading.

        Args:
            line (str): The line to check

        Returns:
            int: The heading level (1-6) or 0 if not a heading
        """
        if not line.strip().startswith("#"):
            return 0

        # Count leading # symbols
        level = 0
        for char in line.strip():
            if char == "#":
                level += 1
            else:
                break
        return min(level, 6)  # markdown only supports h1-h6

    def _split_large_section(self, content, max_length):
        """
        Split a large section into smaller chunks while preserving paragraph boundaries.

        Args:
            content (str): The content to split
            max_length (int): Maximum length for each chunk

        Returns:
            list: List of content chunks
        """
        if len(content) <= max_length:
            return [content]

        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max_length, save current chunk
            if current_length + len(paragraph) > max_length and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If a single paragraph is too long, split it at sentence boundaries
            if len(paragraph) > max_length:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if current_length + len(sentence) > max_length and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(sentence)
                    current_length += len(sentence) + 2  # +2 for newline
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2  # +2 for newline

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_by_headings(self, content):
        """
        Split markdown content into chunks based on headings with hierarchy support.

        Args:
            content (str): The markdown content to split

        Returns:
            list: List of tuples containing (heading, content, metadata) triples
        """
        lines = content.split("\n")
        chunks = []
        current_heading = {"text": "Introduction", "level": 1}
        current_content = []
        heading_stack = []  # Track heading hierarchy

        for line in lines:
            level = self._get_heading_level(line)
            if level > 0:
                # Save current content if exists
                if current_content:
                    # Get the full heading hierarchy
                    heading_hierarchy = [
                        h["text"] for h in heading_stack + [current_heading]
                    ]
                    chunks.append(
                        {
                            "heading": current_heading["text"],
                            "content": "\n".join(current_content).strip(),
                            "metadata": {
                                "heading_level": current_heading["level"],
                                "heading_hierarchy": heading_hierarchy,
                                "parent_heading": heading_stack[-1]["text"]
                                if heading_stack
                                else None,
                            },
                        }
                    )
                    current_content = []

                # Update heading hierarchy
                heading_text = line.strip("#").strip()
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()

                current_heading = {"text": heading_text, "level": level}
                if level > 1 and not heading_stack:
                    heading_stack.append({"text": "Introduction", "level": 1})
                elif level > 1:
                    heading_stack.append(current_heading)
            else:
                current_content.append(line)

        # Add the last chunk if there's remaining content
        if current_content:
            heading_hierarchy = [h["text"] for h in heading_stack + [current_heading]]
            chunks.append(
                {
                    "heading": current_heading["text"],
                    "content": "\n".join(current_content).strip(),
                    "metadata": {
                        "heading_level": current_heading["level"],
                        "heading_hierarchy": heading_hierarchy,
                        "parent_heading": heading_stack[-1]["text"]
                        if heading_stack
                        else None,
                    },
                }
            )

        # Process chunks that exceed max length
        final_chunks = []
        for chunk in chunks:
            content_parts = self._split_large_section(
                chunk["content"], self.max_section_length
            )
            for i, part in enumerate(content_parts):
                part_metadata = chunk["metadata"].copy()
                if len(content_parts) > 1:
                    part_metadata["subsection"] = f"Part {i + 1}/{len(content_parts)}"
                final_chunks.append(
                    {
                        "heading": chunk["heading"],
                        "content": part,
                        "metadata": part_metadata,
                    }
                )

        return final_chunks

    def extract_metadata_from_filename(self, filename):
        """
        Extract metadata from a BOCYL filename.

        Args:
            filename (str): The filename to extract metadata from

        Returns:
            dict: Extracted metadata
        """
        # Pattern: BOCYL-D-DDMMYYYY-NN
        pattern = r"BOCYL-D-(\d{2})(\d{2})(\d{4})-(\d+)"
        match = re.match(pattern, filename)

        if match:
            day, month, year, doc_num = match.groups()
            # Create date object
            doc_date = datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y").strftime(
                "%Y-%m-%d"
            )

            return {
                "doc_id": filename,
                "date": doc_date,
                "doc_number": int(doc_num),
                "source": "BOCYL",
                "type": "official_document",
                "area": "AGRARIA",
                "consejeria": "AGRICULTURA Y PESCA",
            }

        return {"doc_id": filename}

    def extract_metadata_from_content(self, content):
        """
        Extract metadata from document content.

        Args:
            content (str): The document content

        Returns:
            dict: Extracted metadata
        """
        # Extract title from markdown header
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown Title"

        return {"title": title}

    def upsert_document(self, doc_path, max_section_length=None):
        """
        Add or update a document in the vector database.

        Args:
            doc_path (str or Path): Path to the document file
            max_section_length (int, optional): Override default max section length

        Returns:
            dict: Information about the upserted document
        """
        if max_section_length:
            self.max_section_length = max_section_length

        doc_path = Path(doc_path)

        # Read document content
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract document ID and filename
        doc_id = doc_path.stem

        # Check if document already exists and delete if it does
        self.delete_document(doc_id)

        # Extract and combine metadata
        filename_metadata = self.extract_metadata_from_filename(doc_id)
        content_metadata = self.extract_metadata_from_content(content)
        document_metadata = {**filename_metadata, **content_metadata}
        document_metadata["doc_id"] = doc_id

        # Split document into chunks based on headings
        chunks = self._split_by_headings(content)

        # Prepare data for batch insertion
        ids = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            ids.append(chunk_id)

            # Add chunk-specific metadata and sanitize all values
            chunk_metadata = _sanitize_metadata(
                {
                    **document_metadata,
                    **chunk["metadata"],
                    "chunk_id": i,
                    "chunk_total": len(chunks),
                    "heading": chunk["heading"],
                }
            )
            metadatas.append(chunk_metadata)
            documents.append(chunk["content"])

        # Add to collection using the vectorstore's add_texts method
        self.vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

        print(
            f"Added {doc_id} with {len(chunks)} sections and metadata: {document_metadata}"
        )

        # Refresh the retriever after adding new documents
        self._setup_retriever()

        return {
            "doc_id": doc_id,
            "sections": len(chunks),
            "metadata": _sanitize_metadata(document_metadata),
            "section_metadata": [
                _sanitize_metadata(m) for m in metadatas
            ],  # Include detailed section info
        }

    def upsert_documents(self, doc_paths, chunk_size=None):
        """
        Add or update multiple documents in the vector database.

        Args:
            doc_paths (list): List of paths to document files
            chunk_size (int, optional): Size of chunks to split documents into.
                If not provided, uses the default chunk size.

        Returns:
            list: Information about each upserted document
        """
        results = []
        for doc_path in doc_paths:
            result = self.upsert_document(doc_path, chunk_size)
            results.append(result)
        return results

    def display_result(self, result):
        """
        Display a search result in a BOCYL-specific format.

        Args:
            result (dict): The search result to display
        """
        doc = result["document"]
        metadata = result["metadata"]
        score = result["similarity_score"]

        print(f"Document: {metadata['doc_id']}")
        print(f"Title: {metadata.get('title', 'No title')}")
        print(f"Date: {metadata.get('date', 'No date')}")
        print(f"Section: {metadata.get('heading', 'No heading')}")
        if metadata.get("heading_hierarchy"):
            print(f"Location: {' > '.join(metadata['heading_hierarchy'])}")
        if metadata.get("subsection"):
            print(f"Subsection: {metadata['subsection']}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Text Preview: {doc.page_content[:150]}...")
        print()


def get_vectordb_bocyl(remove_existing=False, persist_directory=True):
    """
    Get an instance of the BOCYL vector database.

    Args:
        remove_existing (bool): If True, removes the existing database before creating a new one.
            Default is False.
        persist_directory (bool): Whether to persist the database to disk.
            Default is True.

    Returns:
        BOCYLVectorDB: The initialized vector database
    """
    db = BOCYLVectorDB(persist_directory=persist_directory)
    if remove_existing:
        db.reset_collection()
    return db
