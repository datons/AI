import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import chromadb
from chromadb.config import Settings
import shutil
import os


class BaseVectorDB:
    """Base class for managing ChromaDB collections."""

    def __init__(
        self,
        db_path,
        collection_name,
        model_name=None,
        embedding_function=None,
        persist_directory=True,
    ):
        """
        Initialize connection to ChromaDB and setup the collection.

        Args:
            db_path (str): Path to the ChromaDB database
            collection_name (str): Name of the collection to use
            model_name (str, optional): Name of the HuggingFace model to use for embeddings.
                Not used if embedding_function is provided.
            embedding_function: Custom embedding function to use. If not provided,
                will create one using model_name.
            persist_directory (bool): Whether to persist the database to disk.
                If False, will use in-memory database.
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Setup embedding function
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            model_name = model_name or "BAAI/bge-small-en-v1.5"
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )

        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # Initialize retriever with default settings
        self._setup_retriever()

    def _ensure_db_directory(self):
        """
        Ensure the database directory exists and has proper permissions.
        """
        # Create parent directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the database directory if it doesn't exist
        self.db_path.mkdir(exist_ok=True)

        # Set directory permissions to allow read/write
        os.chmod(self.db_path, 0o777)

        # Set permissions for all parent directories
        current = self.db_path
        while current != current.parent:
            try:
                os.chmod(current, 0o777)
            except PermissionError:
                print(f"Warning: Could not set permissions for {current}")
            current = current.parent

    def _initialize_vectorstore(self):
        """
        Initialize the vector store with the specified collection.

        Returns:
            Chroma: Initialized vector store
        """
        # Initialize ChromaDB client
        if self.persist_directory:
            # Persistent mode
            settings = Settings(
                persist_directory=str(self.db_path),
                is_persistent=True,
                anonymized_telemetry=False,
            )
            client = chromadb.PersistentClient(
                path=str(self.db_path), settings=settings
            )
        else:
            # In-memory mode
            client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Create or get collection
        collection = client.get_or_create_collection(name=self.collection_name)

        # Initialize Langchain's Chroma wrapper
        return Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def reset_collection(self):
        """
        Reset the collection by removing all data.
        """
        if self.persist_directory:
            try:
                # Get the collection and delete it
                collection = self.vectorstore._client.get_collection(
                    self.collection_name
                )
                collection.delete()
                print(f"Deleted collection {self.collection_name}")
            except Exception as e:
                print(f"Error deleting collection: {e}")

        # Reinitialize the vector store
        self.vectorstore = self._initialize_vectorstore()

    def _setup_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Setup the retriever with the specified parameters.

        Args:
            search_type (str): Type of search to perform. Default is "similarity"
            search_kwargs (dict): Additional search parameters
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # Default to returning top 4 results

        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Get this vector store as a retriever.

        Args:
            search_type (str): Type of search to perform
            search_kwargs (dict): Additional search parameters

        Returns:
            Retriever: The configured retriever
        """
        self._setup_retriever(search_type, search_kwargs)
        return self.retriever

    def query(self, query_text, n_results=3, metadata_filter=None):
        """
        Base query method to search the vector database.

        Args:
            query_text (str): The query text
            n_results (int): Number of results to return
            metadata_filter (dict): Optional filter for metadata fields

        Returns:
            dict: Dictionary containing formatted results
        """
        # Prepare query parameters
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results,
        }

        # Add metadata filter if provided
        if metadata_filter:
            query_params["where"] = metadata_filter

        # Run the query
        search_results = self.vectorstore.similarity_search_with_relevance_scores(
            **query_params
        )

        # Format results
        formatted_results = []
        for doc, score in search_results:
            result = {
                "document": doc,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
            formatted_results.append(result)

        # Refresh retriever after any query operation
        self._setup_retriever()

        return {
            "query": query_text,
            "results": formatted_results,
        }

    def delete_document(self, doc_id):
        """
        Delete a document and all its chunks from the vector database.

        Args:
            doc_id (str): The document ID to delete

        Returns:
            bool: True if document was deleted, False if not found
        """
        try:
            collection = self.vectorstore._client.get_collection(self.collection_name)
            results = collection.get(where={"doc_id": doc_id})
            if results and len(results["ids"]) > 0:
                collection.delete(ids=results["ids"])
                print(f"Deleted document {doc_id} with {len(results['ids'])} chunks")
                return True
            print(f"Document {doc_id} not found")
            return False
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def get_document_metadata(self, doc_id):
        """
        Get metadata for a specific document.

        Args:
            doc_id (str): The document ID to get metadata for

        Returns:
            dict: Document metadata or None if not found
        """
        try:
            collection = self.vectorstore._client.get_collection(self.collection_name)
            results = collection.get(where={"doc_id": doc_id}, limit=1)
            if results and len(results["metadatas"]) > 0:
                return results["metadatas"][0]
            return None
        except Exception as e:
            print(f"Error getting document metadata: {e}")
            return None
