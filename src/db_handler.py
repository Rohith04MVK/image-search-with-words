import uuid
import warnings
import chromadb
from chromadb.utils import embedding_functions


from .config import CHROMADB_PATH, EMBEDDING_MODEL

warnings.filterwarnings("ignore")


class ChromaDBHandler:
    """A class to handle saving and querying image captions in ChromaDB."""

    def __init__(self, db_path: str = CHROMADB_PATH, model_name: str = EMBEDDING_MODEL):
        """
        Initializes the ChromaDBHandler with the given database path and model name.

        Args:
            db_path (str): The path to the ChromaDB storage.
            model_name (str): The name of the model to use for embedding functions. Default is 'all-MiniLM-L12-v2'.
        """
        self.db_path = db_path
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name='image_collection', embedding_function=self.embedding_function, metadata={"hnsw:space": "cosine"}
        )

    def save_to_chroma_db(self, caption: str, image_path: str) -> None:
        """
        Saves the image caption to ChromaDB.

        Args:
            caption (str): The caption of the image.
            image_path (str): The path to the image file.
        """
        doc_id = str(uuid.uuid4())

        # Add the document to the collection
        self.collection.add(ids=[doc_id], documents=[caption], metadatas={
                            "image_path": image_path})

    def query_chroma_db(self, caption: str):
        """
        Queries the ChromaDB for similar captions.

        Args:
            caption (str): The caption to query.

        Returns:
            The query results from ChromaDB.
        """
        # Query the collection
        results = self.collection.query(query_texts=[caption])
        return results


db_handler = ChromaDBHandler()


def get_handler():
    return db_handler
