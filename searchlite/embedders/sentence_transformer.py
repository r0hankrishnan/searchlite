import numpy as np
from typing import Union, List

class SentenceTransformerEmbedder:
    def __init__(self, model_name:str) -> None:
        """_summary_

        Args:
            model_name (str): _description_
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The `sentence-transformers` package is required to use SentenceTransformerEmbedder."
                "Please install it with: pip install searchlite[sentence_transformers]"
                )

        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts:Union[List[str], str]) -> np.ndarray:
        """_summary_

        Args:
            texts (Union[List[str], str]): _description_

        Returns:
            np.ndarray: _description_
        """
        
        return self.model.encode(texts)
    
    def __repr__(self):
        st_message = self.model.__repr__()
        message = f"This embedder is a SentenceTransformer instance in a wrapper.\nSentence Transformer __repr__: {st_message}"
        return message
    
    