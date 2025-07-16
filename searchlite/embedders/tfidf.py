import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union, List, Dict

class SkTFIDFEmbedder:
    def __init__(self, **vectorizer_kwargs):
        """_summary_
        """
        
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self.fitted = False
        
    def fit(self, corpus:List[str]) -> None:
        """_summary_

        Args:
            corpus (List[str]): _description_
        """
        
        self.vectorizer = self.vectorizer.fit(corpus)
        self.fitted = True
        
    def encode(self, texts:Union[List[str], str], toarray:bool = True) -> np.ndarray:
        """_summary_

        Args:
            texts (Union[List[str], str]): Text(s) to embed.
            toarray (bool, optional): Boolean to tell method whether or not to convert ouput to dense array. Defaults to True.

        Raises:
            ValueError: If .fit() hasn't been run, .encode() will throw an error.

        Returns:
            np.ndarray: (By default) a dense numpy array of embeddings. Its dimension is equal to the length of the corpus (I think).
        """
        
        if self.fitted:
            if isinstance(texts, str): # If passing just a string, need to pass in a list for correct output format
                vectors = self.vectorizer.transform([texts])
            else:
                vectors = self.vectorizer.transform(texts)
            
            if toarray:
                return vectors.toarray()
            else: 
                return vectors
        else:
            raise ValueError("TFIDF embedder must be fit before creating embeddings. Please run the .fit() method on your text corpus before continuing.")
    
    def __repr__(self):
        embedder_status = "True" if self.fitted else "False, please run .fit() before trying to embed."
        message = f"TFIDFEmbedder object implemented using scikit-learn.\n Embedder fitted: {embedder_status}"
        return message
        
class FromScratchTFIDFEmbedder:

    def __init__(self):
        self.term_frequency: Dict[str:str]= None
        self.idf = None
        self.fitted = False
        
    def fit(self, corpus):
        ...
        
    def encode(self, texts):
        ...