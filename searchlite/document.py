import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from searchlite.embedders import tfidf, base
import pprint
import tabulate
from typing import Literal, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(message)s")
    
class Document():
    def __init__(self, texts:List[str], metadata:List[Dict], embedder:base.EmbedderProtocol = None):
        """_summary_

        Args:
            texts (List[str]): _description_
            metadata (List[Dict]): _description_
            embedder (base.EmbedderProtocol, optional): _description_. Defaults to None.
        """
        
        self.texts = texts.tolist() if not isinstance(texts, list) else texts
        self.metadata = metadata
        self.embeddings = None
        self.embedder = embedder or tfidf.SkTFIDFEmbedder()
        
        if isinstance(self.embedder, tfidf.SkTFIDFEmbedder):
            if self.embedder.fitted:
                pass
            else:
                logging.info("Fitting SkTFIDFEmbedder to texts...")
                try:
                    self.embedder.fit(self.texts)
                    logging.info("SkTFIDFEmbedder fitted.")
                except Exception as e:
                    logging.error(f"Encountered an error: {e}")
   
    def embed(self, in_memory:bool = True, vector_db:str = None) -> Optional[str]:
        """_summary_

        Returns:
            Optional[str]: _description_
        """
        
        if self.embeddings is not None:
            return "Already embedded texts"
        else:
            embeddings = self.embedder.encode(self.texts)
            self.embeddings = embeddings
            return None
    
    def query(self, query_text:str, top_k:int = 3) -> List[Dict]:
        """_summary_

        Args:
            query_text (_type_): _description_
            top_k (int, optional): _description_. Defaults to 3.

        Raises:
            ValueError: _description_

        Returns:
            Dict: _description_
        """
        
        if self.embeddings is None:
            raise ValueError("Text embeddings not found. Please run .embed() before querying.")
        else:
            query_embedding = self.embedder.encode(query_text)
            similarities = cosine_similarity(X = query_embedding.reshape(1,-1), Y = self.embeddings) #Returns ndarray of scores
            
            top_indices = np.argsort(similarities)[0][::-1][0:top_k]
            
            top_scores = [similarities[0][idx] for idx in top_indices]
            top_texts = [self.texts[idx] for idx in top_indices]
            top_metadata = [self.metadata[idx] for idx in top_indices]
            
            output_list_dicts = self._create_output_dict(top_texts, top_metadata, top_scores) 
            
            return output_list_dicts
        
    def display_results(self, output_list_dicts:List[Dict], style:Literal["f-string", "pprint", "tabulate"]) -> None:
        """_summary_

        Args:
            output_list_dicts (list[dict]): _description_
            style (Literal[&quot;f): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if style.lower() == "f-string":
            for i, dictionary in enumerate(output_list_dicts, 1):
                print(f"Result {i}:")
                for key, value in dictionary.items():
                    print(f"    {key}: {value}")
                print()
            return None
        
        elif style.lower() == "pprint":
            pprint.pprint(output_list_dicts)
            return None
            
        elif style.lower() == "tabulate":
            headers = output_list_dicts[0].keys()
            rows = [dictionary.values() for dictionary in output_list_dicts]
            print(tabulate.tabulate(rows, headers = headers, tablefmt = "grid"))
            return None
        
        else:
            _error_opt_list = ", ".join(["f-string", "pprint", "tabulate"])
            raise ValueError(f"Expected options to be one of {_error_opt_list}")

    def _create_output_dict(self, top_texts:List[str], top_metadata:Dict, top_scores:List[float]) -> List[Dict]:
        """_summary_

        Args:
            top_texts (List[str]): _description_
            top_metadata (Dict): _description_

        Returns:
            List[Dict]: _description_
        """
        
        final_output = top_metadata
        for dictionary in final_output:
            idx = top_metadata.index(dictionary)
            dictionary["text"] = top_texts[idx]
            dictionary["similarity score"] = top_scores[idx]
            
        return final_output
    
    def __repr__(self):
        embeddings_ready = "Ready" if self.embeddings is not None else "Not Ready"
        return f"Document instance with {len(self.texts)} texts. Metadata contains the following fields: {', '.join(list(self.metadata[0].keys()))}. Embeddings: {embeddings_ready}."