import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from searchlite.embedders import tfidf, base
import pprint
import tabulate
from typing import Literal, Optional, Dict, List, Union
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(message)s")
    
class Document():
    def __init__(self, texts:List[str], metadata:List[Dict], embedder:Optional[base.EmbedderProtocol] = None):
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
                
        if embedder is not None and not isinstance(embedder, base.EmbedderProtocol):
            raise TypeError("Embedder must be an instance of EmbedderProtocol or None")

        
        if isinstance(self.embedder, tfidf.SkTFIDFEmbedder):
            if self.embedder.fitted:
                pass
            else:
                logging.info("Fitting SkTFIDFEmbedder to texts...")
                try:
                    self.embedder.fit(self.texts)
                    logging.info("SkTFIDFEmbedder fitted.")
                except Exception as e:
                    logging.exception(f"Encountered an error: {e}")
                    raise
   
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
        import copy
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
            top_texts = [copy.deepcopy(self.texts[idx]) for idx in top_indices] # Have to use deepcopy because attributes are mutable - a normal copy will still push any changes back to original attribute
            top_metadata = [copy.deepcopy(self.metadata[idx]) for idx in top_indices]
            
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
        
    @classmethod
    def from_csv(cls, path:str, text_columns:Union[str, List[str]], metadata_columns:Union[str,List[str]], embedder:Optional[base.EmbedderProtocol] = None) -> "Document":        
        # Read in csv
        df = pd.read_csv(path)
        
        return cls._from_dataframe(df = df, text_columns = text_columns, metadata_columns = metadata_columns, embedder = embedder)
    
    @classmethod
    def from_pandas(cls, df, text_columns:Union[str,List[str]], metadata_columns:Union[str, List[str]], embedder:Optional[base.EmbedderProtocol]):
        return cls._from_dataframe(df = df, text_columns = text_columns, metadata_columns = metadata_columns, embedder = embedder)
    
    @classmethod
    def _from_dataframe(cls, df, text_columns:Union[str,List[str]], metadata_columns:Union[str, List[str]], embedder:Optional[base.EmbedderProtocol]):
        
        try:
            df[text_columns]
        except KeyError as e:
            logger.error(f"One of the column names you passed is invalid: {e}")
        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            
        try:
            df[metadata_columns]
        except KeyError as e:
            logger.error(f"One of the metadata names you passed is invalid: {e}")
        except Exception as e:
            logger.error(f"Something went worng: {e}")
        
        # Check type conformity for text column input
        if isinstance(text_columns, list) and len(text_columns) > 1:
            df["doc_texts"] = df[text_columns].apply(lambda row: " ".join(row.astype(str)), axis = 1)
        else:
            df["doc_texts"] = df[text_columns]

        if isinstance(metadata_columns, str):
            metadata_columns = [metadata_columns]
        
        doc_texts = df["doc_texts"]
        doc_metadata = df[metadata_columns].to_dict(orient = "records")
        
        return cls(texts = doc_texts, metadata = doc_metadata, embedder = embedder)
    
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
        embedder_desc = self.embedder
        return f"Document instance with {len(self.texts)} texts. Metadata contains the following fields: {', '.join(list(self.metadata[0].keys()))}. Embeddings: {embeddings_ready}.\nEmbedder:{embedder_desc}"