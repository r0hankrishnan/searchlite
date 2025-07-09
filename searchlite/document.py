from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pprint
import tabulate
from typing import Literal, Union


class Document():
    def __init__(self, texts:list, metadata:list[dict], model:str = None):
        self.texts = texts.tolist() if not isinstance(texts, list) else texts
        self.metadata = metadata
        self.embeddings = None
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
   
    def embed(self):
        if self.embeddings is not None:
            return "Already embedded texts"
        else:
            embeddings = self.model.encode(self.texts)
            self.embeddings = embeddings
    
    def query(self, query_text, top_k = 3):
        if self.embeddings is None:
            raise ValueError("Text embeddings not found. Please run .embed() before querying.")
        else:
            query_embedding = self.model.encode(query_text)
            similarities = cosine_similarity(X = query_embedding.reshape(1,-1), Y = self.embeddings) #Returns ndarray of scores
            
            top_indices = np.argsort(similarities)[0][::-1][0:top_k]
            
            top_texts = [self.texts[idx] for idx in top_indices]
            top_metadata = [self.metadata[idx] for idx in top_indices]
            
            output_list_dicts = self._create_output_dict(top_texts, top_metadata) 
            
            return output_list_dicts
        
    def display_results(self, output_list_dicts:list[dict], style:Literal["f-string", "pprint", "tabulate"]):
        _error_opt_list = ", ".join(["f-string", "pprint", "tabulate"])
        if style.lower() == "f-string":
            for dictionary in output_list_dicts:
                for key,value in dictionary.items():
                    print(f"{key}: {value}", end=" | ")
                print(" ")
        
        elif style.lower() == "pprint":
            pprint.pprint(output_list_dicts)
            
        elif style.lower() == "tabulate":
            headers = output_list_dicts[0].keys()
            rows = [dictionary.values() for dictionary in output_list_dicts]
            print(tabulate.tabulate(rows, headers = headers, tablefmt = "grid"))
            
        else:
            raise ValueError(f"Expected options to be one of {_error_opt_list}")
        
    def _create_output_dict(self, top_texts, top_metadata):
        final_output = top_metadata
        for dictionary in final_output:
            idx = top_metadata.index(dictionary)
            dictionary["text"] = top_texts[idx]
            
        return final_output
    
    def __repr__(self):
        embeddings_ready = "Ready" if self.embeddings is not None else "Not Ready"
        return f"Document instance with {len(self.texts)} texts. Metadata contains the following fields: {', '.join(list(self.metadata[0].keys()))}. Embeddings: {embeddings_ready}."