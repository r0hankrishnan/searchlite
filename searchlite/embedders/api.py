import numpy as np
from typing import Union, List
from searchlite.embedders.base import ApiEmbedFunction
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(message)s")

class ApiEmbedder:
    def __init__(self, client, embed_func:ApiEmbedFunction):
        """_summary_

        Args:
            client (_type_): _description_
            embed_func (ApiEmbedFunction): _description_

        Raises:
            ValueError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        
        self.client = client
        self.embed_func = embed_func
        
        
        try:
            logger.info("Validating embed_func...")
            str_dummy = "test"
            list_str_dummy = ["test", "exam", "homework"]
            res_t1 = embed_func(str_dummy)
            res_t2 = embed_func(list_str_dummy)
            
            if not isinstance(res_t1, np.ndarray) or not isinstance(res_t2, np.ndarray):
                raise ValueError("embed_func must return a numpy ndarray when embedding a string or list of strings")

            logger.info("embed_func validated")
            
        except Exception as e:
            raise TypeError(f"Provided embed_func is not valid: {e}")
        
    def encode(self, texts:Union[List[str], str]):
        """_summary_

        Args:
            texts (Union[List[str], str]): _description_

        Returns:
            _type_: _description_
        """
        
        return self.embed_func(texts)
    
    def __repr__(self):
        return f"Api Embedder Object with client: {self.client}."