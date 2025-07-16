import numpy as np
import ollama
from typing import Union, List
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(message)s")

class OllamaEmbedder:
    def __init__(self, model_name:str):
        """_summary_

        Args:
            model_name (str): _description_

        Raises:
            RuntimeError: _description_
        """
        
        self.model_name = model_name
        self.ollama_server_running = self._check_ollama_server()
        
        if not self.ollama_server_running:
            raise RuntimeError("Ollama server is not running. Please start it before embedding.\nYou can start an Ollama server by launching the Ollama application and running 'ollama serve' in the terminal.")

    def encode(self, texts:Union[List[str], str]) -> Union[List[np.ndarray], np.ndarray]:
        """_summary_

        Args:
            texts (Union[List[str], str]): _description_

        Raises:
            TypeError: _description_

        Returns:
            Union[List[np.ndarray], np.ndarray]: _description_
        """
        
        if isinstance(texts, list):
            embeddings = []
            for idx, doc in enumerate(texts):
                response = ollama.embed(model = self.model_name, input = doc)
                embedding = response["embeddings"]
                embedding = np.squeeze(embedding, axis = 0)
                embeddings.append(embedding)
            
            return embeddings
                
        elif isinstance(texts, str): # Need a different encode implementation if user is just passing query string
            query_response = ollama.embed(model = self.model_name, input = texts)
            query_embedding = np.array(query_response["embeddings"], dtype = float)
            return query_embedding
        
        else: # Add in error handling on the off chance that self.texts is neither List[str] or str
            raise TypeError(f"Expected a list of strings or a string, got {type(texts).__name__}")
        
    def _check_ollama_server(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        
        test_url = "http://localhost:11434/api/tags"
        try:
            response = requests.get(test_url, timeout=1)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"Ollama server responded with status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("Ollama server not running or unreachable.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Ollama server connection timed out.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama server: {e}")
            return False

        
    def __repr__(self):
        status = "Running" if self.ollama_server_running else "Not running. Please launch Ollama and ensure API endpoint is running before embedding."
        message = f"Ollama Embedder object. Chosen model: {self.model_name}. \nOllama server status: {status}"
        return message