from typing import Protocol, List, Union
import numpy as np

class EmbedderProtocol(Protocol):
    """_summary_

    Args:
        Protocol (_type_): _description_
    """
    
    def encode(self, texts: Union[List[str], str]) -> List[float]: 
        ...
    
class ApiEmbedFunction(Protocol):
    def __call__(self, texts: Union[List[str], str]) -> np.ndarray:
        ...