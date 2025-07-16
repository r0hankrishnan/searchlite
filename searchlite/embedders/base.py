from typing import Protocol, List, Union


class EmbedderProtocol(Protocol):
    """_summary_

    Args:
        Protocol (_type_): _description_
    """
    
    def encode(self, texts: Union[List[str], str]) -> List[float]: ...