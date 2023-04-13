import numpy as np
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional

import lib_common as lco


class PineconeIndexElement:
    def __init__(self, key: str, vector: Union[np.ndarray, List[float]], metadata: Dict[str, Any] = None):
        """
        Creates a new PineconeIndexElement.

        Args:
            key (str): The unique identifier for the element.
            vector (Union[np.ndarray, List[float]]): The vector representation of the element.
            metadata (Dict[str, Any], optional): Any additional metadata associated with the element. Defaults to None.
        """
        self.key = key
        self.vector = np.array(vector)
        self.metadata = {} if metadata is None else metadata

    def get_element(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the element.

        Returns:
            Dict[str, Any]: A dictionary containing the key, vector, and metadata of the element.
        """
        return {
            'id': self.key,
            'values': self.vector.tolist(),
            'metadata': lco.cleanup_dict(self.metadata)
        }
