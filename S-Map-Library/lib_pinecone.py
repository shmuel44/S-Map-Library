import pinecone as pc
import pandas as pd
import json as js
import dataiku as di
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional

import lib_common as lco
import cls_pinecone as cpi


def initialize_session(api_key: str, environment: str) -> None:
    """
    Initializes the Pinecone session with the given API key and environment.

    Args:
        api_key (str): The Pinecone API key.
        environment (str): The Pinecone environment to use.
    """
    pc.init(
        api_key=api_key, 
        environment=environment)
    
def get_or_create_index(name: str, dimension: int, metric: str, pod_type: str) -> pc.Index:
    """
    Returns the Pinecone index with the given name, or creates it if it does not exist.

    Args:
        name (str): The name of the index.
        dimension (int): The dimensionality of the vectors in the index.
        metric (str): The distance metric to use for the index.
        pod_type (str): The type of pod to use for the index.

    Returns:
        pc.Index: The Pinecone index.
    """
    if name not in pc.list_indexes():
        pc.create_index(
            name=name, 
            dimension=dimension, 
            metric=metric, 
            pod_type=pod_type)
        
    return pc.Index(index_name=name)

def validate_keys(index: pc.Index, namespace: str, keys: List[str]) -> List[str]:
    """
    Checks if the given keys are defined in the index.

    Args:
        index (pc.Index): The Pinecone index.
        namespace (str): The namespace to use.
        key (List[str]): The keys to check.

    Returns:
        List[str]: The list of valid keys.
    """
    return index.fetch(keys, namespace).vectors.keys()
        

def upsert_elements(index: pc.Index, namespace: str, elements: List[cpi.PineconeIndexElement]) -> None:
    """
    Upserts the given elements into the Pinecone index.

    Args:
        index (pc.Index): The Pinecone index.
        namespace (str): The namespace to use.
        elements (List[cpi.PineconeIndexElement]): The Pinecone index elements to upsert.
    """
    vectors = [e.get_element() for e in elements]
    index.upsert(vectors=vectors, namespace=namespace)
