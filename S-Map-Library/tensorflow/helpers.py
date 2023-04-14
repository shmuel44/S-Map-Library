import tensorflow_hub as th
import os as os



def set_tensorflow_hub_cache_location(path: str) -> None:
    """Sets the location of the Tensorflow Hub cache directory.

    Args:
        path (str): The path to the cache directory.
    """
    os.environ['TFHUB_CACHE_DIR'] = path
    

def get_model(model_url: str) -> th.KerasLayer:
    """Get the Tensorflow Hub model at the given URL.

    Args:
        model_url (str): The URL of the Tensorflow Hub model.

    Returns:
        th.KerasLayer: The Tensorflow Hub model.
    """
    return th.KerasLayer(model_url)
    