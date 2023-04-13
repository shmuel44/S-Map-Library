import tensorflow as tf
import tensorflow_hub as th
import os as os


def set_tensorflow_hub_cache_location(path):
    os.environ['TFHUB_CACHE_DIR'] = path
    
def get_model(model_url):
    return th.KerasLayer(model_url)
    