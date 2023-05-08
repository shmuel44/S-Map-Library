import pandas as pd
import os as os
import pickle
from annoy import AnnoyIndex as ann
from typing import List, Dict, Any
from enum import Enum

from . import feature as sfe



class MetricKind(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    COSINE = 'cosine'
    ANGULAR = 'angular'

class EmbeddingsManagerFileExtensions(Enum):
    PRODUCTS = '.products'
    ANNOY = '.annoy'

class ProductEmbeddingsManager:
    def __init__(
            self, 
            metric: MetricKind,
            feature_set: sfe.FeatureSet,
            n_trees: int=25) -> None:

        self.metric = metric
        self.annoy_metric = metric if metric != 'cosine' else 'angular'
        self.feature_set = feature_set
        self.dimension = feature_set.get_row_embedding_dimension()
        self.n_trees = n_trees

        self.create_index()

    def _clear_index(self):
        if self.embeddings_index is not None:
            self.embeddings_index.unload()

        self.embeddings_index = None
        self.products_from_key = {}
        self.products_from_id = {}

    def create_index(self):
        self._clear_index()

        self.embeddings_index = ann.AnnoyIndex(
            f=self.dimension, 
            metric=self.annoy_metric)

    def load_from_dataframe(
            self,
            dataframe: pd.DataFrame,
            make_final=True) -> None:
        
        self.create_index()

        for row_data in dataframe.itertuples(index=True):
            self.add_product(
                row_data=row_data._asdict())
            
        if make_final:
            self._build_index()

    def add_product(
            self, 
            row_data: Dict[str, Any]) -> int:

        product_id = len(self.products_from_key)
        product_key = self.feature_set.get_row_key(row_data)
        product_embedding = self.feature_set.get_row_embedding(row_data)
        product_metadata = self.feature_set.get_row_metadata(row_data)
        product_output = self.feature_set.get_row_output(row_data)

        product = {
            'index': product_id,
            'key': product_key,
            'embedding': product_embedding, 
            'metadata': product_metadata,
            'output': product_output
        }
        
        self.products_from_key[product_key] = product
        self.products_from_id[product_id] = product

        self.embeddings_index.add_item(
            product_id, 
            product_embedding)

        return product_id
       
    def _build_index(self) -> None:
        '''Builds the Annoy index.
        Builds a forest of n_trees trees. More trees gives higher precision when querying. 
        After calling build, no more items can be added. 
        '''
        self.embeddings_index.build(
            n_trees=self.n_trees)

    def search_similar_products(
            self, 
            product_key: Any, 
            top_k: int=5) -> List[Any]:
        '''Searches for the top_k most similar products to the given product_key.
        Args:
            product_key (Any): The key of the product to search for similar neighbors.
            top_k (int): The number of similar products to return.
        Returns:
            List[Any]: A list of the top_k most similar products.
        '''
        if product_key not in self.products_from_key:
            raise ValueError('Product key "{0}" not found.'.format(product_key))

        product = self.products_from_key[product_key]
        product_embedding = product['embedding']

        indices, distances = self.embeddings_index.get_nns_by_vector(
            v=product_embedding, 
            n=top_k + 1, 
            include_distances=True)

        # Remove the query product from the results
        return [
            (self.products_from_id[index], distance) 
            for index, distance in zip(indices, distances) 
            if index != product['index']]

    def save(
            self, 
            filename: str,
            save_index: bool=False) -> None:
        
        if filename is None:
            raise ValueError("No filename provided for saving.")

        products_filename = filename + EmbeddingsManagerFileExtensions.PRODUCTS.value
        annoy_filename = filename + EmbeddingsManagerFileExtensions.ANNOY.value

        with open(products_filename, 'wb') as f:
            pickle.dump(self.products_from_key, f)

        if save_index:
            '''After saving, no more items can be added.'''
            self.embeddings_index.save(
                fn=annoy_filename)

    def load(
            self,
            filename: str,
            keep_online: bool=False) -> None:

        if filename is None:
            raise ValueError("No filename provided for loading.")

        products_filename = filename + EmbeddingsManagerFileExtensions.PRODUCTS.value
        
        with open(products_filename, 'rb') as f:
            self.products_from_key = pickle.load(f)

        self.products_from_id = {
            product['index']: product 
            for product in self.products_from_key.values()
        }

        self.create_index()
        loaded = False

        if not keep_online:
            annoy_filename = filename + EmbeddingsManagerFileExtensions.ANNOY.value

            if os.path.exists(annoy_filename):
                self.embeddings_index.load(annoy_filename)
                loaded = True

        if not loaded:
            for product in self.products_from_key.values():
                self.embeddings_index.add_item(
                    product['index'], 
                    product['embedding'])

            if not keep_online:
                self._build_index()





