import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pickle
from annoy import AnnoyIndex
from sklearn.metrics import pairwise_distances
from typing import Literal, List, Dict, Any, Union, Tuple
from enum import Enum

from . import feature as sfe



class MetricKind(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    COSINE = 'cosine'
    ANGULAR = 'angular'


class ProductEmbeddingsManager:
    def __init__(
            self, 
            dimension: int,
            metric: MetricKind,
            metadata_columns: List[str]) -> None:

        self.products = {}
        self.dimension = dimension
        self.metric = metric
        self.annoy_metric = metric if metric != 'cosine' else 'angular'
        self.metadata_columns = metadata_columns
        self.create_index()

    def create_index(self):
        # Rebuild the Annoy index and mappings
        self.embeddings_index = AnnoyIndex(self.dimension, metric=self.annoy_metric)
        self.product_index_map = {}
        self.index_product_map = {}
        self.current_index = 0

    def add_product(
            self, 
            product_id: str, 
            embedding: np.ndarray, 
            metadata: Dict[str, Any]):
        
        self.products[product_id] = {'embedding': embedding, 'metadata': metadata}
        self.embeddings_index.add_item(self.current_index, embedding)
        self.product_index_map[product_id] = self.current_index
        self.index_product_map[self.current_index] = product_id
        self.current_index += 1

    def build_index(
            self, 
            n_trees:int=10):
        self.embeddings_index.build(n_trees)

    def remove_product(self, product_id):
        if product_id in self.products:
            del self.products[product_id]
            index = self.product_index_map[product_id]
            del self.product_index_map[product_id]
            del self.index_product_map[index]

    def get_product(self, product_id):
        return self.products.get(product_id, None)

    def save(self, filename):
        if filename is None:
            raise ValueError("No filename provided for saving.")

        with open(filename, 'wb') as f:
            pickle.dump(self.products, f)

    def load(self, filename):
        if filename is None:
            raise ValueError("No filename provided for loading.")

        with open(filename, 'rb') as f:
            self.products = pickle.load(f)

        self.create_index()

        for product_id, product_data in self.products.items():
            self.add_product(product_id, product_data['embedding'], product_data['metadata'])

        self.build_index()

    def search_similar_products(self, query_product_id, top_k=5, include_metadata=False):
        if query_product_id not in self.products:
            raise ValueError(f"Product ID '{query_product_id}' not found in the manager.")

        query_embedding = self.products[query_product_id]['embedding']
        indices, distances = self.embeddings_index.get_nns_by_vector(query_embedding, top_k + 1, include_distances=True)

        # Remove the query product from the results
        results = [(self.index_product_map[index], distance) for index, distance in zip(indices, distances) if index != self.product_index_map[query_product_id]]

        if include_metadata:
            results = [(product_id, distance, self.products[product_id]['metadata']) for product_id, distance in results]

        return results

    def search_similar_products_by_metadata(self, query_metadata, metadata_weight=0.5, top_k=5, include_metadata=False):

        product_ids = []
        similarities = []

        for product_id, product_data in self.products.items():
            metadata_similarity_score = self.metadata_similarity(query_metadata, product_data['metadata'])
            embedding_similarity = 1 - cdist(query_metadata['embedding'].reshape(1, -1), product_data['embedding'].reshape(1, -1), metric='cosine').flatten()[0]
            combined_similarity = metadata_weight * metadata_similarity_score + (1 - metadata_weight) * embedding_similarity
            product_ids.append(product_id)
            similarities.append(combined_similarity)

        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(product_ids[index], similarities[index]) for index in sorted_indices]

        if include_metadata:
            results = [(product_id, similarity, self.products[product_id]['metadata']) for product_id, similarity in results]

        return results
    
    def search_similar_products_combined(self, query_product_id, top_k=5, alpha=0.5, search_k=-1):
        if query_product_id not in self.products:
            raise ValueError(f"Product ID '{query_product_id}' not found in the manager.")

        query_embedding = self.products[query_product_id]['embedding']
        query_metadata = self.products[query_product_id]['metadata']

        # Get nearest neighbors using Annoy index
        query_index = self.product_index_map[query_product_id]
        indices, distances = self.embeddings_index.get_nns_by_vector(query_embedding, self.embeddings_index.get_n_items(), include_distances=True, search_k=search_k)

        similarities = []
        for index, distance in zip(indices, distances):
            if index == query_index:
                continue

            product_id = self.index_product_map[index]
            product_data = self.products[product_id]

            if self.metric == 'angular':
                # Convert angular distance to cosine similarity
                embedding_similarity = 1 - 2 * distance
            elif self.metric in ['euclidean', 'manhattan', 'hamming']:
                embedding_similarity = 1 - distance
            else:  # 'dot' or 'cosine' (Annoy internally uses 'angular' for 'cosine')
                embedding_similarity = distance

            metadata_similarity = self.metadata_similarity(query_metadata, product_data['metadata'])

            combined_similarity = alpha * embedding_similarity + (1 - alpha) * metadata_similarity
            similarities.append((product_id, combined_similarity))

        # Sort by descending similarity and return the top_k results
        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return results


    def search_similar_products_filtered(self, query_product_id, filter_metadata, top_k=5, search_k=-1):
        if query_product_id not in self.products:
            raise ValueError(f"Product ID '{query_product_id}' not found in the manager.")

        query_embedding = self.products[query_product_id]['embedding']

        # Get nearest neighbors using Annoy index
        query_index = self.product_index_map[query_product_id]
        indices, distances = self.embeddings_index.get_nns_by_vector(query_embedding, self.embeddings_index.get_n_items(), include_distances=True, search_k=search_k)

        similarities = []
        for index, distance in zip(indices, distances):
            if index == query_index:
                continue

            product_id = self.index_product_map[index]
            product_data = self.products[product_id]

            # Filter products based on the filter_metadata
            if not all(product_data['metadata'][key] == value for key, value in filter_metadata.items()):
                continue

            embedding_similarity = 1 - distance
            similarities.append((product_id, embedding_similarity))

        # Sort by descending similarity and return the top_k results
        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return results

    def metadata_similarity(self, metadata1, metadata2):
        """This function computes the similarity between two product's metadata
        based on the one-hot encoding representation of the metadata attributes.

        Args:
            metadata1 (dict): the metadata of the first product
            metadata2 (dict): the metadata of the second product

        Returns:
            float: the similarity between the two set of metadata
        """
        # Create a DataFrame with the metadata
        metadata_df = pd.DataFrame([metadata1, metadata2])

        # Create one-hot encoded vectors
        one_hot_encoded_df = pd.get_dummies(
            data=metadata_df, 
            columns=self.metadata_columns)

        # Compute pairwise distances between the one-hot encoded vectors
        distance_matrix = pairwise_distances(one_hot_encoded_df, metric=self.metric)

        # Convert distance to similarity depending on the metric
        if self.metric in ['euclidean', 'manhattan', 'hamming']:
            similarity = 1 - distance_matrix[0, 1]
        elif self.metric == 'angular':
            similarity = 1 - (2 * np.arccos(distance_matrix[0, 1]) / np.pi)
        else:  # 'dot' or 'cosine'
            similarity = 1 - distance_matrix[0, 1]

        return similarity



