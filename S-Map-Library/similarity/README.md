# Product Embeddings Manager

## Overview

This Python class manages product embeddings, which are numpy arrays of a given size (usually around 2048 dimensions). It also handles metadata for each product, such as text, categorical, or numerical attributes. This class allows users to manage, save, and reload product embeddings and metadata for a large number of products. Additionally, it provides a way to search for similar products based on embeddings distance and/or metadata similarities.

## Class Definition

### `ProductEmbeddingsManager`

#### Attributes

- `products`: A dictionary containing product IDs as keys and corresponding product embeddings and metadata as values.
- `filename`: The file path for saving and loading the product embeddings and metadata.

#### Methods

- `__init__(self, filename=None)`: Initializes a new instance of the `ProductEmbeddingsManager` class. Optionally accepts a filename for loading existing product embeddings and metadata.
- `add_product(self, product_id, embedding, metadata)`: Adds a product with the specified ID, embedding, and metadata to the manager.
- `remove_product(self, product_id)`: Removes the product with the specified ID from the manager.
- `get_product(self, product_id)`: Retrieves the product with the specified ID from the manager.
- `save(self, filename=None)`: Saves the product embeddings and metadata to a file. If a filename is provided, it will overwrite the existing filename attribute.
- `load(self, filename=None)`: Loads product embeddings and metadata from a file. If a filename is provided, it will overwrite the existing filename attribute.
- `search_similar_products(self, query_product_id, metric='cosine', top_k=5, include_metadata=False)`: Searches for similar products based on embeddings distance or metadata similarities. Returns the top K similar products as a list of tuples containing product IDs and similarity scores. Optionally includes metadata in the results.
- `search_similar_products_by_metadata(self, query_metadata, metadata_weight=0.5, top_k=5, include_metadata=False)`: Searches for similar products based on a combination of embeddings distance and metadata similarities. Returns the top K similar products as a list of tuples containing product IDs and similarity scores. Optionally includes metadata in the results.

## Usage Example

```python
from product_embeddings_manager import ProductEmbeddingsManager

# Initialize the manager
manager = ProductEmbeddingsManager()

# Add some products
manager.add_product("product_1", embedding_1, metadata_1)
manager.add_product("product_2", embedding_2, metadata_2)

# Save the products to a file
manager.save("products.pkl")

# Load the products from a file
manager.load("products.pkl")

# Search for similar products
similar_products = manager.search_similar_products("product_1", top_k=3)

# Search for similar products based on metadata
similar_products_by_metadata = manager.search_similar_products_by_metadata(
    query_metadata, metadata_weight=0.5, top_k=3
)
```

## Hyperparameters Optimization

In the `ProductEmbeddingsManager` class, there are several variables and arguments that can be considered hyperparameters. These hyperparameters can impact the performance and accuracy of the nearest neighbor search.

Here's a summary of the hyperparameters:

1. `metric`: The distance metric used in the Annoy index for comparing embeddings. Possible values include 'angular', 'euclidean', 'manhattan', 'hamming', and 'dot'.
2. `n_trees`: The number of trees built in the Annoy index, which impacts the search performance and accuracy. A higher value usually leads to better accuracy but slower build times.
3. `search_k`: The number of nodes explored in the Annoy index while searching for nearest neighbors. A higher value results in better accuracy but slower search times.
4. `alpha`: The weight assigned to the embedding similarity when combining it with metadata similarity in the `search_similar_products_combined` method.
5. The similarity measure used for metadata similarity (e.g., Jaccard similarity, Jaro-Winkler similarity, or cosine similarity with one-hot encoding).

To tune these hyperparameters, you can follow these steps:

1. **Prepare a validation dataset**: Split your dataset into training and validation sets. Use the training set to build the `ProductEmbeddingsManager` and the validation set to evaluate its performance.

2. **Define a performance metric**: Choose a performance metric to evaluate the effectiveness of the nearest neighbor search. This could be a measure like mean average precision (MAP), recall@k, or a custom metric relevant to your use case.

3. **Perform a grid search or random search**: For each hyperparameter, define a range of possible values. Use a grid search or random search approach to explore different combinations of these hyperparameter values. For each combination, evaluate the performance of the `ProductEmbeddingsManager` on the validation dataset using the chosen performance metric.

4. **Select the best combination**: Choose the hyperparameter combination that results in the best performance on the validation dataset.

5. **Evaluate on the test dataset**: Finally, evaluate the performance of the `ProductEmbeddingsManager` with the selected hyperparameter combination on a test dataset to get an unbiased estimate of its performance.

For a more advanced approach, you can use Bayesian optimization libraries like `hyperopt` or `optuna` to efficiently search for the best hyperparameter values. These libraries often require less computational resources compared to grid search or random search, especially when dealing with a large number of hyperparameters.