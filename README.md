# Semantic Search Engine with Vectorized Databases
This repository contains the code and documentation for a simple semantic search engine with vectorized databases and the evaluation of its performance. The project focuses on building an efficient indexing system to retrieve information based on vector space embeddings.

## Project Overview

The key components of the project include:
- `VecDB`: A class representing the vectorized database, responsible for storing and retrieving vectors.
- `insert_records()`: A method to insert vectors into the database.
- `retrieve()`: A method to retrieve the top-k most similar based on a given query vector.
- `_cal_score()`: A helper method to calculate the cosine similarity between two vectors.
- `_build_index()`: A placeholder method for implementing an indexing mechanism.

## Getting Started

To get started with the project, follow these steps:
1. Clone the repository to your local machine.
2. Run the provided code and then implement another class for VecDB as per the project requirements.
3. Customize the code and add any additional features as needed.
4. Run the evaluation to assess the accuracy of your implementation. The final evaluation will use the 'eval' function, but I wil update the 'run_queries' function.

## Usage

The project provides a `VecDB` class that you can use to interact with the vectorized database. Here's an example of how to use it:

```python
from VecDB import VecDB

# Create an instance of VecDB
db = VecDB()

# Insert records into the database
records = [
    {
        "id": 1,
        "embed": [0.1, 0.2, 0.3, ...]  # Image vector of dimension 70
    },
    {
        "id": 2,
        "embed": [0.4, 0.5, 0.6, ...]
    },
    ...
]
db.insert_records(records)

# Retrieve similar images for a given query
query_vector = [0.7, 0.8, 0.9, ...]  # Query vector of dimension 70
similar_images = db.retrieve(query_vector, top_k=5)
print(similar_images)
```

Feel free to customize the code to suit your specific use case.
