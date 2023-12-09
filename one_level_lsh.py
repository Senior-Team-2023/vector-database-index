import os
from typing import Dict, List, Annotated
import numpy as np
import faiss
from sklearn.neighbors import KNeighborsClassifier
import pickle
from itertools import product

# from memory_profiler import profile


class VecDB_lsh_one_level:
    def __init__(self, file_path="saved_db.csv", new_db=True) -> None:
        self.file_path = file_path
        self.d = 70  # vector dimensions
        self.max_levels = 0
        self.cos_threshold = 0.87
        self.plane_nbits = 7
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # rows is a list of dictionary, each dictionary is a record
        with open(self.file_path, "a+") as fout:
            # TODO: keep track of the last id in the database,
            # to start the new index from it, if the database is not empty,
            # and if the index algorithm requires it
            for row in rows:
                # get id and embed from dictionary
                id, embed = row["id"], row["embed"]
                # convert row to string to write it to the database file
                # TODO: Convert str(e) to bytes to reduce the size of the file
                # float should be 4 bytes, but str(e) is more than that
                # convert embed to bytes
                # --- possible sol (not tested)---
                # embed_bytes = b"".join(struct.pack("f", e) for e in embed)
                # row_bytes = struct.pack("i", id) + embed_bytes
                # fout.write(row_bytes)
                # fout.write(b"\n")
                # TODO: try to take info from the embed, so you can use it to build the index
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
        # build index after inserting all records,
        # whether on new records only or on the whole database
        self._build_index()

    # @profile
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        probing = int(2 * np.sqrt(2**self.plane_nbits))
        # probing = 2
        print(f"Probing {probing} buckets...")
        # --------HNSW---------
        print("Retrieving...")
        # print("Calculating binary vector of query...")
        query_binary_vec = self._calc_binary_vec(query, self._plane_norms)
        # print(query_binary_vec)
        # to get the binary value of the vector as a string
        query_binary_str = "".join(query_binary_vec[0].astype(str))
        # for i in range(probing):
        nearest_buckets = self._nearest_strings(query_binary_str, probing)
        # -------Worst case---------
        # return self._retrieve_worst_case(query, top_k, query_binary_str)
        # -------KNN---------
        ids, embeds = self._retrieve_KNN(query, top_k, query_binary_str)
        for bucket in nearest_buckets:
            # print(f"Probing bucket {bucket}...")
            ids1, embeds1 = self._retrieve_KNN(query, top_k, bucket)
            if len(ids1) == 0:
                continue
            ids.extend(ids1)
            embeds = np.concatenate((embeds, embeds1))
            # embeds.extend(embeds1)
            # scores.extend(self._retrieve_KNN(query, top_k, bucket))
        ids = np.array(ids)
        cosine_similarity_list = self._vectorized_cal_score(embeds, query)
        # print("Cosine similarity shape:", cosine_similarity_list.shape)
        cosine_similarity_id = np.column_stack((cosine_similarity_list, ids))
        # print("Cosine similarity id shape:", cosine_similarity_id.shape)
        sorted_indices = np.argsort(cosine_similarity_id[:, 0])

        # Sort 'scores_id_array' by 'scores' using the sorted indices
        scores = cosine_similarity_id[sorted_indices[::-1]]
        # print("Scores:", scores)
        return scores[:top_k, 1].astype(np.int32)
        # return self._retrieve_KNN(query, top_k, query_binary_str)

    def _hamming_distance(self, str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def _nearest_strings(self, binary_string, m):
        n = len(binary_string)
        all_strings = ["".join(bits) for bits in product("01", repeat=n)]
        distances = [
            (other_str, self._hamming_distance(binary_string, other_str))
            for other_str in all_strings
        ]
        distances.sort(key=lambda x: x[1])
        return [str_dist[0] for str_dist in distances[1 : m + 1]]

    def _retrieve_worst_case(
        self, query: Annotated[List[float], 70], top_k=5, query_binary_str=""
    ):
        # ------- Worst Case (for loop) ----------
        scores = []
        # open database file to read
        with open(f"./index/{query_binary_str}.csv", "r") as fin:
            # search through the file line by line (sequentially)
            # each row is a record
            for row in fin.readlines():
                row_splits = row.split(",")
                # the first element is id
                id = int(row_splits[0])
                # the rest are embed
                embed = [float(e) for e in row_splits[1:]]
                score = self._cal_score(query, embed)
                # append a tuple of score and id to scores
                scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        # sort and get the top_k records
        scores = sorted(scores, reverse=True)[:top_k]
        # print(scores)
        # return the ids of the top_k records
        return [s[1] for s in scores]

    def _retrieve_KNN(
        self, query: Annotated[List[float], 70], top_k=5, query_binary_str=""
    ):
        print(f"Loading corresponding KNN index {query_binary_str}...")
        try:
            loaded_index = pickle.load(
                open(f"./index/{query_binary_str}.csv.knn", "rb")
            )
        except FileNotFoundError:
            print(f"KNN index {query_binary_str} not found")
            return np.array([]), np.array([[]])

        try:
            distances, indices = loaded_index.kneighbors(query, n_neighbors=top_k)
        except ValueError as e:
            n_neighbors_i = e.args[0].index("n_samples = ") + len("n_samples = ")
            n_neighbors = int(e.args[0][n_neighbors_i])
            print(f"n_neighbors = {n_neighbors}")
            distances, indices = loaded_index.kneighbors(query, n_neighbors=n_neighbors)
        # Retrieve the corresponding IDs for the sorted neighbors
        ids = np.loadtxt(
            f"./index/{query_binary_str}.csv",
            delimiter=",",
            skiprows=0,
            dtype=np.int32,
            usecols=0,
        )
        if len(indices[0]) == 1:
            ids = np.array([ids])
        dataset = np.loadtxt(
            f"./index/{query_binary_str}.csv",
            delimiter=",",
            skiprows=0,
            dtype=np.float32,
            usecols=range(1, 71),
        )
        if len(indices[0]) == 1:
            dataset = np.array([dataset])
        real_ids = [ids[idx] for idx in indices[0]]
        embeds = [dataset[idx] for idx in indices[0]]
        return real_ids, embeds

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _vectorized_cal_score(self, vec1, vec2):
        vec2_broadcasted = np.broadcast_to(vec2, vec1.shape)

        # Calculate the dot product between each vector in vec1 and the broadcasted vec2
        dot_product = np.sum(vec1 * vec2_broadcasted, axis=1)
        # Calculate the dot product between each vector in vec1 and vec2
        # dot_product = np.dot(vec1, vec2.T)

        # Calculate the norm of each vector in vec1
        norm_vec1 = np.linalg.norm(vec1, axis=1)

        # Calculate the norm of vec2
        norm_vec2 = np.linalg.norm(vec2)

        # Calculate the cosine similarity for each pair of vectors
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

        return cosine_similarity.squeeze()

    def _build_index(self):
        # TODO: build index for the database
        print("Building index...")
        # ---- 1. random projection ----
        # create a set of nbits hyperplanes, with d dimensions
        # self._plane_norms = (np.random.rand(self.plane_nbits, self.d) - 0.5) * 2
        self._plane_norms = self.generate_orthogonal_vectors(self.plane_nbits, self.d)

        # open database file to read
        buckets = {}
        with open(self.file_path, "r") as fin:
            # search through the file line by line (sequentially)
            # each row is a record
            for row in fin.readlines():
                row_splits = row.split(",")
                # the first element is id
                id = int(row_splits[0])
                # the rest are embed
                embed = [float(e) for e in row_splits[1:]]
                embed = np.array(embed)

                # ---------Do random projection---------
                # calculate the dot product for each of these
                # to get the binary vector from the hyperplanes
                embed_dot = self._calc_binary_vec(embed, self._plane_norms)
                # vectors.append((id, embed_dot, embed))
                # --- 2. bucketing ---
                # convert from array to string
                # to get the binary value of the vector as a string
                hash_str = "".join(embed_dot.astype(str))
                # create bucket if it doesn't exist
                if hash_str not in buckets.keys():
                    buckets[hash_str] = []
                # add vector position to bucket
                # all vectors that has the same binary value will be in the same bucket
                # append only the id and the embed, not the binary value
                buckets[hash_str].append((id, embed))

        # save the parent database
        self._save_buckets(buckets)  # save for any bucket
        # loop over the buckets
        for k, v in buckets.items():
            # HNSW
            # self._HNSW_build_index(f"{k}_{str(0)}.csv")
            # KNN
            self._KNN_build_index(f"{k}.csv")

    def _KNN_build_index(self, filename: str):
        # open each file inside the index folder
        if filename.endswith(".csv"):
            bucket_rec = []
            with open(f"./index/{filename}", "r") as fin:
                for row in fin.readlines():
                    row_splits = row.split(",")
                    # the first element is id
                    id = int(row_splits[0])
                    # the rest are embed
                    embed = [float(e) for e in row_splits[1:]]
                    embed = np.array(embed)
                    bucket_rec.append((id, embed))
                knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")

                knn.fit(
                    np.array([e[1] for e in bucket_rec]),
                    np.array([e[0] for e in bucket_rec]),
                )
                # save the index using pickle
                with open(f"./index/{filename}.knn", "wb") as fout:
                    pickle.dump(knn, fout)
                # save a file that contains the list of ids
                # with open(f"./index/{filename}.ids", "w") as fout:
                #     for e in bucket_rec:
                #         fout.write(str(e[0]))
                #         fout.write("\n")

    def _save_buckets(self, buckets):
        for key, value in buckets.items():
            with open(f"./index/{key}.csv", "w") as fout:
                # fout.write(",".join(str(e) for e in value))
                # print(value)
                # NOTE: mo4kla bemoi ali nseha fel level <3 <3 <3
                for e in value:
                    fout.write(str(e[0]) + "," + ",".join(str(t) for t in e[1]))
                    fout.write("\n")

    def _load_buckets(self, key):
        buckets = []
        with open(f"./index/{key}.csv", "r") as fin:
            for line in fin.readlines():
                bucket = line.split(",")
                bucket = [float(e) for e in bucket]
                buckets.append((int(bucket[0]), bucket[1:]))
        return buckets

    # Calculate random projection and return binary vector
    def _calc_binary_vec(self, embed, plane_norms):
        embed_dot = np.dot(embed, plane_norms.T)
        # we know that a positive dot product == +ve side of hyperplane
        # and negative dot product == -ve side of hyperplane
        embed_dot = embed_dot > 0
        # convert our boolean arrays to int arrays to make bucketing
        # easier (although is okay to use boolean for Hamming distance)
        embed_dot = embed_dot.astype(int)
        return embed_dot

    def generate_orthogonal_vectors(self, n, d):
        # Start with an empty list to hold the vectors
        vectors = []
        # Loop to generate n vectors
        for _ in range(n):
            # Generate a random vector
            random_shift = np.random.randn(1)
            # random_shift = random_shift if random_shift < 0.9 else 0
            vec = np.random.randn(d)
            # Orthogonalize it against all previously generated vectors
            for basis in vectors:
                vec -= np.dot(vec, basis) * basis
            # Normalize the vector
            vec /= np.linalg.norm(vec)
            # Add the orthogonalized, normalized vector to the list
            vectors.append(vec)
        return np.array(vectors)
