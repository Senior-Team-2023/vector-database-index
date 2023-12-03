import os
from typing import Dict, List, Annotated
import numpy as np
import faiss
from sklearn.neighbors import KNeighborsClassifier
import pickle

# from memory_profiler import profile


class VecDB_lsh_one_level:
    def __init__(self, file_path="saved_db.csv", new_db=True) -> None:
        self.file_path = file_path
        self.d = 70  # vector dimensions
        self.max_levels = 0
        self.cos_threshold = 0.87
        self.plane_nbits = 5
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
        # --------HNSW---------
        print("Retrieving...")
        # print("Calculating binary vector of query...")
        query_binary_vec = self._calc_binary_vec(query, self._plane_norms)
        # print(query_binary_vec)
        # to get the binary value of the vector as a string
        query_binary_str = "".join(query_binary_vec[0].astype(str))
        # -------Worst case---------
        # return self._retrieve_worst_case(query, top_k, query_binary_str)
        # -------HNSW---------
        # return self._retrieve_HNSW(query, top_k, query_binary_str)
        # -------KNN---------
        return self._retrieve_KNN(query, top_k, query_binary_str)

    def _retrieve_worst_case(
        self, query: Annotated[List[float], 70], top_k=5, query_binary_str=""
    ):
        # ------- Worst Case (for loop) ----------
        scores = []
        # open database file to read
        with open(f"./index/{query_binary_str}_{0}.csv.", "r") as fin:
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

    def _retrieve_HNSW(
        self, query: Annotated[List[float], 70], top_k=5, query_binary_str=""
    ):
        print(f"Loading corresponding HNSW index {query_binary_str}...")
        try:
            loaded_index = faiss.read_index(f"./index/{query_binary_str}_{0}.csv.index")
        except FileNotFoundError:
            print(f"KNN index {query_binary_str} not found")
        distances, labels = loaded_index.search(query, top_k)

        # print("distances", distances)
        # print("labels", labels)
        # calculate the score for each vector in the bucket
        print("Calculating score...")
        scores = [(distances[0][i], labels[0][i]) for i in range(len(labels[0]))]
        scores = sorted(scores)[:top_k]
        # return the ids of the top_k records
        # print(scores)
        return [s[1] for s in scores]

    def _retrieve_KNN(
        self, query: Annotated[List[float], 70], top_k=5, query_binary_str=""
    ):
        print(f"Loading corresponding KNN index {query_binary_str}...")
        try:
            loaded_index = pickle.load(
                open(f"./index/{query_binary_str}_{0}.csv.knn", "rb")
            )
        except FileNotFoundError:
            print(f"KNN index {query_binary_str} not found")

        try:
            distances, indices = loaded_index.kneighbors(query, n_neighbors=top_k)
        except ValueError as e:
            n_neighbors_i = e.args[0].index("n_samples = ") + len("n_samples = ")
            n_neighbors = int(e.args[0][n_neighbors_i])
            print(f"n_neighbors = {n_neighbors}")
            distances, indices = loaded_index.kneighbors(query, n_neighbors=n_neighbors)
        # print("distances", distances)
        # print("indices", indices)
        # calculate the score for each vector in the bucket
        print("Calculating score...")
        # scores = [(distances[0][i], indices[0][i]) for i in range(len(indices[0]))]
        # scores = sorted(scores)[:top_k]
        # Sort the neighbors by distance
        sorted_neighbors = sorted(zip(indices[0], distances[0]), key=lambda x: x[1])

        # Retrieve the corresponding IDs for the sorted neighbors

        ids = []
        with open(f"./index/{query_binary_str}_{0}.csv.ids", "r") as fin:
            for row in fin.readlines():
                id = int(row)
                ids.append(id)
                # the rest are embed
        sorted_ids = [ids[idx] for idx, _ in sorted_neighbors][:top_k]

        # return the ids of the top_k records
        print(sorted_ids)
        return sorted_ids

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

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
        self._save_buckets(buckets, 0)  # save for any bucket
        # loop over the buckets
        for k, v in buckets.items():
            # HNSW
            # self._HNSW_build_index(f"{k}_{str(0)}.csv")
            # KNN
            self._KNN_build_index(f"{k}_{str(0)}.csv")

    def _HNSW_build_index(self, filename: str):
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
                # build the HNSW index
                # print("bucket_rec", bucket_rec)
                # bucket_rec = np.array(bucket_rec)
                self._HNSW_index(
                    data=bucket_rec,
                    m=128,
                    ef_construction=200,
                    ef_search=32,
                    filename=filename,
                )

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
                with open(f"./index/{filename}.ids", "w") as fout:
                    for e in bucket_rec:
                        fout.write(str(e[0]))
                        fout.write("\n")

    def _HNSW_index(self, data, m, ef_construction, filename, ef_search):
        index = faiss.IndexHNSWFlat(self.d, m)
        # set efConstruction and efSearch parameters
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        # Wrap the index with IDMap
        id_map = faiss.IndexIDMap(index)
        id_map.add_with_ids(
            np.array([e[1] for e in data]), np.array([e[0] for e in data])
        )
        # save the index
        faiss.write_index(id_map, f"./index/{filename}.index")

    def _save_hyperplanes(self, filename, plane_norms):
        with open(f"./hyperplanes/{filename}", "w") as fout:
            for plane in plane_norms:
                fout.write(",".join([str(e) for e in plane]))
                fout.write("\n")

    def _save_buckets(self, buckets, lvl: int):
        for key, value in buckets.items():
            with open(f"./index/{key}_{lvl}.csv", "w") as fout:
                # fout.write(",".join(str(e) for e in value))
                # print(value)
                # NOTE: mo4kla bemoi ali nseha fel level <3 <3 <3
                for e in value:
                    fout.write(str(e[0]) + "," + ",".join(str(t) for t in e[1]))
                    fout.write("\n")

    def _load_hyperplanes(self, filename):
        plane_norms = []
        with open(f"./hyperplanes/{filename}.csv", "r") as fin:
            for line in fin.readlines():
                plane_norm = line.split(",")
                plane_norm = [float(e) for e in plane_norm]
                plane_norms.append(plane_norm)
        return np.array(plane_norms)

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
            vec = np.random.randn(d) - 0.5
            # Orthogonalize it against all previously generated vectors
            for basis in vectors:
                vec -= np.dot(vec, basis) * basis
            # Normalize the vector
            vec /= np.linalg.norm(vec)
            # Add the orthogonalized, normalized vector to the list
            vectors.append(vec)
        return np.array(vectors)
