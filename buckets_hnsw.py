import os
from typing import Dict, List, Annotated
import numpy as np
import faiss

# from memory_profiler import profile


class VecDB_buckets_HNSW:
    def __init__(self, file_path="saved_db.csv", new_db=True) -> None:
        self.file_path = file_path
        self.d = 70  # vector dimensions
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
        # TODO: for our implementation, we will use the index to retrieve the top_k records
        # then retrieve the actual records from the database
        # to get the binary vector from the hyperplanes
        # print(query)
        print("Retrieving...")
        print("Calculating binary vector of query...")
        query_binary_vec = self._calc_binary_vec(query, self._plane_norms)
        # print(query_binary_vec)
        # to get the binary value of the vector as a string
        query_binary_str = "".join(query_binary_vec[0].astype(str))
        # load bucket from file
        print(f"Loading corresponding HNSW index {query_binary_str}...")
        try:
            loaded_index = faiss.read_index(f"./index/{query_binary_str}.csv.index")
        except FileNotFoundError:
            print(f"HNSW index {query_binary_str} not found")
        distances, labels = loaded_index.search(query, top_k)
        print("distances", distances)
        print("labels", labels)
        # calculate the score for each vector in the bucket
        print("Calculating score...")
        scores = [(distances[0][i], labels[0][i]) for i in range(len(labels[0]))]
        scores = sorted(scores)[:top_k]
        # return the ids of the top_k records
        # print(scores)
        return [s[1] for s in scores]

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
        nbits = 4  # number of hyperplanes and binary vals to produce
        # create a set of nbits hyperplanes, with d dimensions
        self._plane_norms = np.random.rand(nbits, self.d) - 0.5
        # store the hyperplanes in a file
        self._save_hyperplanes("hyperplane0", self._plane_norms)
        # vectors = []
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
        # loop over the buckets
        for k, v in buckets.items():
            # inside the stop codition, to build the index
            # stop condition:
            # min number of records inside the bucket is 5
            # similarity measure (cosine similarity) among all the records inside the bucket >= 0.7
            # if the stop condition is not met, build the index for each bucket
            # for (_,vector) in v:
            vectors = [e[1] for e in v]
            vectors = np.array(vectors)
            norm_product = 1
            dot_product = np.ones(vectors[0].shape)
            for i in range(len(vectors)):
                norm_product = norm_product * np.linalg.norm(vectors[i])
                dot_product = dot_product * vectors[i]

            dot_product = np.sum(dot_product)
            # print("norm_product",norm_product)
            # print("dot_product",dot_product)
            cosine_similarity = dot_product / (norm_product)
            print("cosine_similarity", cosine_similarity)
            if len(v) <= 5 or cosine_similarity >= 0.5:
                self._HNSW_build_index(f"{k}_{str(0)}.csv")
            else:
                self._build_bucket_index(f"{k}_{str(0)}.csv", 1)

    def _build_bucket_index(self, filename: str, lvl: int):
        # TODO: build index for the database
        print("Building index...")
        # ---- 1. random projection ----
        nbits = 4  # number of hyperplanes and binary vals to produce
        # create a set of nbits hyperplanes, with d dimensions
        _plane_norms = np.random.rand(nbits, self.d) - 0.5
        # store the hyperplanes in a file
        self._save_hyperplanes("hyperplane_" + filename + "_" + str(lvl), _plane_norms)
        # vectors = []
        # open database file to read
        buckets = {}
        with open(f"./index/{filename}", "r") as fin:
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
                embed_dot = self._calc_binary_vec(embed, _plane_norms)
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

        # print(buckets)
        # save buckets to files
        print("Saving buckets...")
        self._save_buckets(buckets, lvl)  # save for any bucket

        # loop over the buckets
        for k, v in buckets.items():
            # inside the stop codition, to build the index
            # stop condition:
            # min number of records inside the bucket is 5
            # similarity measure (cosine similarity) among all the records inside the bucket >= 0.7
            # if the stop condition is not met, build the index for each bucket
            # for (_,vector) in v:
            vectors = [e[1] for e in v]
            vectors = np.array(vectors)
            norm_product = 1
            dot_product = np.ones(vectors[0].shape)
            for i in range(len(vectors)):
                norm_product = norm_product * np.linalg.norm(vectors[i])
                dot_product = dot_product * vectors[i]

            dot_product = np.sum(dot_product)
            cosine_similarity = dot_product / (norm_product)
            # print("norm_product",norm_product)
            # print("dot_product",dot_product)
            # print(cosine_similarity)
            if len(v) <= 5 or cosine_similarity >= 0.5:
                self._HNSW_build_index(f"{k}_{str(lvl)}.csv")
            else:
                self._build_bucket_index(f"{k}_{str(lvl)}.csv", lvl + 1)

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
                # bucket_rec = np.array(bucket_rec)
                self._HNSW_index(
                    data=bucket_rec,
                    m=128,
                    ef_construction=200,
                    ef_search=32,
                    filename=filename,
                )

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
        with open(f"./hyperplanes/{filename}.csv", "w") as fout:
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
        with open(f"{filename}.csv", "r") as fin:
            for line in fin.readlines():
                plane_norm = line.split(",")
                plane_norm = [float(e) for e in plane_norm]
                plane_norms.append(plane_norm)
        return plane_norms

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
