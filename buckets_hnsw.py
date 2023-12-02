import os
from typing import Dict, List, Annotated
import numpy as np
import faiss
from sklearn.neighbors import KNeighborsClassifier
import pickle

# from memory_profiler import profile


class VecDB_buckets_HNSW:
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
        # TODO: for our implementation, we will use the index to retrieve the top_k records
        # then retrieve the actual records from the database
        # to get the binary vector from the hyperplanes
        # print(query)
        # query = np.array([0.374540119,0.950714306,0.731993942,0.598658484,0.15601864,0.15599452,0.058083612,0.866176146,0.601115012,0.708072578,0.020584494,0.969909852,0.832442641,0.212339111,0.181824967,0.18340451,0.304242243,0.524756432,0.431945019,0.29122914,0.611852895,0.139493861,0.292144649,0.366361843,0.456069984,0.785175961,0.199673782,0.514234438,0.592414569,0.046450413,0.607544852,0.170524124,0.065051593,0.948885537,0.965632033,0.808397348,0.304613769,0.097672114,0.684233027,0.440152494,0.122038235,0.49517691,0.034388521,0.909320402,0.258779982,0.662522284,0.311711076,0.520068021,0.546710279,0.184854456,0.969584628,0.775132823,0.939498942,0.89482735,0.597899979,0.921874235,0.088492502,0.195982862,0.045227289,0.325330331,0.38867729,0.271349032,0.828737509,0.356753327,0.28093451,0.542696083,0.140924225,0.802196981,0.074550644,0.986886937
        #     ]).reshape(1, -1)
        print("Retrieving...")
        print("Calculating binary vector of query...")
        query_binary_vec = self._calc_binary_vec(query, self._plane_norms)
        # print(query_binary_vec)
        # to get the binary value of the vector as a string
        query_binary_str = "".join(query_binary_vec[0].astype(str))

        # load the hyperplane from file
        leaf_level = 0
        loaded_hyperplane = None
        print(f"max_levels = {self.max_levels}")
        prev_query_binary_str = ""
        for i in range(self.max_levels + 1):
            try:
                # load hyperplanes based on bucket key and level
                loaded_hyperplane = self._load_hyperplanes(
                    f"hyperplane_{query_binary_str}_{i}"
                )

                query_binary_vec = self._calc_binary_vec(query, loaded_hyperplane)

                query_binary_str = "".join(query_binary_vec[0].astype(str))
                print(f"Hyperplane {query_binary_str} found, i = {i}")
                print(f"Previous Hyperplane {prev_query_binary_str} found, i = {i-1}")
                # this holds the query binary str of the parent bucket
                prev_query_binary_str = query_binary_str
            except FileNotFoundError:
                print(f"Hyperplane {query_binary_str} not found, i = {i}")
                # if i == 0:
                #     # if the hyp    erplane of the first level is not found, so take the hyperplane of the first bucket
                #     loaded_hyperplane = self._plane_norms
                # else:
                #     loaded_hyperplane = self._load_hyperplanes(
                #         f"hyperplane_{prev_query_binary_str}_{i-1}"
                #     )
                leaf_level = i

                # 34an no5rog mn al loop b3d ma nla2y al hyperplane
                break

        # query_binary_vec = self._calc_binary_vec(query, loaded_hyperplane)
        # query_binary_str = "".join(query_binary_vec[0].astype(str))

        # load bucket from file
        print(f"Loading corresponding HNSW index {query_binary_str}...")
        try:
            # loaded_index = faiss.read_index(
            #     f"./index/{query_binary_str}_{leaf_level}.csv.index"
            # )
            loaded_index = pickle.load(
                open(f"./index/{query_binary_str}_{leaf_level}.csv.index", "rb")
            )
        except FileNotFoundError:
            print(f"HNSW index {query_binary_str} not found")
        # distances, labels = loaded_index.search(query, top_k)
        try:
            distances, labels = loaded_index.kneighbors(query, n_neighbors=top_k)
        except ValueError as e:
            n_neighbors_i = e.args[0].index("n_samples = ") + len("n_samples = ")
            n_neighbors = int(e.args[0][n_neighbors_i])
            print(f"n_neighbors = {n_neighbors}")
            distances, labels = loaded_index.kneighbors(query, n_neighbors=n_neighbors)
        print("distances", distances)
        print("labels", labels)
        # calculate the score for each vector in the bucket
        print("Calculating score...")
        scores = [(distances[0][i], labels[0][i]) for i in range(len(labels[0]))]
        scores = sorted(scores)[:top_k]
        # return the ids of the top_k records
        print(scores)
        return [s[1] for s in scores]
        # return scores

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
        # nbits = 4  # number of hyperplanes and binary vals to produce
        # create a set of nbits hyperplanes, with d dimensions
        self._plane_norms = np.random.rand(self.plane_nbits, self.d) - 0.5
        # store the hyperplanes in a file
        # self._save_hyperplanes("hyperplane0", self._plane_norms)
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
            vectors_mean = np.mean(vectors, axis=0)
            # norm_product = 1
            # dot_product = np.ones(vectors[0].shape)
            # for i in range(len(vectors)):
            #     norm_product = norm_product * np.linalg.norm(vectors[i])
            #     dot_product = dot_product * vectors[i]

            # dot_product = np.sum(dot_product)
            # calculate the cosine similarity between the mean vector and all the vectors in the bucket and get the max
            cosine_similarity = np.mean(
                [self._cal_score(vectors_mean, vector) for vector in vectors]
            )

            # print("norm_product",norm_product)
            # print("dot_product",dot_product)
            # cosine_similarity = dot_product / (norm_product)
            # TODO: Tune the threshold (e.g. 0.95) to get the best results
            print("cosine_similarity", cosine_similarity)
            if len(v) <= 5 or cosine_similarity >= self.cos_threshold:
                self._HNSW_build_index(f"{k}_{str(0)}.csv")
            else:
                self._build_bucket_index(f"{k}_{str(0)}.csv", 1)

    def _build_bucket_index(self, filename: str, lvl: int):
        # TODO: build index for the database
        print("Building index...")
        # ---- 1. random projection ----
        # nbits = 4  # number of hyperplanes and binary vals to produce
        # create a set of nbits hyperplanes, with d dimensions
        _plane_norms = np.random.rand(self.plane_nbits, self.d) - 0.5
        # store the hyperplanes in a file
        self._save_hyperplanes("hyperplane_" + filename, _plane_norms)
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
            # norm_product = 1
            vectors_mean = np.mean(vectors, axis=0)

            # calculate the cosine similarity between the mean vector and all the vectors in the bucket and get the max
            cosine_similarity = np.mean(
                [self._cal_score(vectors_mean, vector) for vector in vectors]
            )

            # dot_product = np.ones(vectors[0].shape)
            # for i in range(len(vectors)):
            #     norm_product = norm_product * np.linalg.norm(vectors[i])
            #     dot_product = dot_product * vectors[i]
            # mean of the vectors
            # dot_product = np.sum(dot_product)
            # cosine_similarity = dot_product / (norm_product)
            # print("norm_product",norm_product)
            # print("dot_product",dot_product)
            print("cosine_similarity", cosine_similarity)

            if len(v) <= 5 or cosine_similarity >= self.cos_threshold:
                # update the max level
                if lvl > self.max_levels:
                    self.max_levels = lvl

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
                # self._HNSW_index(
                #     data=bucket_rec,
                #     m=128,
                #     ef_construction=200,
                #     ef_search=32,
                #     filename=filename,
                # )
                # Knn using sklearn
                knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
                knn.fit(
                    np.array([e[1] for e in bucket_rec]),
                    np.array([e[0] for e in bucket_rec]),
                )
                # save the index using pickle
                with open(f"./index/{filename}.index", "wb") as fout:
                    pickle.dump(knn, fout)

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
