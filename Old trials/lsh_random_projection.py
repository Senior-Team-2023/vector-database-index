from typing import Dict, List, Annotated
import numpy as np


class VecDBLSH:
    def __init__(self, file_path="saved_db.csv", new_db=True) -> None:
        self.file_path = file_path
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

    # Worst case implementation for retrieve
    # Because it is sequential search
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        # TODO: for our implementation, we will use the index to retrieve the top_k records
        # then retrieve the actual records from the database
        # to get the binary vector from the hyperplanes
        # print(query)
        print("Retrieving...")
        print("Calculating binary vector of query...")
        query_binary_vec = self._calc_binary_vec(query)
        # print(query_binary_vec)
        # to get the binary value of the vector as a string
        query_binary_str = "".join(query_binary_vec[0].astype(str))
        # load bucket from file
        print(f"Loading corresponding bucket {query_binary_str}...")
        try:
            bucket = self._load_buckets(query_binary_str)
        except FileNotFoundError:
            print(f"Bucket {query_binary_str} not found")
            bucket = []
        # calculate the score for each vector in the bucket
        print("Calculating score for each vector in the bucket...")
        scores = []
        for id, embed in bucket:
            score = self._cal_score(query, embed)
            scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        # sort and get the top_k records
        scores = sorted(scores, reverse=True)[:top_k]
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
        d = 70  # vector dimensions
        # create a set of nbits hyperplanes, with d dimensions
        plane_norms = np.random.rand(nbits, d) - 0.5
        # store the hyperplanes in a file
        self._save_hyperplanes(plane_norms)
        # save the hyperplanes to the class
        self._plane_norms = plane_norms
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
                # calculate the dot product for each of these
                # to get the binary vector from the hyperplanes
                embed_dot = self._calc_binary_vec(embed)
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
        self._save_buckets(buckets)

    def _save_hyperplanes(self, plane_norms):
        with open("hyperplanes.csv", "w") as fout:
            for plane in plane_norms:
                fout.write(",".join([str(e) for e in plane]))
                fout.write("\n")

    def _save_buckets(self, buckets):
        for key, value in buckets.items():
            with open(f"./index/{key}.csv", "w") as fout:
                # fout.write(",".join(str(e) for e in value))
                # print(value)
                for e in value:
                    fout.write(str(e[0]) + "," + ",".join(str(t) for t in e[1]))
                    fout.write("\n")

    def _load_hyperplanes(self):
        plane_norms = []
        with open("hyperplanes.csv", "r") as fin:
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

    def _calc_binary_vec(self, embed):
        embed_dot = np.dot(embed, self._plane_norms.T)
        # we know that a positive dot product == +ve side of hyperplane
        # and negative dot product == -ve side of hyperplane
        embed_dot = embed_dot > 0
        # convert our boolean arrays to int arrays to make bucketing
        # easier (although is okay to use boolean for Hamming distance)
        embed_dot = embed_dot.astype(int)
        return embed_dot
