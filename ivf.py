import numpy as np

# from scipy.cluster.vq import kmeans2
from faiss import Kmeans
from typing import Dict, List, Annotated
# from sklearn.decomposition import (
#     PCA,
# )  # TODO: use PCA to reduce the dimension of the vectors
import os
import gc

# from sklearn.cluster import MiniBatchKMeans
# import joblib


class VecDB:
    # constructor
    def __init__(self, file_path="saved_db.csv", new_db=True):
        self.num_part = 32  # number of partitions
        self.centroids = None  # centroids of each partition
        # self.assignments = None  # assignments of each vector to a partition
        self.iterations = 32  # number of iterations for kmeans

        self.file_path = file_path
        if new_db:
            self.database_size = 0
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
        else:
            # load the centriods
            self.centroids = np.load(f"./index_{self.file_path}_centroids.npy")
            # load the length of each cluster
            self.length_of_clusters = np.load(
                f"./index_{self.file_path}_length_of_clusters.npy"
            )
            # get the size of the database from the sum of the length of each cluster
            self.database_size = np.sum(self.length_of_clusters)
            # read database from file
            self.database = np.memmap(
                self.file_path,
                dtype=np.float32,
                mode="r",
                shape=(self.database_size, 71),
            )

    # def insert_records(
    #     self, rows: List[Dict[int, Annotated[List[float], 70]]], build_index=True
    # ):
    def insert_records(self, vectors, build_index=True):
        # rows is a list of dictionary, each dictionary is a record
        # with open(self.file_path, "a+") as fout:
        #     # to start the new index from it, if the database is not empty,
        #     # and if the index algorithm requires it
        #     self.database_size += rows.shape[0]

        #     # print("database_size:", self.database_size)
        #     for i in range(rows.shape[0]):
        #         # get id and embed from dictionary
        #         # id, embed = row["id"], row["embed"]
        #         id = i
        #         embed = rows[i]
        #         # convert row to string to write it to the database file
        #         # NOTE: Convert str(e) to bytes to reduce the size of the file
        #         # float should be 4 bytes, but str(e) is more than that
        #         # NOTE: try to take info from the embed, so you can use it to build the index
        #         row_str = f"{id}," + ",".join([str(e) for e in embed])
        #         fout.write(f"{row_str}\n")
        self.database_size = vectors.shape[0]
        self.database = np.memmap(
            self.file_path,
            dtype=np.float32,
            mode="w+",
            shape=(vectors.shape[0], vectors.shape[1] + 1),
        )
        self.database[:, 0] = np.array([i for i in range(vectors.shape[0])])
        self.database[:, 1:] = vectors[:]
        self.database.flush()

        # del fp
        del vectors
        gc.collect()
        # build index after inserting all records,
        # whether on new records only or on the whole database
        if build_index:
            self.build_index()

    # Worst case implementation for retrieve
    # Because it is sequential search
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        # then retrieve the actual records from the database
        scores = []
        # get the top_k centroids
        k = int(4 * np.sqrt(self.num_part))
        top_centroids = self._get_top_centroids(query, k)
        # print("top_centroids:", top_centroids)
        # load kmeans model
        # kmeans = joblib.load("./kmeans_model.joblib")
        # # get the assignments of the query to the centroids
        # c = kmeans.predict(query)
        # print("c:", c)
        # c = c[0]
        # with open(f"./index/index_{c}.csv", "r") as fin:
        #         for row in fin:
        #             row_splits = row.split(",")
        #             # the first element is id
        #             id = int(row_splits[0])
        #             # the rest are embed
        #             embed = [float(e) for e in row_splits[1:]]
        #             score = self._cal_score(query, embed)
        #             # append a tuple of score and id to scores
        #             scores.append((score, id))
        # for each centrioed, get sorted list constains the nearest top_k vectors to it
        # cosine_similarity_id = []
        cosine_similarity_id_total = np.array([]).reshape(0, 2)
        for centroid in top_centroids:
            try:
                # if len(self.index[centroid]) == 0:
                #     continue
                fp = np.memmap(
                    f"./index_{self.database_size}/index_{centroid}.dta",
                    dtype="float32",
                    mode="r",
                    shape=(self.length_of_clusters[centroid], 71),
                )
                # print number of vectors in this cluster
                print(f"centroid {centroid} shape:", fp.shape)
            except FileNotFoundError:
                continue
            id = fp[:, 0].astype(np.int32)
            # print("id:", id)
            # print("id shape:", id.shape)
            embed = fp[:, 1:]
            cosine_similarity_list = self._vectorized_cal_score(embed, query)
            # cosine_similarity_list = np.vectorize(
            #     self._vectorized_cal_score,
            #     signature=f"({self.index[centroid].shape[0]},70),(1,70)->(70)",
            # )(embed, query)
            # print("cosine_similarity_list:", cosine_similarity_list)
            # print("cosine_similarity_list shape:", cosine_similarity_list.shape)

            # scores = [(s, i) for s, i in zip(scores, id)]
            cosine_similarity_id = np.column_stack((cosine_similarity_list, id))
            # cosine_similarity_id = np.concatenate((cosine_similarity_id, id), axis=1)
            # print("cosine_similarity_id:", cosine_similarity_id)
            # print("cosine_similarity_id shape:", cosine_similarity_id.shape)
            cosine_similarity_id_total = np.concatenate(
                (cosine_similarity_id_total, cosine_similarity_id), axis=0
            )

        sorted_indices = np.argsort(cosine_similarity_id_total[:, 0])

        # Sort 'scores_id_array' by 'scores' using the sorted indices
        scores = cosine_similarity_id_total[sorted_indices[::-1]]
        # print("scores:", scores)
        #     with open(f"./index/index_{centroid}.csv", "r") as fin:
        #         for row in fin:
        #             row_splits = row.split(",")
        #             # the first element is id
        #             id = int(row_splits[0])
        #             # the rest are embed
        #             embed = [float(e) for e in row_splits[1:]]
        #             score = self._cal_score(query, embed)
        #             # append a tuple of score and id to scores
        #             scores.append((score, id))
        # # here we assume that if two rows have the same score, return the lowest ID
        # # sort and get the top_k records
        # scores = sorted(scores, reverse=True)[:top_k]
        # print(scores)
        # return the ids of the top_k records
        return scores[:top_k, 1].astype(np.int32)

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

    def build_index(self):
        print("Building index...")
        # read the database file from csv file
        # id_of_dataset = np.loadtxt(
        #     self.file_path, delimiter=",", skiprows=0, dtype=np.int32, usecols=0
        # )
        # print("id_of_dataset shape:", id_of_dataset.shape)
        # batch_size = a5t ad eh mn al db awel mara
        min_batch_size = 20 * 10**5
        batch_size = (
            min(min_batch_size, int(self.database_size * 0.5))
            if self.database_size >= 10**6
            else self.database_size
        )
        # dataset = np.loadtxt(
        #     self.file_path,
        #     delimiter=",",
        #     skiprows=0,
        #     dtype=np.float32,
        #     usecols=range(1, 71),
        #     max_rows=min(min_batch_size, int(self.database_size * 0.5))
        #     if self.database_size >= 10**6
        #     else None,
        #     # max_rows=10**6
        #     # if self.database_size > 10**6
        #     # else None,
        # )
        # dataset = np.memmap(
        #     self.file_path,
        #     dtype="float32",
        #     mode="r",
        #     shape=(batch_size, 71),
        # )
        dataset = self.database[:batch_size, 1:]
        self.num_part = int(np.sqrt(self.database_size))

        # print("num_part:", self.num_part)

        # using numpy
        index = np.empty(self.num_part, dtype=object)

        for i in range(self.num_part):
            index[i] = []

        # (self.centroids, assignments) = kmeans2(
        #     dataset, self.num_part, iter=self.iterations
        # )
        kmeans = Kmeans(70, self.num_part, niter=20, verbose=True)
        kmeans.train(
            dataset,
            init_centroids=dataset[
                np.random.choice(
                    dataset.shape[0],
                    self.num_part,
                    replace=dataset.shape[0] < self.num_part,
                )
            ],
        )
        self.centroids = kmeans.centroids
        assignments = kmeans.assign(dataset)[
            1
        ]  # returns a label to each row in that array

        print("assignments len:", len(assignments))
        print("assignments:", assignments)
        for n, k in enumerate(assignments):
            # n is the index of the vector
            # k is the index of the cluster
            index[k].append(n)

        del assignments
        gc.collect()

        for i in range(len(index)):
            if len(index[i]) == 0:
                continue
            cluster = index[i]
            new_cluster = np.memmap(
                f"./index_{self.database_size}/index_{i}.dta",
                dtype="float32",
                mode="w+",
                shape=(len(cluster), 71),
            )
            for n, id in enumerate(cluster):
                new_cluster[n][0] = id
                new_cluster[n][1:] = dataset[id]
            new_cluster.flush()
            # del cluster
        # del dataset

        if self.database_size >= 10**6:
            # convert the assignments to list
            # assignments = assignments.tolist()
            # loop over the rest of the database to assign each vector to a cluster by appending the cluster id of this vector to the assignments list
            for i in range(batch_size, self.database_size, batch_size):
                # dataset = np.loadtxt(
                #     self.file_path,
                #     delimiter=",",
                #     skiprows=i,
                #     dtype=np.float32,
                #     usecols=range(1, 71),
                #     max_rows=batch_size
                #     if i + batch_size < self.database_size
                #     else None,
                # )
                # dataset = np.memmap(
                #     self.file_path,
                #     dtype="float32",
                #     mode="r",
                #     offset=i,
                #     shape=(batch_size, 71),
                # )
                dataset = self.database[i : i + batch_size, 1:]

                # find the nearest centroid to the query
                # top_centriods = [
                #     self._get_top_centroids(vector, 1)[0] for vector in dataset
                # ]
                top_centriods = kmeans.assign(dataset)[
                    1
                ]  # returns a label to each row in that array
                # top_centriods = [np.argmax(self._vectorized_cal_score(self.centroids, vector)) for vector in dataset]
                print("top_centriods shape:", len(top_centriods))
                # print("top_centriods:", top_centriods)
                # append the cluster id of each vector to the assignments list
                # assignments = np.append(
                #     assignments,
                #     top_centriods,
                # )
                # assignments.extend(top_centriods)
                for n, k in enumerate(top_centriods):
                    # n is the index of the vector
                    # k is the index of the cluster
                    index[k].append(n + i)

                # save the current IVF after each batch
                for c in set(top_centriods):
                    cluster = index[c]

                    file_path = f"./index_{self.database_size}/index_{c}.dta"
                    try:
                        file_size = os.path.getsize(file_path)

                        # Calculate the number of rows, knowing each row has 71 columns of type float32 (4 bytes each)
                        num_columns = 71

                        bytes_per_row = num_columns * 4  # float32 has 4 bytes

                        num_rows = file_size // bytes_per_row
                        # print("num_rows:", num_rows)
                        # if num_rows != 0:
                        old_cluster = np.memmap(
                            file_path,
                            dtype="float32",
                            mode="r",
                            shape=(num_rows, num_columns),
                        )
                        old_cluster_copy = np.array(old_cluster)
                        old_cluster.flush()
                        del old_cluster
                        # gc.collect()
                    except FileNotFoundError:
                        old_cluster = None
                        old_cluster_copy = None
                    new_shape = (len(cluster), num_columns)
                    # create a new cluster with the new shape
                    new_cluster = np.memmap(
                        file_path, dtype="float32", mode="w+", shape=new_shape
                    )
                    # update the cluster with the new vectors
                    if old_cluster_copy is not None:
                        new_cluster[:num_rows] = old_cluster_copy[:]

                    for j in range(num_rows, len(cluster)):
                        new_cluster[j][0] = cluster[j]
                        new_cluster[j][1:] = dataset[cluster[j] - i]

                    new_cluster.flush()
                    # del old_cluster
                    # del new_cluster
                    # del cluster

                # NOTE:np.save(f"./index_{self.database_size}/index_{cen}.npy", self.index[cen])

                del top_centriods
                gc.collect()
                # delete dataset to free memory
                # del dataset
            # assignments = np.array(assignments)

        # print("assignments len:", len(assignments))
        # for n, k in enumerate(assignments):
        #     # n is the index of the vector
        #     # k is the index of the cluster
        #     self.index[k].append(n)

        # del assignments

        # convert the index to numpy array
        index = np.array(index)
        # print("index shape:", self.index.shape)
        # length of each cluster
        self.length_of_clusters = np.array([len(cluster) for cluster in index])
        del index
        gc.collect()
        # store length of each cluster as memmap
        np.save(
            f"./index_{self.file_path}_length_of_clusters.npy", self.length_of_clusters
        )
        # store the centriods
        np.save(f"./index_{self.file_path}_centroids.npy", self.centroids)

        # save the index clusters to .csv files
        # for i, cluster in enumerate(self.index):
        #     with open(f"./index/index_{i}.csv", "w") as fout:
        #         for n in cluster:
        #             fout.write(f"{id_of_dataset[n]},{','.join(map(str, dataset[n]))}\n")
        # if self.database_size >= 10**6:
        #     for i in range(len(self.index)):
        #         if len(self.index[i]) == 0:
        #             continue
        #         cluster = self.index[i]
        #         self.index[i] = np.memmap(
        #             f"./index_{self.database_size}/index_{i}.dta",
        #             dtype="float32",
        #             mode="w+",
        #             shape=(len(cluster), 71),
        #         )
        #         minID = min(cluster)
        #         maxID = max(cluster)
        #         dataset = np.loadtxt(
        #             self.file_path,
        #             delimiter=",",
        #             skiprows=minID,
        #             dtype=np.float32,
        #             usecols=range(1, 71),
        #             max_rows=maxID - minID + 1,
        #         )
        #         for n, id in enumerate(cluster):
        #             self.index[i][n][0] = id
        #             self.index[i][n][1:] = dataset[id - minID]
        #             # self.index[i][n][1:] = np.loadtxt(
        #             #     self.file_path,
        #             #     delimiter=",",
        #             #     skiprows=id,
        #             #     dtype=np.float32,
        #             #     usecols=range(1, 71),
        #             #     max_rows=1,
        #             # )
        #         del cluster
        #         del dataset
        # else:
        #     for i in range(len(self.index)):
        #         if len(self.index[i]) == 0:
        #             continue
        #         cluster = self.index[i]
        #         self.index[i] = np.memmap(
        #             f"./index_{self.database_size}/index_{i}.dta",
        #             dtype="float32",
        #             mode="w+",
        #             shape=(len(cluster), 71),
        #         )
        #         for n, id in enumerate(cluster):
        #             self.index[i][n][0] = id
        #             self.index[i][n][1:] = dataset[id]
        #         del cluster
        #     del dataset

    def _get_top_centroids(self, query, k):
        # find the nearest centroids to the query
        top_k_centroids = np.argsort(
            np.linalg.norm(self.centroids - np.array(query), axis=1)
        )
        # get the top_k centroids
        top_k_centroids = top_k_centroids[:k]
        return top_k_centroids
