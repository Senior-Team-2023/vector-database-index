import numpy as np
from scipy.cluster.vq import kmeans2
from typing import Dict, List, Annotated
# from sklearn.cluster import MiniBatchKMeans
# import joblib


class IVFDB:
    # constructor
    def __init__(self, file_path="saved_db.csv", new_db=True):
        self.num_part = 32  # number of partitions
        self.centroids = None  # centroids of each partition
        # self.assignments = None  # assignments of each vector to a partition
        self.iterations = 32  # number of iterations for kmeans
        self.index = None
        self.file_path = file_path
        self.database_size = 0
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
        scores = []
        # get the top_k centroids
        k = int(1.5 * np.sqrt(len(self.centroids)))
        top_centroids = self._get_top_centroids(query, k)
        print("top_centroids:", top_centroids)
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
                fp = np.memmap(
                    f"./index/index_{centroid}.dta",
                    dtype="float32",
                    mode="r",
                    shape=(self.index[centroid].shape[0], 71),
                )
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

    def _build_index(self):
        # TODO: build index for the database
        print("Building index...")
        # read the database file from csv file
        # id_of_dataset = np.loadtxt(
        #     self.file_path, delimiter=",", skiprows=0, dtype=np.int32, usecols=0
        # )
        # print("id_of_dataset shape:", id_of_dataset.shape)
        dataset = np.loadtxt(
            self.file_path,
            delimiter=",",
            skiprows=0,
            dtype=np.float32,
            usecols=range(1, 71),
        )
        # print("dataset shape:", dataset.shape)
        # print("dataset[0]:", dataset[0])
        self.database_size = len(dataset)
        self.num_part = int(np.sqrt(self.database_size))

        print("num_part:", self.num_part)

        # using numpy
        self.index = np.empty(self.num_part, dtype=object)
        for i in range(self.num_part):
            # self.index[i] = np.memmap(
            #     f"./index/index_{i}.dta", dtype="float32", mode="w+"
            # )
            self.index[i] = []
        (self.centroids, assignments) = kmeans2(
            dataset, self.num_part, iter=self.iterations
        )
        # kmeans = MiniBatchKMeans(
        #     n_clusters=self.num_part,
        #     random_state=0,
        #     # batch_size=2 * 256,
        #     batch_size=len(id_of_dataset) // 20,
        #     #   max_iter=self.iterations,
        #     n_init="auto",
        # )
        # kmeans.fit(dataset)
        # for i in range(0, len(id_of_dataset), len(id_of_dataset) // 20):
        #     # print("i:", i)
        #     dataset = np.loadtxt(
        #         self.file_path,
        #         delimiter=",",
        #         skiprows=i,
        #         dtype=np.float32,
        #         usecols=range(1, 71),
        #         max_rows=len(id_of_dataset) // 20,
        #     )
        #     kmeans.partial_fit(dataset.astype(np.double))

        # # save the kmeans model using joblib
        # joblib.dump(kmeans, "./kmeans_model.joblib")
        # get the centroids and assignments
        # self.centroids = kmeans.cluster_centers_
        # assignments = kmeans.labels_
        # print("centroids shape:", self.centroids.shape)
        # print("assignments shape:", assignments.shape)
        for n, k in enumerate(assignments):
            # n is the index of the vector
            # k is the index of the cluster
            self.index[k].append(n)

        # convert the index to numpy array
        self.index = np.array(self.index)
        # self.index = np.array(self.index)
        # save the index clusters to .csv files
        # for i, cluster in enumerate(self.index):
        #     with open(f"./index/index_{i}.csv", "w") as fout:
        #         for n in cluster:
        #             fout.write(f"{id_of_dataset[n]},{','.join(map(str, dataset[n]))}\n")
        for i in range(len(self.index)):
            if len(self.index[i]) == 0:
                continue
            cluster = self.index[i]
            self.index[i] = np.memmap(
                f"./index/index_{i}.dta",
                dtype="float32",
                mode="w+",
                shape=(len(cluster), 71),
            )
            for n, id in enumerate(cluster):
                self.index[i][n][0] = id
                self.index[i][n][1:] = dataset[id]

    def _get_top_centroids(self, query, k):
        # find the nearest centroids to the query
        top_k_centroids = np.argsort(
            np.linalg.norm(self.centroids - np.array(query), axis=1)
        )
        # get the top_k centroids
        top_k_centroids = top_k_centroids[:k]
        return top_k_centroids