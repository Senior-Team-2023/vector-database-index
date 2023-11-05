from typing import Dict, List, Annotated
import numpy as np


class VecDBWorst:
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
            for row in rows:
                # get id and embed from dictionary
                id, embed = row["id"], row["embed"]
                # convert row to string to write it to the database file
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
        self._build_index()

    # access database to retrive top_k records according to query
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        scores = []
        # open database file to read
        with open(self.file_path, "r") as fin:
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
        # return the ids of the top_k records
        return [s[1] for s in scores]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        pass
