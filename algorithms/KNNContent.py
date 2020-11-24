
from surprise import AlgoBase
from surprise import PredictionImpossible

import math
import numpy as np
import heapq


from main.DataLoader import DataLoader


class KNN(AlgoBase):
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Compute item similarity matrix based on content attributes
        # Load up genre vectors for every movie
        data_loader = DataLoader()
        categories = data_loader.get_categories()
        print("Computing content-based similarity matrix...")

        # Compute genre distance for every product combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for first_rating in range(self.trainset.n_items):
            if first_rating % 100 == 0:
                print(first_rating, " of ", self.trainset.n_items)
            for second_rating in range(first_rating + 1, self.trainset.n_items):
                first_product_id = int(self.trainset.to_raw_iid(first_rating))
                second_product_id = int(self.trainset.to_raw_iid(second_rating))
                categories_similarity = self.compute_categories_similarity(first_product_id, second_product_id, categories)

                self.similarities[first_rating, second_rating] = categories_similarity

        print("...done.")
        return self

    def compute_categories_similarity(self, product_one, product_two, categories):
        category_product_one = categories[product_one]
        category_product_two = categories[product_two]

        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(category_product_one)):
            x = category_product_one[i]
            y = category_product_two[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy / math.sqrt(sumxx*sumyy)

    def estimate(self, u, i):
        if not self.trainset.knows_user(u) and self.trainset.knows_item(i):
            raise PredictionImpossible('Item is unkown')

        if type(i) is not int:
            raise PredictionImpossible('Item is unkown')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i, rating[0]]
            neighbors.append((genre_similarity, rating[1]))
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda  t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        sim_total = weighted_sum = 0
        for sim_score, rating in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating

        if sim_total == 0:
            raise PredictionImpossible('No neighbors')

        prediction_rating = weighted_sum / sim_total

        return prediction_rating


