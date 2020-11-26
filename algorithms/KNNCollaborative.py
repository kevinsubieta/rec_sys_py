import math
import numpy as np
import heapq

from surprise import AlgoBase
from surprise import PredictionImpossible

from main.DataUserLoader import DataUserLoader


class KNNCollaborative(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Compute item similarity matrix based on content attributes
        # Load up genre vectors for every movie
        data_loader = DataUserLoader()
        cities = data_loader.get_cities()
        gender = data_loader.get_user_sex()
        ages = data_loader.get_user_age()
        print("Computing content-based similarity matrix...")

        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        for thisRating in range(self.trainset.n_items):
            if thisRating % 100 == 0:
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating + 1, self.trainset.n_items):
                first_user_id = int(self.trainset.to_raw_iid(thisRating))
                second_user_id = int(self.trainset.to_raw_iid(otherRating))

                cities_similarity = self.compute_city_similarity(first_user_id, second_user_id, cities)
                gender_similarity = self.compute_sex_similarity(first_user_id, second_user_id, gender)
                ages_similarity = self.compute_age_similarity(first_user_id, second_user_id, ages)

                self.similarities[thisRating, otherRating] = cities_similarity * 2.0 + \
                                                                   gender_similarity * 2.0 + \
                                                                   abs(((ages_similarity * 2) / 100) - 2)


        print("...done.")
        return self


    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        neighbors = []
        for users in self.trainset.ur[u]:
            usersSimilarity = self.similarities[i,users[0]]
            neighbors.append( (usersSimilarity, users[1]) )

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating


    def compute_city_similarity(self, subject_one, subject_two, cities):
        city_subject_one = cities[subject_one]
        city_subject_two = cities[subject_two]

        sumxx, sumxy, sumyy = 0, 0, 0

        if city_subject_one == 1 and city_subject_one == city_subject_two:
            return 1
        else:
            return 0

    def compute_age_similarity(self, subject_one, subject_two, years):
        diff = abs(years[subject_one] - years[subject_two])
        sim = math.exp(-diff / 10.0)
        return sim

    def compute_sex_similarity(self, subject_one, subject_two, sex):
        sex_subj_one = sex[subject_one]
        sex_subj_two = sex[subject_two]
        if sex_subj_one == sex_subj_two:
            return 1
        else:
            return 0
