from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline


# Class to create TrainSet and AntiTrainSet
class EvaluationData:

    def __init__(self, data, popularity_ratings):
        self.rankings = popularity_ratings

        # Build a full training set for evaluating overall properties
        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.train_set, self.test_set = train_test_split(data, test_size=.25, random_state=1)

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test

        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

        # Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.full_train_set)

    def get_full_train_set(self):
        return self.full_train_set

    def get_full_anti_test_set(self):
        return self.full_anti_test_set

    def get_anti_test_set_for_simillarities_users(self, test_subject):
        trainset = self.full_train_set
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items()]
        return anti_testset

    def get_anti_test_set_for_user(self, test_subject):
        trainset = self.full_train_set
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]
        return anti_testset

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    def get_LOOCV_train_set(self):
        return self.LOOCVTrain

    def get_LOOCV_test_set(self):
        return self.LOOCVTest

    def get_LOOCV_anti_test_set(self):
        return self.LOOCVAntiTestSet

    def get_similarities(self):
        return self.simsAlgo

    def get_popularity_rankings(self):
        return self.rankings
