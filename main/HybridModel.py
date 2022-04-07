from surprise import SVD, SVDpp
from surprise import NormalPredictor
from surprise.model_selection import GridSearchCV
from algorithms.KNNContent import KNN
from surprise import KNNBasic
from algorithms.KNNCollaborative import KNNCollaborative

import random
import numpy as np

from evaluators.Evaluator import Evaluator
from evaluators.UserEvaluator import UserEvaluator
from main.DataLoader import DataLoader
from main.DataUserLoader import DataUserLoader


def load_model_data():
    print("Loading products ratings...")
    rating_data_loader = DataLoader()
    data = rating_data_loader.load_products_latest_small()
    ratings = rating_data_loader.get_popularity_ranks()
    return (rating_data_loader, data, ratings)

def load_user_model_data():
    print("Loading users ratings...")
    user_data_loader = DataUserLoader()
    data = user_data_loader.load_users_similarities_small()
    ratings = user_data_loader.get_popularity_ranks()
    return (user_data_loader, data, ratings)

np.random.seed(0)
random.seed(0)

(rating_data_loader, evaluation_dataset, ratings) = load_model_data()
(user_data_loader, user_evaluation_dataset, user_ratings  ) = load_user_model_data()

# -------------------  PARAMETERS FOR SVD ---------------------
param_grid = {'n_epochs': [20, 40], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluation_dataset)
# ------------------------------------------------------------------------------
evaluator = Evaluator(evaluation_dataset, ratings)
user_evaluator = UserEvaluator(user_evaluation_dataset, user_ratings)

# ----------------------  TIME FOR SVD based on Ratings ------------------------
SVD_plus_pls = SVDpp()
evaluator.add_Algorithm(SVD_plus_pls, 'SVD++')
# ------------------------------------------------------------------------------

# ----------------------  TIME FOR Content KNN based in products ----------------
KNNContennt = KNN()
evaluator.add_Algorithm(KNNContennt, 'ContentKNN')
# ------------------------------------------------------------------------------

# --------------  TIME FOR Collaborative KNN based in Users ------------
# User-based KNN
#- UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
# evaluator.add_Algorithm(UserKNN, "User KNN")

KnnUser = KNNCollaborative()
user_evaluator.add_Algorithm(KnnUser, 'UsersSimilarityKNN')
# ------------------------------------------------------------------------------
evaluator.evaluate(True)
evaluator.show_top_N_recommendation(rating_data_loader)

user_evaluator.evaluate(True)
user_evaluator.show_user_similarities(user_data_loader)
