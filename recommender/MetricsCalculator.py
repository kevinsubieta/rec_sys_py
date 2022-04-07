import itertools

from surprise import accuracy
from collections import defaultdict

class MetricsCalculator:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def get_top_n(predictions, n=10, minimium_rating=3.0):
        top_n = defaultdict(list)

        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            if(estimated_rating >= minimium_rating):
                top_n[user_id].append((movie_id, estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[user_id] = ratings[:n]

        return top_n

    def hit_rate(top_n_predicted, left_out_predictions):
        hits = 0
        total = 0

        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_product_id = left_out[1]
            hit = False
            for product_id, predicted_rating in top_n_predicted[user_id]:
                if (left_out_product_id == product_id):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        return hits / total

    def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cut_off=0):
        hits = 0
        total = 0

        for user_id, left_out_product_id, actual_rating, estimated_rating, _ in left_out_predictions:
            if actual_rating >= rating_cut_off:
                hit = False
                for product_id, predicted_rating in top_n_predicted[user_id]:
                    if left_out_product_id == product_id:
                        hit = True
                        break
                if hit:
                    hits += 1

                total += 1

        return hits / total

    def rating_hit_rate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for user_id, left_out_product_id, actual_rating, estimated_rating, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hit = False
            for product_id, predictedRating in top_n_predicted[user_id]:
                if left_out_product_id == product_id:
                    hit = True
                    break
            if (hit) :
                hits[actual_rating] += 1

            total[actual_rating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])


    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
        summation = 0
        total = 0

        for user_id, left_out_product_id, actual_rating, estimated_rating, _ in left_out_predictions:
            hitRank = 0
            rank = 0
            for product_id, predictedRating in top_n_predicted[user_id]:
                rank = rank + 1
                if (left_out_product_id == product_id):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def user_coverage(top_n_predicted, num_users, rating_threshold=0):
        hits = 0
        for product_id in top_n_predicted.keys():
            hit = False
            for movieID, predictedRating in top_n_predicted[product_id]:
                if (predictedRating >= rating_threshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / num_users

    def diversity(topNPredicted, sims_algo):
        n = 0
        total = 0
        sims_matrix = sims_algo.compute_similarities()
        for user_id in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[user_id], 2)
            for pair in pairs:
                product_one = pair[0][0]
                product_two = pair[1][0]
                inner_id1 = sims_algo.trainset.to_inner_iid(str(product_one))
                inner_id2 = sims_algo.trainset.to_inner_iid(str(product_two))
                similarity = sims_matrix[inner_id1][inner_id2]
                total += similarity
                n += 1

        if n != 0:
            s = total / n
        else:
            s = total
        return 1-s

    def novelty(top_n_predicted, rankings):
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for rating in top_n_predicted[user_id]:
                product_id = rating[0]
                rank = rankings[product_id]
                total += rank
                n += 1

        if n!= 0:
            return total / n
        else:
            return total


