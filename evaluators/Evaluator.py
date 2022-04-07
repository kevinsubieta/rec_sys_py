from evaluators.EvaluatedAlgorithm import EvaluatedAlgorithm
from evaluators.EvaluationData import EvaluationData


class Evaluator:
    algorithms = []

    def __init__(self, dataset, rankings):
        evaluation_data = EvaluationData(dataset, rankings)
        self.dataset = evaluation_data

    def add_Algorithm(self, algorithm, name):
        algorithm = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(algorithm)

    def evaluate(self, do_top_N):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.get_name(), "...")
            results[algorithm.get_name()] = algorithm.evaluate(self.dataset, do_top_N)

        # Print results
        print("\n")

        if (do_top_N):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (do_top_N):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def show_top_N_recommendation(self, data_root, test_subject="610963a16add0e1408837c7d", k=10):
        for algorithm in self.algorithms:
            print('Using recommender', algorithm.get_name())

            print('Building recommendation model...')
            trainset = self.dataset.get_full_train_set()
            algorithm.get_algorithm().fit(trainset)

            print("Computing recommendations...")
            test_set = self.dataset.get_anti_test_set_for_user(test_subject)
            predictions = algorithm.get_algorithm().test(test_set)

            recommendations = []

            print("\nBasado en la puntuación, se recomienda:")
            for user_id, product_id, actual_rating, estimated_rating, _ in predictions:
                int_product_id = product_id
                recommendations.append((int_product_id, estimated_rating, user_id))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(data_root.get_movie_name(ratings[0]), ratings[1])


