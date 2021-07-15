from evaluators import EvaluationData
from recommender.MetricsCalculator import MetricsCalculator


class EvaluatedAlgorithm:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def evaluate(self, evaluation_data, do_top_N,  n=10, verbose=True):
        metrics = {}
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluation_data.get_train_set())
        predictions = self.algorithm.test(evaluation_data.get_test_set())
        metrics['RMSE'] = MetricsCalculator.RMSE(predictions)
        metrics['MAE'] = MetricsCalculator.MAE(predictions)

        if do_top_N:
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluation_data.get_LOOCV_train_set())
            left_out_predictions = self.algorithm.test(evaluation_data.get_LOOCV_test_set())
            all_predictions = self.algorithm.test(evaluation_data.get_LOOCV_anti_test_set())
            # LOOK THAT!
            top_n_predicted = MetricsCalculator.get_top_n(all_predictions, n)

            if (verbose):
                print("Computing hit-rate and rank metrics...")

            metrics['HR'] = MetricsCalculator.hit_rate(top_n_predicted, left_out_predictions)
            metrics["cHR"] = MetricsCalculator.cumulative_hit_rate(top_n_predicted, left_out_predictions)
            metrics["ARHR"] = MetricsCalculator.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions)

            #Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")

            self.algorithm.fit(evaluation_data.get_full_train_set())
            all_predictions = self.algorithm.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = MetricsCalculator.get_top_n(all_predictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 3.0:
            metrics['Coverage'] = MetricsCalculator.user_coverage(top_n_predicted,
                                                                   evaluation_data.get_full_train_set().n_users,
                                                                   rating_threshold=4.0)

            metrics['Diversity'] = MetricsCalculator.diversity(top_n_predicted,
                                                               evaluation_data.get_similarities())

            metrics['Novelty'] = MetricsCalculator.novelty(top_n_predicted,
                                                             evaluation_data.get_popularity_rankings())

            if (verbose):
                print("Analysis complete.")

            return metrics

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm