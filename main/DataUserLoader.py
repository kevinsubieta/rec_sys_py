import os
import csv
import sys
import re
import math
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise.dataset import DatasetAutoFolds


class DataUserLoader:
    userPath = '../data/users.csv'
    users_ratings_similarity_path = '../data/users_similarity.csv'

    userID_to_name = {}
    name_to_userID = {}

    def load_users_latest_small(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        self.userID_to_name = {}
        self.name_to_userID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        user_dataset = Dataset.load_from_file(self.userPath, reader=reader)

        with open(self.userPath, newline='', encoding=None) as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    user_id = int(row[0])
                    user_name = row[1]
                    self.userID_to_name[user_id] = user_name
                    self.name_to_userID[user_name] = user_id

        return user_dataset

    def load_users_similarities_small(self):
        self.load_users_latest_small()
        users_similarities = self.get_user_similarities()
        self.build_file(users_similarities, self.users_ratings_similarity_path)
        reader = Reader(line_format='user item rating rating', sep=',', skip_lines=1)
        users_similarities_dataset = Dataset.load_from_file(self.users_ratings_similarity_path, reader=reader)
        users_similarities_rating = self.build_user_similarities_data_list(users_similarities)
        return users_similarities_dataset


    def build_file(self, array, path):
        # data_frame = pd.DataFrame(array)
        # data_frame.drop(0)
        # data_frame.to_csv(path)
        a = np.asarray(array)
        np.savetxt(path, array, fmt=['%i','%i','%.2f'], delimiter=",", header='user_id, other_user_id, rating')

    def get_cities(self):
        cities = defaultdict(list)
        cities_ids = {'Santa Cruz': 0, 'Beni': 1, 'Tarija': 2, 'Pando': 3, 'Chuquisaca': 4, 'Oruro': 5,
                      'Potosí': 6, 'Cochabamba': 7, 'La Paz': 8}
        with open(self.userPath, newline='', encoding=None) as csvfile:
            user_reader = csv.reader(csvfile)
            next(user_reader)
            for row in user_reader:
                user_id = int(row[0])
                city_name = row[4]
                city_id = int(cities_ids[city_name])
                cities[user_id] = city_id

        return cities

    def get_user_age(self):
        years = defaultdict(int)
        with open(self.userPath, newline='', encoding=None) as csvfile:
            user_reader = csv.reader(csvfile)
            next(user_reader)
            for row in user_reader:
                user_id = int(row[0])
                year = row[2]
                if year:
                    years[user_id] = int(year)
        return years


    def get_user_sex(self):
        user_sex = defaultdict(list)
        sex_ids = {'M': 0, 'F': 1}
        with open(self.userPath, newline='', encoding='ISO-8859-1') as csvfile:
            user_reader = csv.reader(csvfile)
            next(user_reader)
            for row in user_reader:
                user_id = int(row[0])
                sex_char = row[3]
                sex_id = int(sex_ids[sex_char])
                user_sex[user_id] = int(sex_id)

        return user_sex

    def get_user_similarities(self):
        cities = self.get_user_sex()
        gender = self.get_user_sex()
        ages = self.get_user_age()
        similarities = []
        users = []

        with open(self.userPath, newline='', encoding='ISO-8859-1') as csvfile:
            user_reader = csv.reader(csvfile)
            next(user_reader)
            for user in user_reader:
                users.append(user)

        for current_user in users:
            for iterate_user in users:
                first_user_id = int(current_user[0])
                second_user_id = int(iterate_user[0])

                if first_user_id != second_user_id:
                    cities_similarity = self.compute_city_similarity(first_user_id, second_user_id, cities)
                    gender_similarity = self.compute_sex_similarity(first_user_id, second_user_id, gender)
                    ages_similarity = self.compute_age_similarity(first_user_id, second_user_id, ages)

                    similarity = cities_similarity * 2.0 + \
                                 gender_similarity * 1.0 + \
                                 abs(((ages_similarity * 2) / 100) - 2)

                    similarities.append((int(first_user_id), int(second_user_id), similarity))

        return similarities

    def build_user_similarities_data_list(self, user_similarities):
        user_similarities_dict = defaultdict(list)
        for similatity in user_similarities:
            user_similarities_dict[similatity[0]].append(similatity)

        return user_similarities_dict



    def compute_city_similarity(self, subject_one, subject_two, cities):
        city_subject_one = cities[subject_one]
        city_subject_two = cities[subject_two]

        sumxx, sumxy, sumyy = 0, 0, 0

        if city_subject_one == city_subject_two:
            return 1
        else:
            return 0


    def compute_age_similarity(self, subject_one, subject_two, years):
        return abs(years[subject_one] - years[subject_two])

    def compute_sex_similarity(self, subject_one, subject_two, sex):
        sex_subj_one = sex[subject_one]
        sex_subj_two = sex[subject_two]
        if sex_subj_one == sex_subj_two:
            return 1
        else:
            return 0


    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.users_ratings_similarity_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                user_id = int(row[0])
                ratings[user_id] += 1
        rank = 1
        for user_id, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[user_id] = rank
            rank += 1
        return rankings

    def get_user_name(self, user_id):
        if user_id in self.userID_to_name:
            return self.userID_to_name[user_id]
        else:
            return ''

    def get_user_id(self, user_name):
        if user_name in self.name_to_userID:
            return self.name_to_userID[user_name]
        else:
            return 0



