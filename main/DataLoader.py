import os
import csv
import sys
import re
from collections import defaultdict

from surprise import Dataset
from surprise import Reader

class DataLoader:

    buysPath = '../data/buys.csv'
    productsPath = '../data/products2.csv'
    ratingPath = '../data/ratings2.csv'
    userPath = '../data/users2.csv'

    productId_to_name = {}
    name_to_product_id = {}

    def load_products_latest_small(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        self.productId_to_name = {}
        self.name_to_product_id = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingPath, reader=reader)
        with open(self.productsPath, newline='', encoding='ISO-8859-1') as csvfile:
            productReader = csv.reader(csvfile)
            next(productReader)  # Skip header line
            for row in productReader:
                product_id = str(row[0])
                product_name = row[1]
                self.productId_to_name[product_id] = product_name
                self.name_to_product_id[product_name] = product_id

        return ratingsDataset

    def get_categories(self):
        categories = defaultdict(list)
        categories_ids = {}
        max_category_id = 0
        with open(self.productsPath, newline='', encoding='ISO-8859-1') as csvfile:
            product_reader = csv.reader(csvfile)
            next(product_reader)
            for row in product_reader:
                product_id = str(row[0])
                categories_list = row[3].split('|')
                categories_id_list = []
                for cat in categories_list:
                    if cat in categories_ids:
                        cat_id = categories_ids[cat]
                    else:
                        cat_id = max_category_id
                        categories_ids[cat] = cat_id
                        max_category_id += 1
                    categories_id_list.append(cat_id)
                categories[product_id] = categories_id_list

        for(product_id, categories_id_list) in categories.items():
            bitfield = [0] * max_category_id
            for cat_id in categories_id_list:
                bitfield[cat_id] = 1
            categories[product_id] = bitfield

        return categories

    def get_user_ratings(self, user_id_to_search):
        user_ratings = []
        hit_user = False
        with open(self.ratingPath, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                user_id = str(row[0])
                if user_id_to_search == user_id:
                    product_id = str(row[1])
                    rating = float(row[2])
                    user_ratings.append((product_id, rating))
        return user_ratings

    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingPath, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                product_id = str(row[1])
                ratings[product_id] += 1
        rank = 1
        for product_id, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[product_id] = rank
            rank += 1
        return rankings

    def get_movie_name(self, product_id):
        if product_id in self.productId_to_name:
            return self.productId_to_name[product_id]
        else:
            return ''


    def get_user_buys(self):
        pass

    def get_user_wish(self):
        pass