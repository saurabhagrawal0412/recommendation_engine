#!/usr/bin/env python
# Driver program for the recommender system
# Usage:
#   python driver.py [file path of the config file]
#

import argparse
from configparser import ConfigParser
import sys

from data_reader import get_ratings_df
import dimensionality_reduction
from neighbor_selection import select_neighbors
from rating_normalization import perform_normalization
import prediction
from similarity_computation import compute_similarity
from validation import compute_validation_error


def main():
    """The main method of the recommendation engine
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Location of the config file')
    config = ConfigParser()
    config.read(parser.parse_args().config_file)
    train_ratings_df, test_ratings_df = get_ratings_df(config)
    train_ratings_df = perform_normalization(config, train_ratings_df)
    similarity_df = compute_similarity(config, train_ratings_df)
    nearest_dict = select_neighbors(config, similarity_df)

    # predicted_ratings = dimensionality_reduction.predict(config, ratings_df)
    # print predicted_ratings.view()
    prediction_df = prediction.get_avg_rating_recc(train_ratings_df, nearest_dict)
    validation_error = compute_validation_error(config, test_ratings_df, prediction_df)
    print 'Validation error ->', validation_error


if __name__ == '__main__':
    main()
