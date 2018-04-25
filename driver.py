#!/usr/bin/env python
# Driver program for the recommender system
# Usage:
#   python driver.py [file path of the config file]
#

from configparser import ConfigParser
from data_reader import get_ratings_df
from rating_normalization import perform_normalization
from similarity_computation import compute_similarity
from neighbor_selection import select_neighbors
import dimensionality_reduction
from validation import compute_validation_error
import prediction
import sys


def main():
    """The main method of the recommendation engine
    """
    config = ConfigParser()
    config_file = sys.argv[1]
    config.read(config_file)
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
