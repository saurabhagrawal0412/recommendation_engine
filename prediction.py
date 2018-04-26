# Contains the methods to compute the predicted ratings

from matrix_factorization import perform_matrix_factorization

import math
import numpy as np
import pandas as pd


def get_avg_rating(ratings_df, nearest_list, item_id):
    """Determines the average rating for the user/item in nearest list for the given item id
    :param ratings_df: Pivot table (dataframe) that stores the ratings between users and items
    :param nearest_list: List that stores the user_id/item_id for the nearest user/item
    :param item_id: Item for which the average rating needs to be determined
    :return: Float average rating if such a rating exists, None otherwise
    """
    total_rating = 0.0
    total_raters = 0
    for nearest in nearest_list:
        total_rating += ratings_df.ix[long(nearest)][long(item_id)]
        total_raters += 1
    return total_rating/total_raters if total_raters > 0 else None


def get_avg_rating_recc(ratings_df, nearest_dict):
    """Computes average ratings from the nearest users/items and returns it in the form of a dataframe
    :param ratings_df: Pivot table (dataframe) that stores the ratings between users and items
    :param nearest_dict: Dictionary that maps a user/item to a list of nearest users/items
    :return: A pivot table that stores the ratings of the new items
    """
    recc_df = ratings_df.copy(deep=True)

    for row_idx, row in recc_df.iterrows():
        nearest = nearest_dict[row_idx]
        for col_idx, cell in row.iteritems():
            if math.isnan(cell):
                recc_df.ix[row_idx][col_idx] = get_avg_rating(ratings_df, nearest, col_idx)
    return recc_df


def get_similarity_weighted_rating(ratings_df, similarity_df, curr_id, nearest_list, item_id):
    """Determines the similarity weighted rating for the user/item in nearest list for the given item id
    :param ratings_df: Pivot table (dataframe) that stores the ratings between users and items
    :param similarity_df: Pivot table (dataframe) that stores the similarity between all pairs of users/items
    :param curr_id: The curr user/item for which the recommended ratings need to be determined
    :param nearest_list: List that stores the user_id/item_id for the nearest user/item
    :param item_id: Item for which the average rating needs to be determined
    :return: Float average rating if such a rating exists, None otherwise
    """
    total_rating = 0.0
    total_similarity = 0.0
    for nearest in nearest_list:
        curr_similarity = similarity_df.ix[long(curr_id)][long(nearest)]
        total_rating += (ratings_df.ix[long(nearest)][long(item_id)] * curr_similarity)
        total_similarity += curr_similarity
    return total_rating/total_similarity if total_similarity > 0 else None


def get_similariy_rating_recc(ratings_df, similarity_df, nearest_dict):
    """Computes similarity weighted ratings from the nearest users/items and returns it in the form of a dataframe
    :param ratings_df: Pivot table (dataframe) that stores the ratings between users and items
    :param similarity_df: Pivot table (dataframe) that stores the similarity between all pairs of users/items
    :param nearest_dict: Dictionary that maps a user/item to a list of nearest users/items
    :return: A pivot table that stores the ratings of the new items
    """
    recc_df = ratings_df.copy(deep=True)
    for row_idx, row in recc_df.iterrows():
        nearest = nearest_dict[row_idx]
        for col_idx, cell in row.iteritems():
            if math.isnan(cell):
                recc_df.ix[row_idx][col_idx] = get_similarity_weighted_rating(ratings_df, similarity_df, row_idx,
                                                                              nearest, col_idx)
    return recc_df


def get_svd_recc(ratings_df, num_features, regularization_amount):
    """Performs Single Value Decomposition on the given ratings dataframe,
    and returns the predicted values by multiplying the user and item matrices
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :param num_features: Number of latent features to generate
    :param regularization_amount: How much regularization to apply
    :return: The pivot table (DataFrame) storing the predicted ratings
    """
    u, vt = perform_matrix_factorization(ratings_df.values, num_features, regularization_amount)
    predicted_ratings = np.dot(u, vt)
    row_index = list(ratings_df.index.values)
    col_index = list(ratings_df.columns)
    prediction_df = pd.DataFrame(predicted_ratings, index=row_index, columns=col_index)
    return prediction_df


def predict(config, ratings_df, similarity_df, nearest_dict):
    """Computes the prediction ratings as per the method in the config file
    :param config: The ConfigParser object
    :param ratings_df: The pivot table dataframe that stores the ratings given by users to the items
    :param similarity_df: Dataframe containing the similarity between the n users/items
    :param nearest_dict: A dictionary that maps a user/item to at most n nearest users/items
    :return: The pivot table dataframe that stores the predicted ratings
    """
    method = config.get('PREDICTION', 'Method')
    if method == 'AvgRating':
        return get_avg_rating_recc(ratings_df, nearest_dict)
    elif method == 'SimWeightedRating':
        return get_similariy_rating_recc(ratings_df, similarity_df, nearest_dict)
    elif method == 'SVD':
        num_features = int(config.get('PREDICTION', 'SVDNumFeatures'))
        regularization_amount = float(config.get('PREDICTION', 'SVDRegAmt'))
        return get_svd_recc(ratings_df, num_features, regularization_amount)
