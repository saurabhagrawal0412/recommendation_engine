# Contains the various methods to select (nearest) neighbors for a user/item

import math
from sorted_list import SortedList


def perform_top_n_filtering(similarity_df, n):
    """Finds the n most similar user/item, and returns it in the form of a list of tuples
    :param similarity_df: Pandas dataframe representing pairwise similarities between m users/items
    :param n: The number of neighbors
    :return: A dictionary that maps a user/item to at most n nearest users/items
    """
    neighbor_dict = dict()
    for row_idx, row in similarity_df.iterrows():
        nearest = SortedList(n)
        for col_idx, cell in row.iteritems():
            if not (math.isnan(cell) or row_idx == col_idx):
                nearest.insert(cell, col_idx)
        neighbor_dict[row_idx] = nearest.get_all()
    # print neighbor_list
    return neighbor_dict


def perform_threshold_filtering(similarity_df, threshold):
    """Finds the user/item whose similarity >= threshold, and returns it in the form of a list of tuples
    :param similarity_df: Pandas dataframe representing pairwise similarities between m users/items
    :param threshold: The similarity threshold
    :return: A list of tuples that maps a user/item to at most n nearest users/items
    """
    neighbor_dict = dict()
    cols = len(similarity_df.columns)
    for row_idx, row in similarity_df.iterrows():
        nearest = SortedList(cols)
        for col_idx, cell in row.iteritems():
            if not (math.isnan(cell) or row_idx == col_idx or cell < threshold):
                nearest.insert(cell, col_idx)
        neighbor_dict[row_idx] = nearest.get_all()
    # print neighbor_list
    return neighbor_dict


def select_neighbors(config, similarity_df):
    """Selects the user/item according to the parameters specified in the config file
    :param config: The ConfigParser object
    :param similarity_df: Pandas dataframe representing pairwise similarities between m users/items
    :return: A dictionary that maps a user_id/item_id to neighboring users/items
    """
    method = config.get('NEIGHBOR_SELECTION', 'FilteringMethod')
    if method == 'TopN':
        n = int(config.get('NEIGHBOR_SELECTION', 'N'))
        return perform_top_n_filtering(similarity_df, n)
    elif method == 'Threshold':
        threshold = float(config.get('NEIGHBOR_SELECTION', 'Threshold'))
        return perform_threshold_filtering(similarity_df, threshold)


def print_nearest_neighbors(nearest_users):
    """Prints the nearest neighbor dictionary
    :param nearest_users: A dictionary that maps a user_id/item_id to neighboring users/items
    """
    print 'Printing nearest users:'
    for user_id, nearest in nearest_users.iteritems():
        print 'User id:', user_id
        print nearest
