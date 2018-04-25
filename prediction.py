# Contains the methods to compute the predicted ratings

import math


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
    return None if total_raters == 0 else total_rating/total_raters


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
