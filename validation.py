# Contains the methods to compute validation error for a recommender system
# by comparing the test (hidden) ratings and the predicted ratings

import math


def compute_mean_absolute_error(ratings_df, predicted_df):
    """Compares the actual and the predicted ratings and computes the mean absolute error
    :param ratings_df: The pivot table dataframe that stores the ratings given by users to the items
    :param predicted_df: The pivot table dataframe that stores the predicted ratings
    :return: The float mean absolute error
    """
    total_error = 0.0
    point_count = 0
    for (rat_row_idx, rat_row), (pred_row_idx, pred_row) in zip(ratings_df.iterrows(), predicted_df.iterrows()):
        for (rat_col_idx, rat_cell), (pred_col_idx, pred_cell) in zip(rat_row.iteritems(), pred_row.iteritems()):
            if not (math.isnan(rat_cell) or math.isnan(pred_cell)):
                total_error += abs(pred_cell - rat_cell)
                point_count += 1
    # print 'Total error ->', total_error, 'Point count ->', point_count
    return total_error/point_count if point_count != 0 else 0


def compute_root_mean_squared_error(ratings_df, predicted_df):
    """Compares the actual and the predicted ratings and computes the root mean squared error
    :param ratings_df: The pivot table dataframe that stores the ratings given by users to the items
    :param predicted_df: The pivot table dataframe that stores the predicted ratings
    :return: The float root mean squared error
    """
    total_sq_error = 0.0
    point_count = 0
    for (rat_row_idx, rat_row), (pred_row_idx, pred_row) in zip(ratings_df.iterrows(), predicted_df.iterrows()):
        for (rat_col_idx, rat_cell), (pred_col_idx, pred_cell) in zip(rat_row.iteritems(), pred_row.iteritems()):
            if not (math.isnan(rat_cell) or math.isnan(pred_cell)):
                total_sq_error += (pred_cell - rat_cell) ** 2
                point_count += 1
    # print 'Total squared error ->', total_sq_error, 'Point count ->', point_count
    return math.sqrt(total_sq_error/point_count) if point_count != 0 else 0


def compute_validation_error(config, ratings_df, predicted_df):
    """Compares the actual and the predicted ratings and computes the error as per the config file
    :param config: The ConfigParser object
    :param ratings_df: The pivot table dataframe that stores the ratings given by users to the items
    :param predicted_df: The pivot table dataframe that stores the predicted ratings
    :return: The float error
    """
    method = config.get('VALIDATION', 'Method')
    if method == 'MAE':
        return compute_mean_absolute_error(ratings_df, predicted_df)
    elif method == 'RMSE':
        return compute_root_mean_squared_error(ratings_df, predicted_df)
