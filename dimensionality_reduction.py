# Utility script tp compute

from matrix_factorization import perform_matrix_factorization
import numpy as np
import pandas as pd


def perform_svd(ratings_df, k):
    """Performs Single Value Decomposition on the given ratings dataframe,
    and returns the predicted values by multiplying the user and item matrices
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :param k: The number of parameters
    :return: The pivot table (DataFrame) storing the predicted ratings
    """
    # from scipy.sparse.linalg import svds
    # u, sigma, vt = svds(ratings_df, k=k)
    # sigma = np.diag(sigma)
    u, vt = perform_matrix_factorization(ratings_df.values)
    predicted_ratings = np.dot(u, vt)
    row_index = list(ratings_df.index.values)
    col_index = list(ratings_df.columns)
    prediction_df = pd.DataFrame(predicted_ratings, index=row_index, columns=col_index)
    return prediction_df

