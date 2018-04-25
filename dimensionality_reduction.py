import numpy as np


def perform_svd(ratings_df, k):
    """Performs Single Value Decomposition on the given ratings dataframe,
    and returns the predicted values by multiplying the user and item matrices
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: The prediction matrix
    """
    from scipy.sparse.linalg import svds
    u, sigma, vt = svds(ratings_df, k=k)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(u, vt)
    # predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return predicted_ratings


def predict(config, ratings_df):
    """Performs prediction according to the configuration, and returns the prediction matrix
    :param config: The ConfigParser object
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: The prediction matrix
    """
    method = config.get('DIMENSIONALITY_REDUCTION', 'Method')
    if method == 'SVD':
        k = int(config.get('DIMENSIONALITY_REDUCTION', 'k'))
        return perform_svd(ratings_df, k)
