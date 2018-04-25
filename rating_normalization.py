# Contains various methods to perform rating normalization for a recommender system


def perform_user_mean_centering(ratings_df):
    """Performs mean-centering on the given ratings matrix by subtracting the mean rating given by the user
       from the given rating
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: The mean centered ratings_df
    """
    mean_ratings_df = ratings_df.mean(axis=1)
    ratings_df = ratings_df.subtract(mean_ratings_df, axis='index')
    return ratings_df


def perform_z_score_normalization(ratings_df):
    """Performs z-score normalization on the given ratings matrix by subtracting the mean rating given by the user
       from the given rating and dividing the standard deviation
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: The normalized ratings_df
    """
    mean_ratings_df = ratings_df.mean(axis=1)
    std_ratings_df = ratings_df.std(axis=1)
    ratings_df = ratings_df.subtract(mean_ratings_df, axis='index')
    ratings_df = ratings_df.divide(std_ratings_df, axis='index')
    return ratings_df


def perform_normalization(config, ratings_df):
    """Checks if normalization flag is enabled or not. If it is, this method calls the appropriate normalization
    method and returns the ratings dataframe
    :param config: The ConfigParser object
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: The normalized ratings_df
    """
    if config.getboolean('RATING_NORMALIZATION', 'IsEnabled'):
        method = config.get('RATING_NORMALIZATION', 'Method')
        if method == 'MeanCentering':
            ratings_df = perform_user_mean_centering(ratings_df)
        elif method == 'ZScoreNormalization':
            ratings_df = perform_z_score_normalization(ratings_df)
    return ratings_df
