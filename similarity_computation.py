# Contains various methods to compute similarity between all pairs of users/items


def compute_cosine_sim(ratings_df):
    """Computes the cosine similarity between all pairs of users and returns it in the form of a pandas dataframe
    :param ratings_df: Numpy asymmetric matrix with user-id as rows and movie-id as columns or vice-versa
    :return: (n X n) Dataframe containing the cosine similarity between the n users/items
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(ratings_df.fillna(0))


def compute_pearson_corr(ratings_df):
    """Computes the Pearson correlation between all pairs of users and returns it in the form of a pandas dataframe
    :param ratings_df: Numpy asymmetric matrix with user-id as rows and movie-id as columns or vice-versa
    :return: (n X n) Dataframe containing the Pearson correlation between the n users/items
    """
    return ratings_df.T.corr(method='pearson')


def compute_spearman_rank_corr(ratings_df):
    """Computes the Spearman rank correlation between all pairs of users and
       returns it in the form of a pandas dataframe
    :param ratings_df: Numpy asymmetric matrix with user-id as rows and movie-id as columns or vice-versa
    :return: (n X n) Dataframe containing the Spearman Rank correlation between the n users/items
    """
    return ratings_df.T.corr(method='spearman')


def compute_similarity(config, ratings_df):
    """Computes the similarity between all pairs of users according to the similarity method defined in the config file
       and returns it in the form of a pandas dataframe
    :param config: The ConfigParser object
    :param ratings_df: Dataframe with user-id as rows and movie-id as columns
    :return: (n X n) Dataframe containing the similarity between the n users/items
    """
    method = config.get('SIMILARITY_WEIGHT_COMPUTATION', 'Method')
    user_or_item = config.get('SIMILARITY_WEIGHT_COMPUTATION', 'UserOrItem')
    df = ratings_df if user_or_item == 'User' else ratings_df.transpose()
    if method == 'Cosine':
        return compute_cosine_sim(df)
    elif method == 'Pearson':
        return compute_pearson_corr(df)
    elif method == 'Spearman':
        return compute_spearman_rank_corr(df)
