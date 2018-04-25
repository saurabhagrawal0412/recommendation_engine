import pandas as pd
import numpy as np


def test_config_parser():
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini")
    print config.sections()
    is_rating_normalization = config.getboolean('RATING_NORMALIZATION', 'RatingNormalization')
    print 'is_rating_normalization: ', is_rating_normalization
    rating_normalization_method = config.get('RATING_NORMALIZATION', 'RatingNormalizationMethod')
    print 'rating_normalization_method: ', rating_normalization_method


def display_data_frame(df):
    import os
    import webbrowser
    html = df.to_html(na_rep="")
    with open("review_matrix.html", "w") as f:
        f.write(html)
    full_filename = os.path.abspath("review_matrix.html")
    webbrowser.open("file://{}".format(full_filename))


def read_data():
    df = pd.read_csv('data/ml-100k/u2.test', sep='\t', lineterminator='\n')
    df.columns = ['user_id', 'movie_id', 'value', 'ts']
    del df['ts']
    ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)
    return ratings_df


def create_test_df():
    import random
    my_mat = [[0] * 5 for i in range(5)]
    for i in range(5):
        for j in range(5):
            my_mat[i][j] = (i * 5) + j + 1 + random.randint(-10, 10)
    df = pd.DataFrame(my_mat)
    print 'Printing matrix', my_mat
    print 'Printing dataframe', df

    print 'my_mat[0][2] ->', my_mat[0][2]
    print 'df[0][2] ->', df[0][2]
    return df


def test_df_std():
    df = create_test_df()
    std_df = df.std(axis=1)
    print std_df
    df = df.divide(std_df, axis='index')
    print df


def test_df_mean():
    df = create_test_df()
    mean_df = df.mean(axis=1)
    mean_df = mean_df.transpose()
    print mean_df
    df = df.subtract(mean_df, axis='index')
    print df


def create_test_ratings():
    import random
    USERS = 50
    MOVIES = 10
    RATING = 5
    TS = 878543075
    rating_set = set()

    while len(rating_set) < 100:
        user_id = random.randint(1, USERS)
        movie_id = random.randint(1, MOVIES)
        rating = random.randint(1, RATING)
        rating_set.add((user_id, movie_id, rating, TS))

    rating_list = list()
    for rating in rating_set:
        rating_list.append(list(rating))

    for rating in rating_list:
        print '\t'.join(map(str, rating))


def test_cosine_similarity():
    from sklearn.metrics.pairwise import cosine_similarity
    test_df = create_test_df()
    test_df[2][2] = None
    print test_df
    cosine_sim_df = cosine_similarity(test_df.fillna(0))
    print cosine_sim_df


def test_pearson_correlation():
    test_df = create_test_df()
    test_df[2][1] = None
    print test_df
    cor_df = test_df.T.corr(method='pearson')
    print cor_df


def test_spearman_correlation():
    test_df = create_test_df()
    test_df[2][2] = None
    print test_df
    cor_df = test_df.T.corr(method='spearman')
    print cor_df


def sort_by_index_test():
    df = pd.DataFrame(np.random.randn(8, 10))
    for col in range(5):
        df[1][col] = None
    from data_reader import display_dataframe
    display_dataframe(df, 'df.html')
    idx = np.argsort(df.fillna(-float('inf')), axis=1)
    display_dataframe(idx, 'idx.html')

    top_n = 5
    nearest = list()

    rows, cols = idx.shape
    for row_idx in range(rows):
        curr = list()
        for i in range(cols-1, cols-top_n-1, -1):
            col_idx = idx[i][row_idx]
            if not df[col_idx][row_idx]:
                break
            curr.append(col_idx)
        nearest.append(curr)
    print nearest


def iloc_test():
    df = pd.DataFrame(np.random.randn(4, 6))
    print 'Printing df:\n', df
    print 'Printing df[3][1]:', df[3][1]
    print 'Printing df.iloc[-1][1]:', df.iloc[-1][1]


def test_indices_of_pivot_table():
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini")
    training_file = config.get('RATINGS_TRAINING_FILE', 'Location')
    separator = config.get('RATINGS_TRAINING_FILE', 'Separator')
    separator = '\t'
    line_terminator = config.get('RATINGS_TRAINING_FILE', 'LineTerminator')
    line_terminator = '\n'

    df = pd.read_csv(training_file, sep=separator, lineterminator=line_terminator)
    columns_str = config.get('RATINGS_TRAINING_FILE', 'Columns')
    columns = columns_str.split(',')
    df.columns = columns
    # del df['ts']
    ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', values='value', aggfunc=np.max)
    print 'Printing ratings_df:'
    # for idx, row in ratings_df.iterrows():
    #     print 'Index ->', idx, 'Row ->\n', row
    for row_idx, row in ratings_df.iterrows():
        print row_idx
        for col_idx, cell in row.iteritems():
            print col_idx, cell
    # cols = ratings_df.columns
    # for col in cols:
    #     print col, type(col)
    # print ratings_df
    # print ratings_df[long(2)][long(4)]


def svd_test():
    from scipy.sparse.linalg import svds
    m, n = 6, 3
    a = np.random.randn(m, n)  # + 1.j * np.random.randn(m, n)
    print a
    U, sigma, Vt = svds(a, k=2)


def main():
    print 'Hello'
    sort_by_index_test()


if __name__ == '__main__':
    main()
