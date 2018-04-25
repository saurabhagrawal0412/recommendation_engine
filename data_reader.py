# Contains the methods to read ratings data from a file

import numpy as np
import pandas as pd

np.set_printoptions(precision=3)


def read_ratings_from_file(file_path, separator, line_terminator, columns):
    """Reads the ratings data from the file residing at file_path, and returns the in-memory pandas dataframe
    :param file_path: String file path
    :param separator: , for CSV file and \t for a TSV file
    :param line_terminator: Character used to terminate a line in the file
    :param columns: List of column strings
    :return: Pivot table (dataframe) with user-id as rows and movie-id as columns
    """
    separator = '\t'
    line_terminator = '\n'
    df = pd.read_csv(file_path, sep=separator, lineterminator=line_terminator)
    df.columns = columns
    ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', values='value', aggfunc=np.max)
    return ratings_df


def get_ratings_df(config):
    """Reads the ratings data from the train and test files, and returns the in-memory pandas dataframes
    :param config: ConfigParser object for the config file
    :return: Two pivot tables (dataframes) one each for train and test; with user-id as rows and movie-id as columns
    """
    training_file = config.get('RATINGS_FILES', 'TrainFile')
    test_file = config.get('RATINGS_FILES', 'TestFile')
    separator = config.get('RATINGS_FILES', 'Separator')
    line_terminator = config.get('RATINGS_FILES', 'LineTerminator')
    columns = config.get('RATINGS_FILES', 'Columns').split(',')
    train_ratings_df = read_ratings_from_file(training_file, separator, line_terminator, columns)
    test_ratings_df = read_ratings_from_file(test_file, separator, line_terminator, columns)
    # display_dataframe(train_ratings_df, 'train_ratings.html')
    # display_dataframe(test_ratings_df, 'test_ratings.html')
    return train_ratings_df, test_ratings_df


def display_dataframe(df, file_name):
    """Displays a pandas dataframe on a webbrowser
    :param df: The pandas dataframe to be displayed
    :param file_name: Name of the html file to be generated
    """
    import os
    import webbrowser
    html = df.to_html(na_rep="")
    with open(file_name, "w") as f:
        f.write(html)
    full_filename = os.path.abspath(file_name)
    webbrowser.open("file://{}".format(full_filename))
