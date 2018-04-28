#!/usr/bin/env python
# Program that implements the graph algorithm for the recommendation system
# Usage:
#   python graph_test.py [file path of the credentials file]
#

import argparse
from configparser import ConfigParser
import math
import numpy as np
import pandas as pd
import pickle
from py2neo import authenticate, Graph
import sys


TRAIN_FILE = 'data/ml-100k/ua.base'
TEST_FILE = 'data/ml-100k/ua.test'
IS_GRAPH_CONSTRUCTED = True


class Edge:
    """Object that stores various attributes of an edge
    """
    def __init__(self, properties):
        """Constructor for Edge
        :param properties: py2neo.core.PropertySet object that holds information about the edge
        """
        self.s_val = properties['s_val']
        self.t_val = properties['t_val']
        self.pred_val = properties['pred_val']

    def __str__(self):
        """Returns the string representation of the Edge object
        """
        return 'S:%d  T:%d  PredVal:%.2f' % (self.s_val, self.t_val, self.pred_val)


class Path:
    """Object that stores a path
    """
    def __init__(self, record):
        """Constructor for Path
        :param record: py2neo.cypher.core.Record object that holds information about all the edges
        """
        self.total_dist = 0.0
        self.edge_list = list()
        for relationship in record[0]:
            curr_edge = Edge(relationship.properties)
            self.edge_list.append(curr_edge)
            self.total_dist += curr_edge.pred_val

    def __str__(self):
        """Returns the string representation of the Path object
        """
        return '\n'.join(map(str, self.edge_list))

    def __cmp__(self, other):
        """Method to compare the current and the other object
        :param other: The other Path object
        :return: The smaller object
        """
        return cmp(self.total_dist, other.total_dist)


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


def get_ratings_df(file_path):
    """Reads the ratings file at the file path and returns the in-memory dataframe
    :param file_path: String file path
    :return: Pandas dataframe with rows as different ratings
    """
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    rating_df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return rating_df


def perform_authentication(cred_config):
    """Performs authentication for the neo4j database
    :param cred_config: ConfigParser object for credentials config file
    :return:
    """
    host_port = cred_config.get('NEO4J', 'HostPort')
    user_name = cred_config.get('NEO4J', 'UserName')
    password = cred_config.get('NEO4J', 'Password')
    authenticate(host_port, user_name, password)


def create_graph(cred_config, train_df):
    """Creates the neo4j graph by inserting all the ratings as edges
    :param cred_config: ConfigParser object for credentials config file
    :param train_df: Pandas dataframe with rows as different ratings
    :return: py2neo Graph object
    """
    database_uri = cred_config.get('NEO4J', 'DatabaseURI')
    graph = Graph(database_uri)

    if not IS_GRAPH_CONSTRUCTED:
        tx = graph.cypher.begin()
        statement = ("MERGE (u:User {user_id:{A}}) "
                     "MERGE (i:Item {item_id:{C}}) "
                     "MERGE (u)-[r:Rated {rating:{B},timestamp:{D}}]->(i);")
        for r, row in train_df.iterrows():
            tx.append(statement, {'A': int(row.loc['user_id']), 'B': float(row.loc['rating']),
                                  'C': int(row.loc['item_id']), 'D': int(row.loc['timestamp'])})
            if r % 100 == 0:
                tx.process()
        tx.commit()

        graph.cypher.execute('CREATE INDEX ON :User(user_id)')
        graph.cypher.execute('CREATE INDEX ON :Item(item_id)')

        unique_users = train_df['user_id'].unique()
        create_pred_edges(cred_config, unique_users, graph)

    return graph


def is_horts(common_count, rated_item_count, min_horting):
    """Determines whether a user1 horts user2 or not
    :param common_count: The number of items commonly rated by user1 and user2
    :param rated_item_count: The total number of items rated by user1
    :param min_horting: The minimum horting threshold
    :return: Returns true if the horting value >= min_horting
    """
    return True if (common_count + 0.0) / rated_item_count >= min_horting else False


def compute_predictability(rating_sum1, rating_sum2, common_count, max_rating):
    """Computes the predictability of user2 predicting user1
    :param rating_sum1: The sum of the ratings of all items rated by user1
    :param rating_sum2: The sum of the ratings of all items rated by user2
    :param common_count: The number of items commonly rated by user1 and user2
    :param max_rating: The max rating that could be given to any item
    :return: Float predictability of user2 predicting user1
    """
    s_list = [-1, 1]
    t_list = [i+1 for i in range(2 * max_rating - 1)]
    s_val, t_val, pred_val = None, None, sys.float_info.max

    for s_curr in s_list:
        for t_curr in t_list:
            curr_val = abs(0.0 + rating_sum1 - (rating_sum2 * s_curr) - (common_count * t_curr)) / common_count
            if curr_val < pred_val:
                pred_val = curr_val
                s_val = s_curr
                t_val = t_curr
    return pred_val, s_val, t_val


def count_rated_items(graph, user_id):
    """Finds and returns the number of items rated by a user identified by the user_id
    :param graph: py2neo Graph object
    :param user_id: User ID of the user for whom count of rated items need to be determined
    :return: Count of rated items for the given user
    """
    statement = 'MATCH (u:User {user_id:{user_id}})-[r:Rated]->(i:Item) RETURN count(i.item_id) AS COUNT;'
    tx = graph.cypher.begin()
    tx.append(statement, {'user_id': int(user_id)})
    result = tx.commit()
    return int(result[0][0]['COUNT'])


def create_pred_edge(graph, user_id1, user_id2, pred_val, s_val, t_val):
    """Creates the predictability edge between 2 users
    :param graph: py2neo Graph object
    :param user_id1: User ID of the first user
    :param user_id2: User ID of the second user
    :param pred_val: Float predictability value
    :param s_val: s value for the prediction relation
    :param t_val: t value for the prediction relation
    :return:
    """
    print 'Inside create_pred_edge'
    statement = ('MATCH (u1:User {user_id:{user_id1}}) '
                 'MATCH (u2:User {user_id:{user_id2}}) '
                 'MERGE (u1)-[p1:Predictability {pred_val:{pred_val}, s_val:{s_val}, t_val:{t_val}}]->(u2);')
    tx = graph.cypher.begin()
    tx.append(statement, {'user_id1': int(user_id1), 'user_id2': int(user_id2), 'pred_val': float(pred_val),
                          's_val': int(s_val), 't_val': int(t_val)})
    tx.process()
    tx.commit()


def check_and_create_pred_edge(graph, common_count, user_id1, count1, rating_sum1, user_id2, rating_sum2,
                               max_rating, min_horting, max_predictability):
    """Creates a predictability edge user1 -> user2 if user1 horts user2 and user2 predicts user1
    :param graph: py2neo Graph object
    :param common_count: The number of items commonly rated by user1 and user2
    :param user_id1: User ID of user1
    :param count1: The total number of items rated by user1
    :param rating_sum1: The sum of the ratings of all items rated by user1
    :param user_id2: User ID of user2
    :param rating_sum2: The sum of the ratings of all items rated by user2
    :param max_rating: The max rating that could be given to any item
    :param min_horting: The minimum horting threshold
    :param max_predictability: Threshold used to determine whether to create a predictability edge or not
    :return:
    """
    # Checking if user1 horts user2
    if is_horts(common_count, count1, min_horting):
        print 'User1:', user_id1, 'horts User2:', user_id2
        # Computing the predictability of user2 predicting user1
        pred_val, s_val, t_val = compute_predictability(rating_sum1, rating_sum2, common_count, max_rating)
        print 'User1: PredVal:', pred_val, 'SVal:', s_val, 'TVal:', t_val
        if pred_val < max_predictability:
            print 'User2:', user_id2, 'predicts User1:', user_id1
            create_pred_edge(graph, user_id1, user_id2, pred_val, s_val, t_val)


def create_pred_edges(cred_config, unique_users, graph):
    """Creates the predictability edges between eligible nodes
    :param cred_config: ConfigParser object for credentials config file
    :param unique_users: List of unique user ids for which we need to add predictability edges
    :param graph: py2neo Graph object
    :return:
    """
    min_horting = float(cred_config.get('GRAPH', 'MinHorting'))
    max_predictability = float(cred_config.get('GRAPH', 'MaxPredictability'))
    max_rating = int(cred_config.get('GRAPH', 'MaxRating'))

    statement = 'MATCH (u1:User {user_id:{user_id}})-[r1:Rated]->(i:Item)<-[r2:Rated]-(u2:User) ' \
                'WHERE u2.user_id > u1.user_id ' \
                'RETURN u2.user_id AS USER_ID2, sum(r1.rating) AS RATING_SUM1, ' \
                'sum(r2.rating) AS RATING_SUM2, count(i.item_id) AS COMMON_COUNT;'

    for user_id1 in unique_users:
        count1 = count_rated_items(graph, user_id1)
        tx = graph.cypher.begin()
        tx.append(statement, {'user_id': int(user_id1)})
        result = tx.commit()
        for row in result[0]:
            user_id2, common_count = int(row['USER_ID2']), int(row['COMMON_COUNT'])
            rating_sum1, rating_sum2 = float(row['RATING_SUM1']), float(row['RATING_SUM2'])
            count2 = count_rated_items(graph, user_id2)
            check_and_create_pred_edge(graph, common_count, user_id1, count1, rating_sum1, user_id2, rating_sum2,
                                       max_rating, min_horting, max_predictability)
            check_and_create_pred_edge(graph, common_count, user_id2, count2, rating_sum2, user_id1, rating_sum1,
                                       max_rating, min_horting, max_predictability)


def find_shortest_path(graph, user_id1, user_id2, max_path_len):
    """Finds the shortest predictability path between 2 users
    :param graph: py2neo Graph object
    :param user_id1: User ID of the first user
    :param user_id2: User ID of the second user
    :param max_path_len: Max allowed path length between the 2 users
    :return:
    """
    statement = 'MATCH (u1:User {user_id:{user_id1}}) ' \
                'MATCH (u2:User {user_id:{user_id2}}) ' \
                'MATCH (u1)-[p:Predictability*..%d]->(u2) RETURN p;' % max_path_len
    tx = graph.cypher.begin()
    tx.append(statement, {'user_id1': int(user_id1), 'user_id2': int(user_id2), 'max_path_len': int(max_path_len)})
    result = tx.commit()
    path_list = [Path(record) for record in result[0]]
    return min(path_list) if len(path_list) > 0 else None


def get_users_from_item(graph, item_id):
    """Finds all the users who rated the item and returns a dict that maps from their user_id to rating
    :param graph: py2neo Graph object
    :param item_id: ID of the item for which we need to determine the users
    :return: Dict that maps user_id to their rating
    """
    statement = 'MATCH (u:User)-[r:Rated]->(i:Item {item_id:{item_id}}) ' \
                'RETURN u.user_id AS USER_ID, r.rating AS RATING;'
    tx = graph.cypher.begin()
    tx.append(statement, {'item_id': int(item_id)})
    result = tx.commit()
    user_dict = {row['USER_ID']: row['RATING'] for row in result[0]}
    return user_dict


def predict_first_rating(end_rating, shortest_path):
    """Predicts the rating given by the first user in the shortest path
    :param end_rating: The rating given to the item by the last user
    :param shortest_path: Path shortest path
    :return: Float first rating
    """
    curr_rating = end_rating
    rev_edges = list(reversed(shortest_path.edge_list))
    for edge in rev_edges:
        curr_rating = curr_rating * edge.s_val + edge.t_val
    return curr_rating


def predict_rating(graph, user_id, item_id, path_length):
    """Determines predicted rating for a user for a particular item
    :param graph: py2neo Graph object
    :param user_id: User ID of the user
    :param item_id: Item ID of the item
    :param path_length: Maximum path length of the path
    :return: Float predicted rating
    """
    user_dict = get_users_from_item(graph, item_id)
    total_rating, divisor = 0.0, 0.0
    for other_user, other_rating in user_dict.iteritems():
        shortest_path = find_shortest_path(graph, user_id, other_user, path_length)
        if shortest_path is not None:
            total_rating += predict_first_rating(other_rating, shortest_path)
            divisor += 1.0
    return (total_rating + 0.0) / divisor if divisor > 0 else None


def predict_test_ratings(cred_config, graph, test_df):
    """Predicts the ratings for the test datapoints
    :param cred_config: ConfigParser object for the credentials config file
    :param graph: py2neo Graph object
    :param test_df: Dataframe that contains test ratings in a matrix with rows as users and columns as items
    :return: Dataframe that stores the predicted ratings
    """
    path_length = int(cred_config.get('GRAPH', 'MaxPathLength'))
    prediction_df = test_df.copy(deep=True)
    counter = 0
    for row_idx, row in prediction_df.iterrows():
        user_id, item_id, actual_rating = int(row['user_id']), int(row['item_id']), int(row['rating'])
        predicted_rating = predict_rating(graph, user_id, item_id, path_length)
        if predicted_rating is not None:
            print 'ActualRating:', actual_rating, 'PredictedRating:', predicted_rating
            prediction_df.ix[row_idx]['rating'] = predicted_rating
        else:
            prediction_df.ix[row_idx]['rating'] = np.nan
        counter += 1
        print 'Counter:', counter
    return prediction_df


def main():
    """Main method
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cred_config_file', help='Location of the credentials file')
    cred_config = ConfigParser()
    cred_config.read(parser.parse_args().cred_config_file)
    train_df = get_ratings_df(TRAIN_FILE)

    perform_authentication(cred_config)
    graph = create_graph(cred_config, train_df)
    test_df = get_ratings_df(TEST_FILE)
    pred_df = predict_test_ratings(cred_config, graph, test_df)
    file_handler = open('pred_df.data', 'w')
    pickle.dump(pred_df, file_handler)
    test_pivot = pd.pivot_table(test_df, index='user_id', columns='movie_id', values='rating', aggfunc=np.max)
    pred_pivot = pd.pivot_table(pred_df, index='user_id', columns='movie_id', values='rating', aggfunc=np.max)
    mae = compute_mean_absolute_error(test_pivot, pred_pivot)
    print 'MeanAbsoluteError:', mae
    rmse = compute_root_mean_squared_error(test_pivot, pred_pivot)
    print 'RootMeanAbsoluteError', rmse


if __name__ == '__main__':
    sys.stdout = open('data/graph_log.txt', 'w')
    main()
