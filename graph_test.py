#!/usr/bin/env python
# Program that implements the graph algorithm for the recommendation system
# Usage:
#   python graph_test.py [file path of the credentials file]
#

import argparse
from configparser import ConfigParser
import numpy as np
import matplotlib as plt
import pandas as pd
from py2neo import authenticate, Graph
import sys


RATINGS_FILE = 'data/ml-100k/ua.base'


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


def create_graph(cred_config, ratings_df):
    """Creates the neo4j graph by inserting all the ratings as edges
    :param cred_config: ConfigParser object for credentials config file
    :param ratings_df: Pandas dataframe with rows as different ratings
    :return: py2neo Graph object
    """
    database_uri = cred_config.get('NEO4J', 'DatabaseURI')
    graph = Graph(database_uri)

    tx = graph.cypher.begin()
    statement = ("MERGE (u:User {user_id:{A}}) "
                 "MERGE (i:Item {item_id:{C}}) "
                 "MERGE (u)-[r:Rated {rating:{B},timestamp:{D}}]->(i);")

    for r, row in ratings_df.iterrows():
        tx.append(statement, {'A': int(row.loc['user_id']), 'B': int(row.loc['rating']),
                              'C': int(row.loc['item_id']), 'D': int(row.loc['timestamp'])})
        if r % 100 == 0:
            tx.process()
    tx.commit()

    graph.cypher.execute('CREATE INDEX ON :User(user_id)')
    graph.cypher.execute('CREATE INDEX ON :Item(item_id)')


def main():
    """Main method
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cred_config_file', help='Location of the credentials file')
    cred_config = ConfigParser()
    cred_config.read(parser.parse_args().cred_config_file)
    ratings_df = get_ratings_df(RATINGS_FILE)
    perform_authentication(cred_config)
    graph = create_graph(cred_config, ratings_df)


if __name__ == '__main__':
    sys.stdout = open('data/graph_log.txt', 'w')
    main()
