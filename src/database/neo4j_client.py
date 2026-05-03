import os

from neo4j import GraphDatabase


def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")
    return GraphDatabase.driver(uri, auth=(user, password))
