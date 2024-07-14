#!/usr/bin/env python3
"""Inserts a new document in a collection based on kwargs"""


def insert_school(mongo_collection, **kwargs):
    """function that inserts a new document in a collection"""
    new_document = mongo_collection.insert_one(kwargs).inserted_id
    return new_document
