#!/usr/bin/env python3
"""lists all documents in a collection"""


def list_all(mongo_collection):
    """function that lists all documents in a collection"""
    all_documents = mongo_collection.find()
    list_documents= list(all_documents)
    return list_documents
