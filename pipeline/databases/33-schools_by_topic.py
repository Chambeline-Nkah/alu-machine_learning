#!/usr/bin/env python3
"""Returns the list of school having a specific topic"""
def schools_by_topic(mongo_collection, topic):
    """ returns the lsit of schools having a specific topic"""
    schools = mongo_collection.find({"topics": topic})
    return schools
