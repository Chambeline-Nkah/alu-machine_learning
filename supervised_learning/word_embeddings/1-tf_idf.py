#!/usr/bin/env python3
"""Creating a TF-IDF embedding"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Creating a TF-IDF embedding

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used

    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings"""

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    counts = vectorizer.fit_transform(sentences)
    embed = counts.toarray()
    features = vectorizer.get_feature_names()
    return embed, features
