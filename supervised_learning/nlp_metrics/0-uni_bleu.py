#!/usr/bin/env python3
"""Calculating the unigram BLEU score for a sentence"""

import numpy as np


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score"""
    len_sent = len(sentence)
    len_ref = []
    words = {}

    for translation in references:
        len_ref.append(len(translation))
        for word in translation:
            if word in sentence and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    index = np.argmin([abs(len(i) - len_sent) for i in references])
    close_trans = len(references[index])

    if len_sent > close_trans:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(close_trans) / float(len_sent))
    BLEU_score = BLEU * np.exp(np.log(total / len_sent))

    return BLEU_score
