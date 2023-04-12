import preprocessing
import value_assignment as va
from tqdm import tqdm
import pandas as pd
import numpy as np


def prepareWordValues(nlp, w2v, sentences, db):
    """
    nlp - language model that assigns grammatical properties to words.
    wv  - word2vec embedding model.
    sentences - list of strings
    db - pandas database or sql db key
    """
    property2vec = {}

    print("Getting word properties")
    for sentence in tqdm(sentences):
        properties, words = zip(*va.getWordProp(nlp, sentence))
        for prop, word in zip(list(properties), list(words)):
            # add grammatical property to the dictionary
            if prop not in property2vec:
                property2vec[prop] = []

            # Add word vector embedding to grammatical group
            property2vec[prop].append(va.getWordEmbedding(w2v, word))

    print(len(unique_props))
    for unique_prop in unique_props:
        print(f"Current Property: {unique_prop}")
        # for curr_sentence, curr_grammar in zip(words, props):

            
    return None

        

def main():
    print("Preprocessing")
    book = "thesympathizer"
    sentences = preprocessing.cleanRawFile(f"{book}_raw.txt")

    # filter to only the first 100 sentences for speed purposes
    sentences = sentences[:300]
    
    sentences = preprocessing.filterEmptySentences(sentences)
    sentences = preprocessing.uncasedSentences(sentences)

    print("Loading models")
    nlp = va.loadDependencyModel()
    # wv = va.loadWord2VecModel()


    print("Assigning values to words")
    prepareWordValues(nlp, None, sentences, pd.DataFrame())
    





main()
