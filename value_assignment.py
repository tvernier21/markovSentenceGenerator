import gensim.downloader as gensim_api
import spacy
import numpy as np


# Spacy NLP model for Dependency Parsing
def loadDependencyModel():
    return spacy.load('en_core_web_sm')


def getWordProp(nlp, sentence):
    doc = nlp(sentence)
    return [[token.pos_, token.text] for token in doc]


def loadWord2VecModel(model_name):
    return gensim_api.load(model_name)


def getWordEmbedding(model, word):
    try:
        return np.array(model[word], dtype=np.float32)
    except:
        return np.zeros(300, dtype=np.float32)

