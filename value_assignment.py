import gensim.downloader as gensim_api
import spacy


# Spacy NLP model for Dependency Parsing
def loadDependencyModel():
    return spacy.load('en_core_web_sm')


def getWordProp(nlp, sentence):
    doc = nlp(sentence)
    return [[token.pos_, token.text] for token in doc]


def loadWord2VecModel():
    return gensim_api.load('word2vec-google-news-300')


def getWordEmbedding(model, word):
    return model[word]
