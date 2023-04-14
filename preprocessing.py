from gensim.models import Word2Vec
import spacy

replaceable_chars = {',': '',
                     '(': '',
                     ')': '',
                     "'s": '',
                     '!': '.',
                     '?': '.',
                     '/': ' ',
                     '\n': ' ',
                     '\xe2\x80\x99': '',
                     '\xe2\x80\x9c': '',
                     '\xe2\x80\x9d': '',
                     '\xe2\x80\x94': ''}

 
def replaceErroneousChars(string):
    new_string = string
    for old, new in replaceable_chars.items():
        new_string = new_string.replace(old, new)
    return new_string


def cleanRawFile(path):
    sentences = []
    with open(path, "r") as f:
        sentence = ""
        for line in f.readlines():
            sentence += replaceErroneousChars(line).strip() + " "

        sentences = sentence.split('.')
    return sentences


def filterEmptySentences(sentences):
    return list(filter(lambda x: len (x) >= 7, sentences))


def uncasedSentences(sentences):
    return [s.strip().lower() for s in sentences]


def splitSentences(sentences):
    return [sentence.strip().split() for sentence in sentences]


if __name__ == "__main__":
    book = "thesympathizer"
    sentences = uncasedSentences(filterEmptySentences(cleanRawFile(f"{book}_raw.txt")))
    nlp = loadDependencyModel()
    for i in range(5):
        doc = assignDependency(nlp, sentences[i])
