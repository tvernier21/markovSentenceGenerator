import preprocessing
import value_assignment as va
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import random


def prepareWordValues(nlp, w2v, sentences):
    """
    nlp - language model that assigns grammatical properties to words.
    w2v  - word2vec embedding model.
    sentences - list of strings
    db - pandas database or sql db key
    """
    db = {}
    unique_words = set()

    print("Getting word properties")
    for sentence in tqdm(sentences):
        properties, words = zip(*va.getWordProp(nlp, sentence))
        for prop, word in zip(list(properties), list(words)):

            # add grammatical property to the dictionary
            if prop not in db:
                db[prop] = {"embeddings": np.array([], dtype=np.float32),
                            "min": None,
                            "max": None, 
                            "words": None,
                            "unique_words": set(),
                            "kmeans": None,
                            "k": 0,
                            "idx": -1}

            # Add word vector embedding to grammatical group
            db[prop]["embeddings"] = np.append(db[prop]["embeddings"],
                                              va.getWordEmbedding(w2v, word))
            db[prop]["unique_words"].add(word)
            unique_words.add(word)

    print("Calculating the number of clusters per property")
    total_k = 150
    p_k = 0
    for prop, data in db.items():
        embeddings = data['embeddings'].reshape(-1, 300)
        embeddings = (embeddings - np.min(embeddings)) / (np.max(embeddings) - np.min(embeddings))
        db[prop]['embeddings'] = embeddings + .001
        db[prop]['min'] = np.min(embeddings)
        db[prop]['max'] = np.max(embeddings)
        k = 1
        if data['embeddings'].shape[0] > 1000:
            k = max(4, int((len(data['unique_words']) / len(unique_words)) * total_k))

        print("Prop", prop)
        print("prop unique words", len(data['unique_words']))
        print("prop words", len(unique_words))
        print("k =", k)

        db[prop]['idx'] = p_k
        db[prop]['k'] = k
        db[prop]['words'] = [[] for i in range(k)]
        p_k += k


        db[prop]['kmeans'] = KMeans(n_clusters=db[prop]['k'], n_init=10).fit(embeddings)


    print("Create Probability Matrix")
    prob_M = np.zeros((p_k+1, p_k+1), dtype=np.float32)
    for sentence in tqdm(sentences):
        properties, words = zip(*va.getWordProp(nlp, sentence))

        # The index is set to the "." index, or the end of a sentence.
        # This means that when we are looking to generate sentences, we can
        # use the last row of the matrix to figure out the most likely 
        # starting column (property X cluster)
        prev_idx = p_k
        for prop, word in zip(list(properties), list(words)):
            prop_idx = db[prop]['idx']

            embedding = np.array(va.getWordEmbedding(w2v, word), dtype=np.float32).reshape(-1,300)
            embedding = (embeddings - db[prop]['min']) / (db[prop]['max'] - db[prop]['min'])

            prop_cluster_idx = db[prop]['kmeans'].predict(embedding)[0]
            curr_idx = prop_idx + prop_cluster_idx
            
            db[prop]['words'][prop_cluster_idx].append(word)
            prob_M[prev_idx, curr_idx] += 1

            prev_idx = curr_idx

        curr_idx = p_k
        prob_M[prev_idx, curr_idx] += 1 

    print("Fixing matrix")
    prob_M = prob_M / prob_M.sum(axis=1, keepdims=True)

    return db, prob_M


def find_prop_and_cluster_idx(db, prev_idx):
    for prop, data in db.items():
        start = data['idx']
        end = start + data['k']
        if prev_idx >= start and prev_idx < end:
            return prop, prev_idx - start
    return None, None


def generate_sentences(db, prob_M, n_sims, filepath):
    f = open(filepath, "w")
    for i in tqdm(range(n_sims)):
        prob_vec = prob_M[-1, :]
        idx = random.choices(range(prob_vec.shape[0]),
                             weights=prob_vec,
                             k=1)[0]
        prop, cluster_idx = find_prop_and_cluster_idx(db, idx)
        word = random.choices(db[prop]['words'][cluster_idx],
                              k=1)[0]

        sentence = word
        while True: # Not equal to the last row/col (".")
            prob_vec = prob_M[idx, :]
            idx = random.choices(range(prob_vec.shape[0]),
                                 weights=prob_vec,
                                 k=1)[0]
            if idx == prob_M.shape[0] - 1:
                break
            prop, cluster_idx = find_prop_and_cluster_idx(db, idx)
            word = random.choices(db[prop]['words'][cluster_idx],
                                  k=1)[0]
            sentence += " " + word

        sentence += ".\n"
        f.write(sentence)
    f.close()
    return


def main():
    print("Preprocessing")
    book = "thesympathizer"
    sentences = preprocessing.cleanRawFile(f"{book}_raw.txt")

    # filter to only the first 100 sentences for speed purposes
    # sentences = sentences[:300]
    
    sentences = preprocessing.filterEmptySentences(sentences)
    sentences = preprocessing.uncasedSentences(sentences)

    print("Loading models")
    nlp = va.loadDependencyModel()
    w2v = va.loadWord2VecModel('word2vec-google-news-300')


    print("Assigning values to words")
    db, prob_M = prepareWordValues(nlp, w2v, sentences)

    print("Simulate new sentences")
    generate_sentences(db, prob_M, 100, "newsentences.txt")


main()
