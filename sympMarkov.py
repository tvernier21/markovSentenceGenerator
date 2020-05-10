import numpy as np
import pandas as pd
import gensim
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import random

def Simulate(num_sims, Q, t, bgak_dist):
    print("Writting Sentences")
    f = open("fakesentences.txt", "w")
    print(t)
    print(t.shape)
    print(list(range(t.shape[0])))
    for x in range(num_sims):
        print(x)
        i = random.choices(range(t.shape[0]),
                            weights=t,
                            k=1)[0]
        word = random.choices(list(bgak_dist[i].keys()),
                                weights=list(bgak_dist[i].values()),
                                k=1)[0]
        sentence = word
        while word is not ".":
            print(Q[i,:])
            print(Q[i,:].shape)
            i = random.choices(population=range(t.shape[0]),
                                weights=Q[i,:],
                                k=1)[0]
            word = random.choices(population=list(bgak_dist[i].keys()),
                                    weights=list(bgak_dist[i].values()),
                                    k=1)[0]
            sentence += " " + word
        sentence += "\n"
        f.write(sentence)
        print(sentence)
    f.close()

def generateProbabilities(k):
    print("loading model...")
    stop_words = set(stopwords.words('english'))
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

    print("loading words from thesympathizer.txt")
    print("...")
    f = open("thesympathizer.txt", "r")
    words = []
    for line in f.readlines():
        for word in line.strip('.\n').split():
            words.append(word.strip().lower())
        words.append(".")
    f.close()


    print("Creating word2vec values...")
    words_vec = []
    for word in words:
        if word in model.vocab:
            vec = model[word]
            words_vec.append(np.array(vec))

    # Creating K clusters from the word2vec vectors
    print("Calculating Kmeans...")
    kmeans = KMeans(n_clusters=k, n_init=10).fit(np.array(words_vec))
    centroids = kmeans.cluster_centers_

    print("and Done kmeans")

    Q = np.zeros((k+2, k+2), dtype=np.float64)
    bgak = [{} for i in range(k+2)]
    bgak_total = np.zeros(k+2)
    t_distribution = np.zeros(k+2)

    print("\n Now updating probabilites for each bag of words")
    for n in range(len(words)-1):
        word_i = words[n]
        word_j = words[n+1]

        # Grouping each word intos its correct bag.
        word_i_exists = True
        if word_i in stop_words:
            i = k
        elif word_i == ".":
            i = k+1
        elif word_i in model.vocab:
            vec = np.array(model[word_i])
            idx = kmeans.predict(vec.reshape(1,-1))
            i = idx[0]
        else:
            word_i_exists = False

        word_j_exists = True
        if word_j in stop_words:
            j = k
        elif word_j in model.vocab:
            vec = np.array(model[word_j])
            idx = kmeans.predict(vec.reshape(1,-1))
            j = idx[0]
        elif word_j == ".":
            j = k+1
        else:
            word_j_exists = False

        #Cases to catch all beginning of sentences.
        if n == 0 and word_i_exists:
            t_distribution[i] += 1
        if word_i == "." and word_j_exists:
            t_distribution[j] += 1

        # now update the matrix
        if (word_i_exists and word_j_exists):
            if word_i in bgak[i]:
                bgak[i][word_i] += 1 # incrementing cases of specific word
            else:
                bgak[i][word_i] = 1
            bgak_total[i] += 1       # incrementing total words of this category

            Q[i,j] += 1



    print("Distribution of words in each group")
    for i in range(k+2):
        print("--------")
        print("Group",i, "-", bgak_total[i])
        if bgak_total[i] > 0:
            # Creating Conditional Probability
            Q[i,:] /= bgak_total[i]
            # Create Probability Distribution within each bag of selecting a word.
            for word, occurences in bgak[i].items():
                bgak[i][word] = occurences/bgak_total[i]
            # print(bgak[i].items(
        else:
            Q[i,:] = 1/(k+2)

    #Fix t_distributions
    t_distribution /= (bgak_total[k+1] + 1)
    print(t_distribution)
    print(Q)

    t_df = pd.DataFrame(t_distribution)
    t_df.to_csv(f't_distribution_{k}.csv')
    Q_df = pd.DataFrame(Q)
    Q_df.to_csv(f'transition_{k}.csv')

    # Q = np.nan_to_num(Q, nan=1/(k+2))

    return t_distribution, Q, bgak

def parseBook():
    f = open("thesympathizer_raw.txt", "r")
    sentences = []
    sentence = ""
    sup = 0
    for line in f.readlines():
        line = line.replace(',', ' ').replace('\xe2\x80\x99',' ').replace('\xe2\x80\x9c',' ').replace('\xe2\x80\x9d',' ').replace('(',' ').replace(')',' ').replace('\xe2\x80\x94',' ').replace('!','.').replace('?','.').strip('\n')
        parsed = line.split('.')

        if sentence is not "":
            sentence += " " + parsed.pop(0)
            if "." in line:
                sentences.append(sentence)
                sentence = ""
        for i in range(len(parsed)):
            # if the last split
            if (i+1 == len(parsed)) and "." not in line[-4:]:
                sentence = parsed[i]
            else:
                sentences.append(parsed[i])

    f.close()
    # print(len(sentences))
    # print(sentences[:20])
    # print(sentences[-20:])

    f_new = open("thesympathizer.txt", "w")
    for sentence in sentences:
        if len(sentence) > 4:
            string = sentence + ".\n"
            f_new.write(string)
    f_new.close()

if __name__ == "__main__":
    # parseBook()
    k = 100
    t, Q, bgak_dist = generateProbabilities(k)
    n = 1000
    Simulate(n, Q, t, bgak_dist)
