# markovSentenceGenerator

A markov chain approach to generating sentences, admittedly a complete failure.

Summary:
1. Use spacy's nlp model to extract word properties (verb, noun, etc.)
2. Use gensim.downloader to get Google's word2vec embedding model
3. Cluster the embedded words, first filtering on word property, to group words by "similarity"
4. Calculate the Markov Probability Matrix
    - Find the probability that word i is followed by word i+1
    - Each word is generalized to its property and cluster
5. Use the probability matrix to simulate new sentences


TODO:
- compartmentalize tasks to functions
- save different parts of the code to make future runs mroe time efficient
