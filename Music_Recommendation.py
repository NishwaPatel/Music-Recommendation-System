import numpy as np
import pandas as pd
ab = pd.read_csv("../song_5000.csv")
print(ab)
print(ab.shape)
print(ab.head(5))
print(ab['song'])
print(ab['text'][0])
ab['text'] = ab['text'].str.lower().replace(r'[^\w\s]','').replace(r'\n',' ',regex=True)
print(ab['text'][0])
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [ps.stem(w) for w in tokens]

    return " ".join(stemming)

ab['text'] = ab['text'].apply(lambda x: tokenization(x))

print(ab['text'][0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfid = TfidfVectorizer(stop_words='english')
matrix = tfid.fit_transform(ab['text'])

print(matrix.shape)

similarity = cosine_similarity(matrix)

print(similarity[0])

print(ab['song'][3])

song_name = ab['song']


def recommendation(song):
    idx = ab[ab['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True,
                       key=lambda x: x[1])  ### 1st number song have similarity with its own and others...

    songs = []
    for i in distances[1:21]:
        songs.append(ab.iloc[i[0]].song)

    return songs

re = recommendation('Without Love')

print("--------------------")

print(re)


import pickle

with open('ab.pkl', 'wb') as f:
    pickle.dump(ab, f)
with open('ab.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('similarity.pkl','wb') as f:
    pickle.dump(similarity, f)
with open('similarity.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('song_name.pkl','wb') as f:
    pickle.dump(song_name, f)
with open('song_name.pkl', 'rb') as f:
    loaded_model = pickle.load(f)