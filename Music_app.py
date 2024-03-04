import pickle
import streamlit as st

st.header("Music Recommendation System")

ab = pickle.load(open("ab.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
song_name = pickle.load(open("song_name.pkl", "rb"))

def recommend_song(song):
    idx = ab[ab['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for i in distances[1:6]:
        songs.append(ab.iloc[i[0]].song)

    return songs

selected_Music = st.selectbox(
    "Type or Select a Music",
    song_name
)

if st.button('Show Recommendations'):
    recommendation_song = recommend_song(selected_Music)
    col1,col2,col3,col4,col5 = st.columns(5)


    with col1:
        st.text(recommendation_song[0] if len(recommendation_song) > 0 else "")

    with col2:
        st.text(recommendation_song[1] if len(recommendation_song) > 1 else "")

    with col3:
        st.text(recommendation_song[2] if len(recommendation_song) > 2 else "")

    with col4:
        st.text(recommendation_song[3] if len(recommendation_song) > 3 else "")

    with col5:
        st.text(recommendation_song[4] if len(recommendation_song) > 4 else "")





