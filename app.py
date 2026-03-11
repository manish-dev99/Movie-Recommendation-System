import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- UI SETTINGS ---------------- #

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#FF4B4B;
}

.movie-card {
    background-color:#1E1E1E;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
}

.movie-title{
    font-size:20px;
    font-weight:bold;
}

.rating{
    color:#FFD700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)

st.write("Find movies similar to your favorite film.")

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():

    df = pd.read_csv("movies.csv")

    df['Number of Rators'] = (
        df['Number of Rators']
        .str.replace("\xa0(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("(", "", regex=False)
    )

    df['Number of Rators'] = df['Number of Rators'].fillna("0")

    df['Number of Rators'] = (
        df['Number of Rators']
        .str.replace("K", "000", regex=False)
        .astype(float)
        .astype(int)
    )

    df['Ratings'] = df['Ratings'].str.replace("N/a", "0")

    df['Film Name'] = df['Film Name'].str.replace(r'^\d+\.?\s*', '', regex=True)

    df = df.dropna(subset=['Summary']).reset_index(drop=True)

    df['Summary'] = df['Summary'].str.lower().str.strip()

    return df

df = load_data()

# ---------------- NLP MODEL ---------------- #

@st.cache_resource
def create_model():

    tfidf = TfidfVectorizer(stop_words="english")
    tf_matrix = tfidf.fit_transform(df['Summary'])

    knn = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
        algorithm="brute"
    )

    knn.fit(tf_matrix)

    return tf_matrix, knn

tf_matrix, knn = create_model()

# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie_name):

    movie_name = movie_name.lower()

    if movie_name not in df['Film Name'].str.lower().values:
        return None

    index = df[df['Film Name'].str.lower()==movie_name].index[0]

    distance, indices = knn.kneighbors(tf_matrix[index])

    movies = []

    for i in range(1,len(indices[0])):

        movie_index = indices[0][i]

        name = df.iloc[movie_index]['Film Name']
        rating = df.iloc[movie_index]['Ratings']
        year = df.iloc[movie_index]['Year']

        movies.append((name,rating,year))

    return movies


# ---------------- UI ---------------- #

movie_list = sorted(df['Film Name'].unique())

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)

if st.button("Recommend Movies 🍿"):

    results = recommend(selected_movie)

    if results is None:

        st.error("Movie not found")

    else:

        st.subheader("Recommended Movies")

        for name,rating,year in results:

            imdb_link = f"https://www.google.com/search?q={name}+movie"

            st.markdown(f"""
            <div class="movie-card">

            <div class="movie-title">{name}</div>

            ⭐ Rating: <span class="rating">{rating}</span>  
            📅 Year: {year}

            <br>

            🔗 <a href="{imdb_link}" target="_blank">View Movie</a>

            </div>
            """, unsafe_allow_html=True)