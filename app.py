import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------- NETFLIX STYLE CSS ---------- #

st.markdown("""
<style>

body{
background-color:#141414;
color:white;
}

.netflix-header{
font-size:50px;
font-weight:bold;
color:#E50914;
text-align:center;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:40px;
}

.movie-card{
background-color:#1f1f1f;
padding:20px;
border-radius:10px;
transition:0.3s;
}

.movie-card:hover{
transform:scale(1.05);
background-color:#2b2b2b;
}

.movie-title{
font-size:18px;
font-weight:bold;
}

.rating{
color:#FFD700;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="netflix-header">NETFLIX MOVIE RECOMMENDER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies similar to your favorites</div>', unsafe_allow_html=True)

# ---------- LOAD DATA ---------- #

@st.cache_data
def load_data():
    
    df = pd.read_csv("movies.csv")
    
    df['Film Name'] = df['Film Name'].str.replace(r'^\d+\.?\s*', '', regex=True)
    
    df = df.dropna(subset=['Summary']).reset_index(drop=True)
    
    df['Summary'] = df['Summary'].str.lower().str.strip()
    
    df['Ratings'] = df['Ratings'].str.replace("N/a","0")
    
    return df

df = load_data()

# ---------- MODEL ---------- #

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

# ---------- RECOMMENDATION FUNCTION ---------- #

def recommend(movie):

    movie = movie.lower()

    if movie not in df['Film Name'].str.lower().values:
        return None

    index = df[df['Film Name'].str.lower()==movie].index[0]

    distance, indices = knn.kneighbors(tf_matrix[index])

    results = []

    for i in range(1,len(indices[0])):

        movie_index = indices[0][i]

        name = df.iloc[movie_index]['Film Name']
        rating = df.iloc[movie_index]['Ratings']
        year = df.iloc[movie_index]['Year']

        results.append((name,rating,year))

    return results

# ---------- UI ---------- #

movie_list = sorted(df['Film Name'].unique())

selected_movie = st.selectbox(
"🎥 Choose a movie",
movie_list
)

if st.button("Recommend Movies 🍿"):

    recommendations = recommend(selected_movie)

    if recommendations is None:
        st.error("Movie not found")

    else:

        st.subheader("Recommended Movies")

        cols = st.columns(5)

        for i,(name,rating,year) in enumerate(recommendations):

            with cols[i]:

                link = f"https://www.google.com/search?q={name}+movie"

                st.markdown(f"""
                <div class="movie-card">

                <div class="movie-title">{name}</div>

                ⭐ <span class="rating">{rating}</span>

                <br>

                📅 {year}

                <br><br>

                <a href="{link}" target="_blank">View Movie</a>

                </div>
                """, unsafe_allow_html=True)
