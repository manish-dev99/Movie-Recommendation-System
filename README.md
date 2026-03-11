# Movie-Recommendation-System
# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using **Machine Learning and Natural Language Processing (NLP)**.  
This application recommends movies based on the **similarity of movie summaries**.

The project also includes a **beautiful web interface built with Streamlit** where users can select a movie and get similar movie recommendations instantly.

---

## 🚀 Live Demo

🌐 **Live App:**  
https://movie-recommendation-system-uq7a7ywx2hku4gbn4p5nzg.streamlit.app/

---

## 📌 Project Overview

Recommendation systems are widely used in platforms like **Netflix, Amazon Prime, and Spotify** to suggest relevant content to users.

This project implements a **Content-Based Filtering approach**, where movies are recommended based on the similarity between their summaries.

The similarity is calculated using:

- **TF-IDF Vectorization**
- **Cosine Similarity**
- **K-Nearest Neighbors (KNN)**

---

## 🧠 How the System Works

1. **Data Preprocessing**
   - Cleaning movie titles
   - Handling missing values
   - Text normalization

2. **Text Processing**
   - Convert text to lowercase
   - Remove extra spaces
   - Remove stopwords

3. **Feature Extraction**
   - Convert movie summaries into numerical vectors using **TF-IDF**

4. **Similarity Search**
   - Use **KNN with cosine similarity** to find similar movies

5. **Recommendation**
   - Display the top similar movies to the selected movie

---

## 🛠 Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **TF-IDF Vectorizer**
- **K-Nearest Neighbors (KNN)**
- **Streamlit**

---

## 📊 Features

✔ Interactive Streamlit UI  
✔ Movie selection dropdown  
✔ Instant movie recommendations  
✔ Rating and year display  
✔ External movie search links  
✔ Deployed web application

---

## 📂 Project Structure
