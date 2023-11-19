import streamlit as st
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.bm25 import BM25
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# Step 1: Data Preparation
data_path = "dataset"
categories = os.listdir(data_path)

# Step 2: Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text, lowercasing=True, remove_stopwords=True, porter_stemming=True):
    tokens = nltk.word_tokenize(text)
    if lowercasing:
        tokens = [token.lower() for token in tokens]
    if remove_stopwords:
        tokens = [token for token in tokens if token.isalpha() and token.lower() not in stop_words]
    if porter_stemming:
        tokens = [ps.stem(token) for token in tokens]
    return ' '.join(tokens)

# Step 3: Term Weighting
def calculate_tfidf(docs, tfidf_type='Standard TF-IDF'):
    if tfidf_type == 'Standard TF-IDF':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
    elif tfidf_type == 'Smoothed TF-IDF':
        vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
        tfidf_matrix = vectorizer.fit_transform(docs)
    elif tfidf_type == "Probabilistic TF-IDF":
        tfidf_matrix = calculate_probabilistic_tfidf(docs)
    else:
        raise ValueError("Invalid TF-IDF type. Choose 'Standard TF-IDF', 'Smoothed TF-IDF', or 'Probabilistic TF-IDF'.")
    
    return tfidf_matrix


def calculate_probabilistic_tfidf(docs):
    N = len(docs)  # Total number of documents in the corpus
    
    # Create a vocabulary mapping terms to integer indices
    vocabulary = {term: idx for idx, term in enumerate(set(token for doc in docs for token in doc.split()))}
    
    # Term frequency matrix
    tf_matrix = np.zeros((N, len(vocabulary)))
    
    # Calculate term frequency for each document
    for i, doc in enumerate(docs):
        for term in doc.split():
            tf_matrix[i, vocabulary[term]] += 1
    
    # Document frequency vector
    df_vector = np.zeros(len(tf_matrix[0]))
    
    # Calculate document frequency for each term
    for i in range(len(df_vector)):
        df_vector[i] = np.count_nonzero(tf_matrix[:, i])
    
    # Inverse Document Frequency (IDF)
    idf_vector = np.log((N - df_vector + 0.5) / (df_vector + 0.5))
    
    # Term Frequency-Inverse Document Frequency (TF-IDF)
    #tfidf_matrix = tf_matrix * idf_vector[:, np.newaxis]  # Add np.newaxis to broadcast idf_vector
    tfidf_matrix = tf_matrix * idf_vector.reshape(1, -1)

    
    return tfidf_matrix



# Step 4: Vector Space Representation
def vectorize_documents(docs, tfidf_type='standard'):
    preprocessed_docs = [preprocess_text(doc) for doc in docs]
    tfidf_matrix = calculate_tfidf(preprocessed_docs, tfidf_type)
    return tfidf_matrix

# Step 5: Similarity Measure
def calculate_similarity(tfidf_matrix, similarity_measure='cosine'):
    if similarity_measure == 'cosine':
        similarity_matrix = cosine_similarity(tfidf_matrix)
    elif similarity_measure == 'euclidean':
        similarity_matrix = 1 / (1 + euclidean_distances(tfidf_matrix))
    else:
        raise ValueError("Invalid similarity measure. Choose 'cosine' or 'euclidean'.")
    
    return similarity_matrix

# Step 6: Ranking
def rank_documents(similarity_matrix):
    # Sum similarity scores for each document to get a total similarity score
    total_similarity_scores = similarity_matrix.sum(axis=1)

    # Enumerate through indices and scores and store in a list of tuples
    index_score_tuples = list(enumerate(total_similarity_scores.flatten()))

    # Sort the list of tuples based on similarity scores in descending order
    sorted_index_score_tuples = sorted(index_score_tuples, key=lambda x: x[1], reverse=True)

    # Unpack the sorted tuples into separate lists for indices and scores
    ranked_indices = [index for index, _ in sorted_index_score_tuples]

    return ranked_indices, total_similarity_scores.flatten()[ranked_indices]



def tf_idf():
    st.image("image/tfidf.png")
    
    # Step 1: Preprocessing Options
    st.subheader("Step 1: Preprocessing Options")
    lowercasing = st.checkbox("Lowercasing", value=True)
    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    porter_stemming = st.checkbox("Porter Stemming", value=True)

    # Step 2: TF-IDF Options
    st.subheader("Step 2: TF-IDF Options")
    tfidf_type = st.selectbox("Choose TF-IDF Algorithm", ['Standard TF-IDF', 'Smoothed TF-IDF', "Probabilistic TF-IDF"])
    if tfidf_type == "Standard TF-IDF":
        st.image("image/tf-idf-nomal.png")
    elif tfidf_type == "Smoothed TF-IDF":
        st.image("image/tf-idf-smooth.png")
    elif tfidf_type == "Probabilistic TF-IDF":
        st.image("image/tf-idf-pro.png")

    # Step 3: Similarity Measure Options
    st.subheader("Step 3: Similarity Measure Options")
    similarity_measure = st.selectbox("Choose Similarity Measure", ['cosine', 'euclidean'])
    if similarity_measure == "cosine":
        st.image("image/cosine.png")
    elif similarity_measure == "euclidean":
        st.image("image/euclidean.png")
    
    # Create a button to trigger the algorithm
    if st.button("Run Algorithm"):

        for category in categories:
            category_path = os.path.join(data_path, category)
            docs = [open(os.path.join(category_path, doc)).read() for doc in os.listdir(category_path)] 
            
            # Apply selected preprocessing options
            preprocessed_docs = [preprocess_text(doc, lowercasing, remove_stopwords, porter_stemming) for doc in docs]
            
            # Vector Space Representation
            tfidf_matrix = vectorize_documents(preprocessed_docs, tfidf_type)
            
            # Similarity Measure
            similarity_matrix = calculate_similarity(tfidf_matrix, similarity_measure)
            
            # Ranking
            ranked_indices, total_similarity_scores = rank_documents(similarity_matrix)
            
            # Display results in a table
            st.subheader(f"Category: {category}")
            results = []
            for idx, score in zip(ranked_indices, total_similarity_scores[ranked_indices]):
                results.append({
                    "Document": f"Document {idx + 1}",
                    "Similarity Score": f"{score:.4f}",
                    "Abstract": docs[idx][:50],  # Display the first 50 characters of each document
                })

            # Display the table for each category
            st.table(results)

if __name__ == "__main__":
    tf_idf()
