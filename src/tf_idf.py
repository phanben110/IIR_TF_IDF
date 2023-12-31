import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Create a Porter stemmer
ps = PorterStemmer()

# Load English stop words
stop_words = set(stopwords.words('english'))

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
        tfidf_matrix = calculate_probabilistic_tfidf(docs)  # You need to implement calculate_probabilistic_tfidf function
    else:
        raise ValueError("Invalid TF-IDF type. Choose 'Standard TF-IDF', 'Smoothed TF-IDF', or 'Probabilistic TF-IDF'.")
    
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

# Step 4: Probabilistic TF-IDF Calculation
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
    tfidf_matrix = tf_matrix * idf_vector.reshape(1, -1)
    
    return tfidf_matrix



def tf_idf():
    st.image("image/tfidf.png")

    # User Input Area
    st.subheader("Input text")
    input_text = st.text_area("Enter your text here:")

    # Step 1: Preprocessing Options
    st.subheader("Step 1: Preprocessing Options")
    lowercasing = st.checkbox("Lowercasing", value=True)
    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    porter_stemming = st.checkbox("Porter Stemming", value=True)

    # Step 2: TF-IDF Options
    st.subheader("Step 2: TF-IDF Options")

    toggle_show_example = st.toggle("Show example", value=False)
    if toggle_show_example:
        st.image("image/compareTF-IDF.png")

    # st.image("image/compareTF-IDF.png")
    tfidf_type = st.selectbox("Choose TF-IDF Algorithm", ['None','Standard TF-IDF', 'Smoothed TF-IDF', "Probabilistic TF-IDF"])
    tfidf_type_copy  = tfidf_type
    if tfidf_type == "Standard TF-IDF":
        st.image("image/tf-idf-nomal.png")
    elif tfidf_type == "Smoothed TF-IDF":
        st.image("image/tf-idf-smooth.png")
    elif tfidf_type == "Probabilistic TF-IDF": 
        st.image("image/tf-idf-pro.png")

    # Step 3: Similarity Measure Options
    st.subheader("Step 3: Similarity Measure Options")
    similarity_measure = st.selectbox("Choose Similarity Measure", ['None','cosine', 'euclidean'])
    if similarity_measure == "cosine":
        st.image("image/cosine.png")
    elif similarity_measure == "euclidean":
        st.image("image/euclidean.png")

    if similarity_measure is not "None" and tfidf_type is not "None":
        if st.button("Run Algorithm"):
            # Process user input
            # Path to the dataset
            dataset_path = "dataset"

            # Get the list of document paths
            document_paths = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_path) for file in files]

            # Read the contents of each document
            documents = [open(document_path, 'r', encoding='utf-8').read() for document_path in document_paths]

            # Preprocess documents
            preprocessed_documents = [preprocess_text(doc, lowercasing=True, remove_stopwords=True, porter_stemming=True) for doc in documents]

            # Add the input text to the preprocessed documents list
            preprocessed_documents.append(preprocess_text(input_text, lowercasing=True, remove_stopwords=True, porter_stemming=True))

            # Choose the TF-IDF type (you can change this to 'Smoothed TF-IDF' or 'Probabilistic TF-IDF')
            tfidf_matrix = calculate_tfidf(preprocessed_documents, tfidf_type)

            # Calculate similarity between documents
            similarity_matrix = calculate_similarity(tfidf_matrix, similarity_measure)

            # Create a dictionary to store the total similarity for each category
            category_similarity = defaultdict(float)


            # Get the indices of documents sorted by similarity
            sorted_indices = similarity_matrix[-1].argsort()[::-1]

            # Check for the minimum length between document_paths and similarity_matrix
            min_length = min(len(document_paths), len(similarity_matrix[-1]))

            document_info = []

            # Print the ranked documents and calculate total similarity for each category
            for index in sorted_indices[:min_length]:
                if index < len(document_paths):
                    document_path = document_paths[index]
                    category = os.path.basename(os.path.dirname(document_path))  # Extract category from the folder name
                    similarity = similarity_matrix[-1][index]
                    abstract = documents[index]  # Assuming documents contain the full text of each document

                    # Append document information to the list
                    document_info.append({
                        "Document Name": os.path.basename(document_path),
                        "Category": category,
                        "Similarity Score": similarity,
                        "Abstract": abstract[:100]  # Displaying the first 100 characters of the abstract
                    })

            # Create a DataFrame from the list of document information
            df = pd.DataFrame(document_info)

            st.markdown(f'<p style="text-align:center; color:red;">Ranked Documents</p>', unsafe_allow_html=True)
            st.table(df.head(100).sort_values(by="Similarity Score", ascending=False))


            # Print the ranked documents and calculate total similarity for each category
            print("Ranked Documents:")
            for index in sorted_indices[:min_length]:
                if index < len(document_paths):
                    document_path = document_paths[index]
                    category = os.path.basename(os.path.dirname(document_path))  # Extract category from the folder name
                    similarity = similarity_matrix[-1][index]

                    # Print document information
                    # print(f"{document_path} - Category: {category} - Similarity: {similarity}")

                    # Accumulate the total similarity for the category
                    category_similarity[category] += similarity

            # Print the total similarity for each category
            # print("\nTotal Similarity for Each Category:")
            # for category, total_similarity in category_similarity.items():
            #     print(f"{category}: {total_similarity}")

            # Sort and print the categories based on total similarity
            sorted_categories = sorted(category_similarity.items(), key=lambda x: x[1], reverse=True)
            # print("\nRanking by Total Similarity:")
            # for category, total_similarity in sorted_categories:
            #     print(f"{category}: {total_similarity}")

            # Visualization using Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            total_similarity_sum = sum(total_similarity for _, total_similarity in sorted_categories)
            percentage_values = [total_similarity for _, total_similarity in sorted_categories]

            bar_plot = sns.barplot(
                x=[category for category, _ in sorted_categories],
                y=percentage_values,
                palette="viridis"
            )

            bar_plot.set(xlabel="Category", ylabel="Total Similarity", title="Ranking by Total Similarity")
            plt.xticks(rotation=45, ha='right')

            # Display percentage values on top of the bars
            for i, value in enumerate(percentage_values):
                bar_plot.text(i, value , f"{value:.2f}", ha='center', va='bottom', fontsize=8, color='black')

            plt.tight_layout()
            st.pyplot(plt)

            # Step 6: Toggle Button for Comparison Table
            st.sidebar.subheader("Step 6: Toggle for Comparison Table")
            show_comparison_table = st.sidebar.checkbox("Show Comparison Table", value=True)

            if show_comparison_table:
                # Calculate Percentage of Total Similarity for each algorithm and each category
                similarity_dict = defaultdict(lambda: defaultdict(float))

                for tfidf_type in ['Standard TF-IDF', 'Smoothed TF-IDF', 'Probabilistic TF-IDF']:
                    tfidf_matrix = calculate_tfidf(preprocessed_documents, tfidf_type)
                    similarity_matrix = calculate_similarity(tfidf_matrix, similarity_measure)
                    sorted_indices = similarity_matrix[-1].argsort()[::-1]

                    for index in sorted_indices[:min_length]:
                        if index < len(document_paths):
                            document_path = document_paths[index]
                            category = os.path.basename(os.path.dirname(document_path))
                            similarity = similarity_matrix[-1][index]
                            similarity_dict[category][tfidf_type] += similarity

                # Normalize values to percentages for each category
                percentage_values = {}
                for category, values in similarity_dict.items():
                    total_similarity_sum = sum(values.values())
                    percentage_values[category] = {tfidf_type: value for tfidf_type, value in values.items()}

                # Display Comparison Table

                # Display Comparison Table
                st.markdown(f'<p style="text-align:center; color:red;">Total Similarity for TF-IDF Algorithms (by Category)</p>', unsafe_allow_html=True)

                # Create a DataFrame for the comparison table
                comparison_data = {'Category': list(percentage_values.keys())}
                for tfidf_type in ['Standard TF-IDF', 'Smoothed TF-IDF', 'Probabilistic TF-IDF']:
                    comparison_data[tfidf_type] = [values.get(tfidf_type, 0) for values in percentage_values.values()]

                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.set_index('Category', inplace=True)
                # # Highlight and print in red for "Category" column and TF-IDF algorithm columns
                styled_df = comparison_df.style.applymap(lambda x: f"color: {'red' if x != 0 else 'black'}") \
                                                .set_table_styles([
                                                    {'selector': 'thead',
                                                    'props': [('background-color', 'lightgrey')]}
                                                ])
                
                # Sort DataFrame based on selected tfidf_type
                if tfidf_type_copy in ['Standard TF-IDF', 'Smoothed TF-IDF', 'Probabilistic TF-IDF']:
                    sorted_df = comparison_df.sort_values(by=tfidf_type_copy, ascending=False)
                  # # Highlight and print in red for "Category" column and TF-IDF algorithm columns
                    sorted_df = comparison_df.style.applymap(lambda x: f"color: {'red' if x != 0 else 'black'}") \
                                                    .set_table_styles([
                                                        {'selector': 'thead',
                                                        'props': [('background-color', 'lightgrey')]}
                                                    ])
                    st.table(sorted_df)  # Display the sorted DataFrame
                else:
                    st.table(styled_df)  # Display the original DataFrame without sorting



if __name__ == "__main__":
    tf_idf()