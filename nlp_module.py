"""
    A program to query senteces.
"""
import nltk
import numpy as np
import math
import pandas as pd
import numpy as np
import sklearn as sk
import random
import matplotlib.pyplot as plt
#Import FileReader to read in files and create dataframes
import file_reader as fr
import joblib
import knn_module as knn

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def load_glove_vectors(filepath):
    glove_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_vectors[word] = vector
        return glove_vectors

def process_user_input(user_input, glove_vectors):    
    tokens = user_input.lower().split()
    vectors = [glove_vectors.get(token, np.zeros(len(list(glove_vectors.values())[0]))) for token in tokens]
    max_len = max(len(v) for v in vectors)
    vectors = [np.pad(v, (0, max_len - len(v)))[:max_len] for v in vectors] 
    feature_vector = np.mean(vectors, axis=0)
    return feature_vector
   


def cossim(vA, vB):
    """
    Calcuate the cosine similarity value.
    Returns the similarity value, range: [-1, 1]
    :param vA:
    :param vB:
    :return: similarity
    """
    return np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))

def main():
    print("Hello! Welcome to our text analyzer. Please input some text.")
    userInput = input()
    knn_model = joblib.load('knn_model.joblib')
    glove_file_path = './glove.6B/glove.6B.50d.txt'
    vectorizer = TfidfVectorizer()

    glove_vectors = load_glove_vectors(glove_file_path)
    user_feature_vector = process_user_input(userInput, glove_vectors, vectorizer)
    # currently running into issues with X's feature size...
    prediction = knn_model.predict(user_feature_vector.reshape(1, -1))
    if (prediction[0] == 0):
        print("The text you entered is likely human-written.")
    else:
        print("The input is likely AI-generated.")




main()