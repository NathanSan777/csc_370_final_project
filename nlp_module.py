"""
    A program to query senteces.
"""
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import math
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt
#Import FileReader to read in files and create dataframes
import file_reader as fr

def load_embeddings(filename):
    """
    This function loads the embedding from a file and returns 2 things
    1) a word_map, this is a dictionary that maps words to an index.
    2) a matrix of row vectors for each work, index the work using the vector.

    :param filename:
    :return: word_map, matrix
    """
    count = 0
    matrix = []
    word_map = {}
    with open(filename, encoding="utf8") as f:
        # with open(filename) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            word = items[0]
            rest = items[1:]
            # print("word:", word)
            word_map[word] = count
            count += 1

            rest = list(map(float, rest))
            matrix.append(rest)
    matrix = np.array(matrix)
    return word_map, matrix

def load_text_words(filename):
    """
    This function takes a text document and creates a list of words for the document.
    It returns the list of text words.
    :param filename:
    :return: text_words
    """
    text_words = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            items = list(map(str.lower, items))
            items = list(map(lambda x: x.strip('.,[]!?;:'), items))
            text_words.extend(items)
    # return a list of words from the text.
    return text_words

def sentence_to_vector(query_sentence, word_map, matrix):
    words = nltk.word_tokenize(query_sentence)
    sentence_vector = np.zeros(50)
    for word in words:
        # print(word)
        if word in word_map:
            sentence_vector += matrix[word_map[word],:]
    return sentence_vector
    
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
    embedding_filename = './glove.6B/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)
    file = './review_data/Training_Essay_Data.csv'
    df = fr.convert_csv_to_dataframe(file)
    df_text = df['text']
    df_labels = df['generated']
    human_text = df_text[df_labels == 0]
    ai_text = df_text[df_labels == 1]

    human_vector_length = len(human_text)
    ai_generated_vector_length = len(ai_text)

    human_vector = np.zeros(human_vector_length)
    ai_generated_vector = np.zeros(ai_generated_vector_length)
    user_sentence_vector = np.zeros(human_vector_length)
    for sentence in human_text:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word in word_map:
                human_vector += matrix[word_map[word], :]

    # Convert AI-generated text to vector
    for sentence in ai_text:
        words = nltk.word_tokenize(sentence)
        for word in words:
            if word in word_map:
                ai_generated_vector += matrix[word_map[word], :]

    # Convert user input to vector
    user_sentence_vector = sentence_to_vector(userInput, word_map, matrix)

    # Calculate cosine similarity between user input and both human-written and AI-generated text
    similarity_to_human = cossim(user_sentence_vector, human_vector)
    similarity_to_ai_generated = cossim(user_sentence_vector, ai_generated_vector)

    # Set a threshold for determining if the input is similar to AI-generated text
    threshold = 0.8  # Adjust the threshold accordingly

    if similarity_to_ai_generated > threshold:
        print("The input is similar to AI-generated text.")
    else:
        print("The input is likely human-written.")


    


main()