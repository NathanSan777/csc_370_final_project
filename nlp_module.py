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


def sentence_to_vector(sentence, word_map, matrix):
    words = nltk.word_tokenize(sentence)
    sentence_vector = np.zeros((matrix.shape[1], ))
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
    #Load embeds 
    embedding_filename = './glove.6B/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)

    #Load in csv and get texts from it
    file = './review_data/Training_Essay_Data.csv'
    df = fr.convert_csv_to_dataframe(file)
    df_text = df['text']
    df_labels = df['generated']
    # Seperate human and ai texts
    human_text = df_text[df_labels == 0]
    ai_text = df_text[df_labels == 1]

    #Create empty vectors to store info in
    human_vector = np.zeros((matrix.shape[1], ))
    ai_generated_vector = np.zeros((matrix.shape[1], ))
    user_sentence_vector = np.zeros((matrix.shape[1], ))


    #Convert human text to vector
    for sentence in human_text:
        sentence_vector = sentence_to_vector(sentence, word_map, matrix)
        human_vector += sentence_vector
    print("Done tokenizing human text.")


    # Convert AI-generated text to vector
    for sentence in ai_text:
        sentence_vector = sentence_to_vector(sentence, word_map, matrix)
        ai_generated_vector += sentence_vector

    print("Done tokenizing ai text.")

    # Convert user input to vector
    user_sentence_vector = sentence_to_vector(userInput, word_map, matrix)

    print("Shape of user_sentence vector: ", user_sentence_vector.shape)
    print("Shape of human vector: ", human_vector.shape)
    print("Shape of ai_generated_vector: ", ai_generated_vector.shape)
    print("Human vector is: ", human_vector)
    print("AI vector is ", ai_generated_vector)
    print("User vector is: ", user_sentence_vector)

    # Calculate cosine similarity between user input and both human-written and AI-generated text
    #Error here with ValueError
    similarity_to_human = cossim(user_sentence_vector, human_vector)
    similarity_to_ai_generated = cossim(user_sentence_vector, ai_generated_vector)
    print("Simularity to human-written text: ", similarity_to_human)
    print("Similarity to AI-generated text: ", similarity_to_ai_generated)

    #Based on the similarities, determine if the input is human-written or AI-generated
    if similarity_to_ai_generated > similarity_to_human:
        print("The input is similar to AI-generated text.")
    else:
        print("The input is likely human-written.")
    
    # Create a bar chart
    labels = ['Human-written simularity', 'AI-generated simularity']
    similarities = [similarity_to_human, similarity_to_ai_generated]
    #Plot the similarities
    plt.bar(labels, similarities, color=['blue', 'red'])
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity between User Input and Text Types')
    plt.show()

main()