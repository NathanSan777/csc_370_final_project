"""
A program to test the accuracy of GloVe vectors using KNN.
The program reads in a CSV file of reviews,converts them into GloVe 
vectors, and then makes predictions on if a review is human-written 
or AI-generated after being trained. 
It then displays the accuracy for each fold.

By: Nathan Sanchez, Trung Pham, Suleman Baloch
  
"""
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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
    """
    A method to convert a sentence into a GloVe vector.

    """
    #Tokenize sentence
    words = nltk.word_tokenize(sentence)
    #Convert the tokens into the vector
    sentence_vector = np.zeros((matrix.shape[1], ))
    # If the words inside of the sentence are in the word map,
    # add the words from the word map into the sentence vector
    for word in words:
        # print(word)
        if word in word_map:
            sentence_vector += matrix[word_map[word],:]
    return sentence_vector

def convert_csv_to_glove(csv_file, wordmap, matrix):
    """
    A method to convert a CSV file into a GloVe vector.
    It takes in a csv file, makes it a dataframe, and then
    converts each sentence into a GloVe vector.

    """
    # Load in our dataframe
    df = fr.convert_csv_to_dataframe(csv_file)
    df = fr.get_random_lines(df, 1000)
    # Get columns
    df_text = df['text']
    df_labels = df['generated']

    text_vectors = []
    for sentence in df_text: 
        text_vector = sentence_to_vector(sentence, wordmap, matrix)
        text_vectors.append(text_vector)
    # Return a tuple of an array of text vectors, as well as
    # the labels of the dataframe
    return np.array(text_vectors), df_labels

def train_knn_classifier(text_vectors, labels):
    """
    A method to train a knn model with GloVe vectors, where it 
    takes a random fold and goes through k neighbors.

    """
    fold_accuracies = []

    for k in range(1, 21, 2):
        X_train, X_test, y_train, y_test = train_test_split(text_vectors, labels, test_size=0.2, random_state=42)
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)

    cross_val_accuracies = []
    for k in range(1, 21, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        cross_val_accuracy = np.mean(cross_val_score(knn_classifier, text_vectors, labels))
        cross_val_accuracies.append(cross_val_accuracy)

    print("Average accuracy of model: ", np.mean(cross_val_accuracies))
    plt.plot(range(1,21, 2), cross_val_accuracies, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Scores')
    plt.show()

    return knn_classifier                          

def main():
    
    #Load embeds 
    embedding_filename = './glove.6B/glove.6B.50d.txt'
    word_map, matrix = load_embeddings(embedding_filename)

    #Load in csv and get texts from it
    file = './review_data/Training_Essay_Data.csv'
    df = fr.convert_csv_to_dataframe(file)

    text_vectors, labels = convert_csv_to_glove(file, word_map, matrix)

    knn_glove_model = train_knn_classifier(text_vectors, labels)

main()