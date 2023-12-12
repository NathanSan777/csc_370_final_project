import pandas as pd
import numpy as np
import sklearn as sk
import nltk
import random
import matplotlib.pyplot as plt
#Import FileReader to read in files and create dataframes
import file_reader as fr

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def save_knn_model(model, filepath):
    """Save a trained knn model for later use"""
    joblib.dump(model, filepath)

def main():

    print("Currently running knn-model.")

    file = './review_data/Training_Essay_Data.csv'
    
    df = fr.convert_csv_to_dataframe(file)
    sample_df = fr.get_random_lines(df, 1000)
    np.random.seed(42)

    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(sample_df['text'])
    y = df['generated']

    # Loop through k values 1 to 21, to see which has the best accuracy
    k_values = range(1, 21, 2)

    #Create number of folds to split our dataframe into 
    num_folds = 10
    # Create 10 folds of our data: that means 100 test data points, 900 training data points for sample size of 1000
    kFolds = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    # Store accuracy scores later for total mean accuracy of model
    accuracy_scores = []

    # Go through each k value and use cross-validation
    for k in k_values:
        fold_scores = []
        # Get each set of training and testing indexes within the kFold 
        for train_index, test_index in kFolds.split(X):
            # Split our data into training and testing sets
            X_train, X_test = X[train_index].toarray(), X[test_index].toarray()
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Create a KNN classifier with k value
            model = KNeighborsClassifier(n_neighbors=k)
            # Train our model with the data we just got
            model.fit(X_train, y_train)
            # Predict the labels of our test data using that fold's prediction
            fold_score = model.score(X_test, y_test)
            fold_scores.append(fold_score)
        accuracy_scores.append(np.mean(fold_scores))
    
    overall_accuracy = np.mean(accuracy_scores)
    print("Overall mean accuracy: ", overall_accuracy)
    
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title('k-NN Cross Validation Scores')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross Validation Accuracy')
    plt.show()
    save_knn_model(model, 'knn_model.joblib')

main()