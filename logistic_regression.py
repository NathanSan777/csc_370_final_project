import pandas as pd
import numpy as np
import sklearn as sk
import nltk
import random
import matplotlib.pyplot as plt
import file_reader as fr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# 0 means human, 1 means ai generated

def main():
    print("Currently running logistic regression model.")
    # Use filereader.py to load in dataset. The main dataset.
    file = './review_data/Training_Essay_Data.csv'
    # These are other datasets we picked up to test if we are overfitting
    # our data. They'll be commented out when we're done testing this.
    file2 = './review_data/train_essays_7_prompts.csv'
    file3 = './review_data/train_essays_RDizzl3_seven_v1.csv'
    file4 = './review_data/train_essays_RDizzl3_seven_v2.csv'
    file5 = './review_data/train_essays_7_prompts_v2.csv'

    df = fr.convert_csv_to_dataframe(file)
    sample_df = fr.get_random_lines(df, 10000)
    # print(sample_df)
    vectorizer = TfidfVectorizer()

    col_names = ['text', 'generated']
    X = sample_df['text'] #The feature we're analyzing.
    y = sample_df['generated'] #The label we're trying to predict.
    print("Printing features:")
    print(X)
    print("Printing target variable:")
    print(y)

    #Create testing and training sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    # Since we're using binary classification, use tfidf vectors
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression(random_state=42)

    lr_model.fit(X_train_tfidf, y_train)
    y_pred = lr_model.predict(X_test_tfidf)

    score = accuracy_score(y_test, y_pred)
    print("Accuracy of logistic regression: ", score)

    # Cross-validation code 
    feature_names = vectorizer.get_feature_names_out()
    coef_values = lr_model.coef_[0]

    # Create a DataFrame to show feature importance
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef_values})
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

    # Print the top features
    print("Top 10 Features:")
    print(feature_importance_df.head(10))

    # Create and plot confusion matrix based on prediction

    cm = confusion_matrix(y_test, y_pred)

    # Since our test split is 0.25, the numbers inside of the boxes
    # should total to 25% of the number of reviews we read in.
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # Add labels and tick marks to the conf. matrix to look nice
    classes = ["Human-Written", "AI-Generated"]
    tickMarks = np.arange(len(classes))
    plt.xticks(tickMarks, classes, rotation=45)
    plt.yticks(tickMarks, classes)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Code to go through each of the confusion matrix's boxes and label them.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')

    plt.show()


main()