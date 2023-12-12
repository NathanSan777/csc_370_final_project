import pandas as pd
import numpy as np
import sklearn as sk
import nltk
import random
import matplotlib.pyplot as plt


# A method to quickly convert our csv data into a dataframe.
# A value of 0 indivates that it is a human-written review,
# and a value of 1 indicates that it is a machine-generated review.
def convert_csv_to_dataframe(filepath):
    df = pd.read_csv(filepath, sep=",")
    df = df.dropna()
    return df


def get_random_lines(df, amount):
    # print("Getting a random selection of ", amount, "rows in this dataframe.")
    total_rows = df.shape[0]
    random_indices = np.random.choice(total_rows, size=amount, replace=False)
    random_selection_df = df.iloc[random_indices]
    return random_selection_df


def main():

    print("Hello! Beginning to read file...")  

main()