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
    # print(df)
    # print("File has been read and converted to a dataframe!")
    return df


def get_random_lines(df, amount):
    # print("Getting a random selection of ", amount, "rows in this dataframe.")
    total_rows = df.shape[0]
    random_indices = np.random.choice(total_rows, size=amount, replace=False)
    random_selection_df = df.iloc[random_indices]
    return random_selection_df


def main():

    print("Hello! Beginning to read file...")
    # file = './review_data/Training_Essay_Data.csv'
    # file2 = './review_data/train_essays_7_prompts.csv'
    # file3 = './review_data/train_essays_RDizzl3_seven_v1.csv'
    # file4 = './review_data/train_essays_RDizzl3_seven_v2.csv'
    # file5 = './review_data/train_essays_7_prompts_v2.csv'

    # df = convert_csv_to_dataframe(file)
    # sample_df = get_random_lines(df, 100)
    # print(sample_df)
    

main()