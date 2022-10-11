import pandas as pd
import random as rd
import os
import pickle


# Function to load pickled data
def load_pickled_data(stage: str):
    # stage is either raw, filtered, or welch

    train_path = "data/pickles/" + stage + "_train_data.pkl"
    eval_path = "data/pickles/" + stage + "_eval_data.pkl"

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    with open(eval_path, "rb") as f:
        eval_data = pickle.load(f)

    return train_data, eval_data


# Function to save pickled data
def save_pickled_data(train_data, eval_data, stage: str):
    # stage is either raw, filtered, or processed

    train_path = "data/pickles/" + stage + "_train_data.pkl"
    eval_path = "data/pickles/" + stage + "_eval_data.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)

    with open(eval_path, "wb") as f:
        pickle.dump(eval_data, f)


# Function to load and modify original csv files, returning a dataframe
def parse_to_df(filepath):

    # each file has two lines of header info
    # - age, given in "Age = 50" format
    # - column names, but first column has a # in front of it

    # grab the age from the first line
    age = int(filepath.readline().split(" ")[2])

    with open(filepath) as f:
        age = f.readline().split(" ")[2]
        # remove the # from the first column name
        column_names = f.readline().split(",")

    # change the first column name to "EEG FP1-REF"
    column_names[0] = "EEG FP1-REF"

    df = pd.read_csv(filepath, skiprows=2, names=column_names)

    # crop the data down to the rows between 200000 and 300000
    df = df.loc[200000:300000]

    # select a random start index between 0 and (100000 - 1250)
    start_index = rd.randint(0, 100000 - 1250)

    # grab the next 1250 rows from the start index
    df = df.loc[start_index: start_index + 1250]

    # only keep the columns we need
    # - EEG FP1-REF, EEG FP2-REF, EEG F3-REF, EEG F4-REF, EEG C3-REF, EEG C4-REF, EEG P3-REF,
    # - EEG P4-REF, EEG F7-REF, EEG F8-REF, EEG T4-REF, EEG T5-REF, EEG T6-REF, EEG A1-REF,
    necessary_cols = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF",
                      "EEG P3-REF", "EEG P4-REF", "EEG F7-REF", "EEG F8-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF", "EEG A1-REF"]
    df = df[necessary_cols]

    return df, age


# Function to load data from the data/eval and data/train folders
def load_train_test_data(n_subjects):
    # n_subjects will be the number of subjects to load
    # both train and test data are pickled for faster loading

    # grab  a list of the contents of the data/train folder
    train_data_paths = os.listdir("data/train")

    # grab a list of the contents of the data/eval folder
    # will test against all subjects in eval folder
    eval_data_paths = os.listdir("data/eval")

    # reduce to a random sample of n_subjects from the data/train folder
    train_data_paths = rd.sample(train_data_paths, n_subjects)

    # reduce evaluation data to a random sample of n_subjects//3 from the data/eval folder
    eval_data_paths = rd.sample(eval_data_paths, n_subjects // 3)

    # prepare two lists to hold the dataframes
    train_data = []
    eval_data = []

    # load the dataframes from the train data
    for t_path in train_data_paths:
        train_data.append(parse_to_df("data/train/" + t_path))

    # load the dataframes from the eval data
    for e_path in eval_data_paths:
        eval_data.append(parse_to_df("data/eval/" + e_path))

    # pickle the data for faster loading
    save_pickled_data(train_data, eval_data, "raw")

    # return the data
    return train_data, eval_data
