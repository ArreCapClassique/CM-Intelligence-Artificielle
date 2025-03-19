import pandas as pd
import numpy as np
import os


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan


def featureEnginer(Xtrain, Xtest):
    X = pd.concat([Xtrain, Xtest])
    title_list = [
        "Mrs",
        "Mr",
        "Master",
        "Miss",
        "Major",
        "Rev",
        "Dr",
        "Ms",
        "Mlle",
        "Col",
        "Capt",
        "Mme",
        "Countess",
        "Don",
        "Jonkheer",
    ]
    X["Title"] = X["Name"].map(lambda x: substrings_in_string(x, title_list))
    X = X.drop("Name", axis=1)

    cabin_list = ["A", "B", "C", "D", "E", "F", "T", "G", "Unknown"]
    X["Cabin"] = X["Cabin"].fillna("Unknow")
    X["Deck"] = X["Cabin"].map(lambda x: substrings_in_string(x, cabin_list))

    #X = X.drop("Ticket", axis=1)

    X["FamilySize"] = X["SibSp"] + X["Parch"]

    X = pd.get_dummies(X, dummy_na=True)
    return X.iloc[: Xtrain.shape[0]], X.iloc[Xtrain.shape[0] :]


def getData():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    train = pd.read_csv(r"train.csv")
    Xtrain = train.iloc[:, 2:]
    ytrain = train.iloc[:, 1]
    test = pd.read_csv(r"test.csv")
    Xtest = test.iloc[:, 1:]
    Xtrain, Xtest = featureEnginer(Xtrain, Xtest)
    return Xtrain, Xtest, ytrain
