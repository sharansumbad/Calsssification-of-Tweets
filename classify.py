# classify.py : Classify text objects into two categories
#
# Author:  Sharanbasav Sumbad  (ssumbad)
# m        Milind Vakharia     (mivakh)
#          Vedant Benadikar    (vbenadik)
#
#
# Based on skeleton code by D. Crandall, March 2021

# A classifier is a machine learning model that is used to discriminate different objects based on certain features.A
# A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of
# the classifier is based on the Bayes theorem. The idea behind the classifier was (or the approach) was taken from
# the reference[2] as it is easy to understand and implement Firstly, we are pre-processing the training data as well
# as the test data by removing the special characters, extra spaces , numerical . We have also removed some of the
# english stop words and converted all the textual data to lower case.  When we encounter a word not present in a
# city we punish(multiply by a pseudo count) that city  by a factor. We use Naive bayes law: P(Posterior) = P(
# Likelihood)*P(Prior) to find how likely it is that a tweet belongs to a particular Region (East coast or West coast)
# with the maximum P( Posterior) value for a tweet will be classified as the region that tweet belong to. We
# ignore the denominator of the Bayes law as it is constant for all regions.


# References
# [1] https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
# [2] https://www.youtube.com/watch?v=O2L2Uv9pdDA&t=626s
# [3] https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# [4] https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/

import re
import sys
import string
from string import digits
import pandas as pd
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_file(filename):
    objects = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ', 1)
            labels.append(parsed[0] if len(parsed) > 0 else "")
            objects.append(parsed[1] if len(parsed) > 1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}


# The function below cleans the data by removing the special characters , extra space , numeric data and concerts the
# remaining into lower-case, we first converted it in to data frame as we were more comfortable with it and also
# experimented with in jupyter notebook while cleaning the data and then returned it back into the dictionary format
# as the load_file does

def CleanData(data):
    # convert the dictionary into a Data frame
    data = pd.DataFrame.from_dict(data, orient='index')
    data = data.transpose()
    # seprating classes
    classes = data["classes"]
    classes.dropna(inplace=True)
    data = data.drop(['classes'], axis=1)
    # common stop Words in English , English Stop-word list referred from
    # "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's
    # %2520list%2520of%2520english%2520stopwords" and added a few  more words by inspecting data.
    stop_words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all', 'just',
                  'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to',
                  'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him',
                  'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are',
                  'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 'be', 'we', 'who',
                  'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 'or', 'own', 'into', 'yourself', 'down',
                  'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more',
                  'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will',
                  'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have',
                  'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most',
                  'such', 'why', 'off', 'yours', 'so', 'the', 'having', 'once', 'jobs', 'job', 'amp', 'im']
    data["objects"] = data['objects'].str.replace('[^\w\s]', '')
    # cleaning data removing numbers and special characters and converting it into all lower case
    data["objects"] = data.replace(regex=r'[0-9]', value=' ')
    data["objects"] = data.replace(regex=r'_', value=' ')
    data['objects'] = data['objects'].str.lower()
    pat = '|'.join(r"\b{}\b".format(x) for x in stop_words)
    data['objects'] = data['objects'].str.replace(pat, '')
    # converting back to the original dictionary and again adding class
    clean_data = data.reset_index().to_dict(orient='list')
    clean_data['classes'] = classes.to_list()
    clean_data.pop("index")
    return clean_data


# here use of logarithmic arithmetic is done to avoid numerical underflow (as mentioned in detail in reference [4])
def fit_model(vocabulary, prior_prob, likelihood_prob, classes, test_data):
    result = []
    for line in test_data["objects"]:
        posterior_prob = math.log(prior_prob[0]) - math.log(prior_prob[1])
        extractedsentence = sentence(line)
        for word in extractedsentence:
            if word in vocabulary:
                posterior_prob += (math.log(likelihood_prob[0][word]) - math.log(likelihood_prob[1][word]))
        if posterior_prob > 0:
            result.append(classes[0])
        else:
            result.append(classes[1])
    return result


# the approach is simple we maintain the counts of the attributes and then later use them in the calculation of the
# posterior , prior and likelihood. we also go for laplace smoothing which would add a constant to all the
# probabilities as the ones with zero probabilities would results in zero probability when multiplied.

def classifier(train_data, test_data):
    train = CleanData(train_data)
    test = CleanData(test_data)
    priors = []
    classes_no = len(train["classes"])
    classcount = [0 for x in range(classes_no)]
    wordcount = [{} for x in range(classes_no)]
    vocabulary = {}
    for line_no in range(len(train["objects"])):
        classidx = train["classes"].index(train["labels"][line_no])
        classcount[classidx] += 1
        line = sentence(train["objects"][line_no])
        for word in line:
            if word not in vocabulary:
                wordcount[0][word] = 0
                wordcount[1][word] = 0
                vocabulary[word] = True
            wordcount[classidx][word] += 1
    total_msgs = sum(classcount)
    for i in range(classes_no):
        priors.append(classcount[i] / total_msgs)
    likelihoods = [{} for x in range(classes_no)]
    uniquewords = len(vocabulary)
    for word in vocabulary:
        likelihoods[0][word] = (wordcount[0][word] + 1) / (classcount[0] + uniquewords)
        likelihoods[1][word] = (wordcount[1][word] + 1) / (classcount[1] + uniquewords)
    result = fit_model(vocabulary, priors, likelihoods, train["classes"], test)
    return result


# This is a function that takes the list of list present in the dictionary (that is a sentence) and returns just the
# list of words in the particular sentences.
def sentence(line):
    sentencelist = []
    for word in line.split(" "):
        if word != "" and word != " ":
            sentencelist.append(word)
    return sentencelist


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if (train_data["classes"] != test_data["classes"] or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results = classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))

