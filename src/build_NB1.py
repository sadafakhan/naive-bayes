"""Sadaf Khan, LING570, HW3, 01/26/2021. Implements a Multivariate Bernoulli Naive Bayes model on a text file of
feature vector, class label pairs."""

import math
import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# TAKING INPUTS

# vectors in .txt format

training_data = sys.argv[1]
test_data = sys.argv[2]

# delta used in add-delta smoothing when calculating the class prior P(c)
class_prior_delta = float(sys.argv[3])

# delta used in add-delta smoothing when calculating the conditional probability P(f|c)
cond_prob_delta = float(sys.argv[4])

# stores the values of P(c) and P(f|c)
model_file = sys.argv[5]

# classification result on the training and test data
sys_output = sys.argv[6]

# FORMATTING DATA

# initialize counters, data structures
inst2row = 0
feat2col = 0
vocab = {}
columns = {}
training_labels = []

# format training data
formatted_training = open(os.path.join(os.path.dirname(__file__), training_data), 'r').read().split("\n")[:-1]

# extract the vocabulary (columns) and number of instances (rows)
for vec in formatted_training:
    inst2row += 1
    word_counts = vec.split(" ")[1:-1]

    for pair in word_counts:
        feat = pair.split(":")[0]

        # if an encountered word-feature isn't accounted for in the vocabulary
        # add it into the reference dicts
        if feat not in vocab:
            vocab[feat] = feat2col
            columns[feat2col] = feat
            feat2col += 1

# create array of training instance x word presence
training_array = np.zeros((inst2row, feat2col))
for i in range(inst2row):
    split = formatted_training[i].split(" ")
    class_label = split[0]
    word_counts = split[1:-1]

    # keep track of training labels
    training_labels.append(class_label)

    # gather word presence information
    for pair in word_counts:
        feat = pair.split(":")[0]

        # since this is bernoulli, we only care if the word exists,
        # but we don't care about its particular count
        training_array[i, vocab[feat]] = 1

# add rows and column headers, for later convenience
training_df = pd.DataFrame(data=training_array,
                           index=['array' + str(i)
                                  for i in range(training_array.shape[0])],
                           columns=[columns[i]
                                    for i in range(len(columns))])

# add class label column
training_df['%Class_Label%'] = training_labels

# TRAINING
# want to estimate P(c) and P(w|c)

# hold onto feature distributions per class label
cond_dist = {}
prior_dist = {}


# probability of a (x | c_i) = product of probabilities w_1....w_n | c_i
# first, calculate P(w_t | c_i) and P(c_i)
def counter(label):
    w_given_c = {}
    c_count = training_df['%Class_Label%'].value_counts()[label]

    # count (word, class_label) instances
    for word in vocab:
        # find instances in dataframe
        word_df = training_df.loc[(training_df[word] == 1) & (training_df['%Class_Label%'] == label)]
        w_c_count = word_df.shape[0]

        # account for smoothing
        p_w_c = (w_c_count + cond_prob_delta) / (c_count + (2 * cond_prob_delta))

        # set feature likelihood in this particular class
        w_given_c[word] = p_w_c

    # more smoothing
    if class_prior_delta > 0:
        p_c = (c_count + class_prior_delta) / (len(vocab) + c_count * len(set(training_labels)))
    else:
        p_c = c_count / (c_count * len(set(training_labels)))

    # assign the distribution to a label
    cond_dist[label] = w_given_c
    prior_dist[label] = p_c


# execute for our particular labels
counter("talk.politics.guns")
counter("talk.politics.mideast")
counter("talk.politics.misc")

# write model to file
with open(model_file, 'w') as model:
    model.write("%%%%% prior prob P(c) %%%%%\n")
    for label in set(training_labels):
        prob = prior_dist[label]
        model.write(label + "\t" + str(prob) + "\t" + str(math.log(prob, 10)) + "\n")

    model.write("%%%%% conditional prob P(f|c) %%%%%\n")
    for label in cond_dist:
        model.write("%%%%% conditional prob P(f|c) c=" + label + " %%%%%\n")
        for feat in cond_dist[label]:
            p_w_c = cond_dist[label][feat]
            model.write(feat + "\t" + label + "\t" + str(p_w_c) + "\t" + str(math.log(p_w_c, 10)) + "\n")

# TESTING
# want to calculate P(c|x) = P(c)P(x|c)/P(x). we have P(c).
# x = f1....fn, so P(x|c) = product of all P(f|c) = product of all P(w|c) * product of all 1-P(w|c)
# = Z_c * product of all (P(w|c)/(1-P(w|c)), where Z_c = product of all (1 - P(w|c))


# calculate Z_c beforehand
Z = {}

for label in cond_dist:
    # initialize to 0, as log_base 10 of multiplicative identity is 0
    z = 0
    # calculate product of all (1 - P(w|c)) across vocab
    for feat in vocab:
        p_w_c = cond_dist[label][feat]
        z += math.log(1 - p_w_c, 10)
    Z[label] = z

formatted_testing = open(os.path.join(os.path.dirname(__file__), test_data), 'r').read().split("\n")[:-1]


# calculate P(c)P(x|c)/P(x) = P(c) * Z_c * (product of all (P(w|c)/(1-P(w|c)) for w in x) / P(x)
# we have P(c) and Z_c. We need the Prod(P(w|c)/(1-P(w|c)) and P(x)
def classify(instance):
    split = instance.split(" ")
    actual_label = split[0]
    word_counts = split[1:-1]

    label_dist = {}
    for label in cond_dist:
        likelihood = 0

        # iterate through the features and multiply all (P(w|c)/(1-P(w|c))
        for pair in word_counts:
            feat = pair.split(":")[0]

            if feat in cond_dist[label]:
                p_w_c = cond_dist[label][feat]
                p_not_w_c = 1 - p_w_c
                likelihood += math.log((p_w_c / p_not_w_c), 10)

            # word is OOV
            else:
                likelihood += math.log(cond_prob_delta / (cond_prob_delta * len(vocab)), 10)

        # multiply by Z_c to get P(x|c)
        likelihood += Z[label]

        # calculate P(x) = Summation P(x|c) P(c) for all c
        p_x = 0
        for c in prior_dist:
            p_c = math.log(prior_dist[c], 10)
            p_x_c = likelihood
            p_x += (p_c + p_x_c)

        # multiply by P(c)
        likelihood += math.log(prior_dist[label], 10)

        # divide by P(x)
        likelihood -= p_x

        label_dist[label] = likelihood

    return label_dist, actual_label


# for confusion matrix use
actual_testing = []
actual_training = []
predicted_testing = []
predicted_training = []

array_counter = 0
# start writing to system
with open(sys_output, 'w') as output:
    output.write("%%%%% training data:\n")

    for vec in formatted_training:
        label_dist, actual_label = classify(vec)
        actual_training.append(actual_label)

        output.write("array:" + str(array_counter) + " " + actual_label + " ")
        array_counter += 1

        training_labels_sorted = ""
        for label in sorted(label_dist, key=label_dist.get):
            training_labels_sorted += (label + " " + str(label_dist[label]) + " ")

        predicted_training.append(sorted(label_dist, key=label_dist.get)[0])
        output.write(training_labels_sorted + "\n")

    output.write("%%%%% testing data:\n")

    for vec in formatted_testing:
        label_dist, actual_label = classify(vec)
        actual_testing.append(actual_label)

        output.write("array:" + str(array_counter) + " " + actual_label + " ")
        array_counter += 1

        testing_labels_sorted = ""
        for label in sorted(label_dist, key=label_dist.get):
            testing_labels_sorted += (label + " " + str(label_dist[label]) + " ")

        predicted_testing.append(sorted(label_dist, key=label_dist.get)[0])
        output.write(testing_labels_sorted + "\n")

# EVALUATE
label_set = ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"]
# create confusion matrices and accuracy scores
train_cm = confusion_matrix(actual_training, predicted_training, labels=label_set)
train_accuracy = accuracy_score(actual_training, predicted_training)
train_formatted = pd.DataFrame(train_cm, index=label_set, columns=label_set)


test_cm = confusion_matrix(actual_testing, predicted_testing, labels=label_set)
test_accuracy = accuracy_score(actual_testing, predicted_testing)
test_formatted = pd.DataFrame(test_cm, index=label_set, columns=label_set)

pd.set_option("display.max_rows", None, "display.max_columns", None)

print("Confusion matrix for the training data: \n")
print("row is the truth, column is the system output \n \n")
print(train_formatted)
print("\n\n")
print("Training accuracy="+str(train_accuracy))


print("Confusion matrix for the testing data: \n")
print("row is the truth, column is the system output \n \n")
print(test_formatted)
print("\n\n")
print("Testing accuracy="+str(test_accuracy))

