"""Sadaf Khan, LING570, HW3, 01/26/2021. Implements a Multinomial Naive Bayes model"""

import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import math

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
    for label in word_counts:
        feat = label.split(":")[0]

        # if an encountered word-feature isn't accounted for in the vocabulary
        # add it into the reference dicts
        if feat not in vocab:
            vocab[feat] = feat2col
            columns[feat2col] = feat
            feat2col += 1

# create array of training instance x word count
training_array = np.zeros((inst2row, feat2col))
for i in range(inst2row):
    split = formatted_training[i].split(" ")
    class_label = split[0]
    word_counts = split[1:-1]

    # keep track of training labels
    training_labels.append(class_label)

    # gather word count information
    for pair in word_counts:
        feat = pair.split(":")[0]
        count = pair.split(":")[1]

        # since this is multinomial, we care about the particular count
        training_array[i, vocab[feat]] = count

# add rows and column headers, for later convenience
training_df = pd.DataFrame(data=training_array,
                           index=['array' + str(i)
                                  for i in range(training_array.shape[0])],
                           columns=[columns[i]
                                    for i in range(len(columns))])

# add class label column
training_df['%Class_Label%'] = training_labels

# TRAINING
# want to estimate P(c|x), P(c) and P(w|c)

# hold onto feature distributions per class label and word count
cond_dist = {}
prior_dist = {}

for label in set(training_labels):
    label_df = training_df.loc[(training_df['%Class_Label%'] == label)]
    w_given_c = {}

    # estimate P(c)
    c_count = training_df['%Class_Label%'].value_counts([label])

    if class_prior_delta > 0:
        p_c = (c_count + class_prior_delta) / (len(vocab) + c_count * len(set(training_labels)))
    else:
        p_c = c_count / (c_count * len(set(training_labels)))

    prior_dist[label] = p_c

    # estimate P(w|c)
    for word in vocab:
        num = label_df[word].sum() + cond_prob_delta
        denom = len(vocab) * cond_prob_delta + label_df.sum(numeric_only=True).sum()
        p_w_c = num/denom
        w_given_c[word] = p_w_c



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
# want to calculate argmax P(c) P(x|c). We have P(c).
# P(x|c) = Product P(w_k|c)^count(w_k) for every word in the test instance

def classify(instance):
    split = instance.split(" ")
    actual_label = split[0]
    word_counts = split[1:-1]

    label_dist = {}
    for label in cond_dist:

        # initialize probability to P(c)
        likelihood = math.log(prior_dist[label], 10)

        for pair in word_counts:
            feat = pair.split(":")[0]
            count = pair.split(":")[1]

            if feat in cond_dist[label]:
                p_w_c = cond_dist[label][feat]**count
                likelihood += math.log(p_w_c, 10)

            # word is OOV
            else:
                likelihood += math.log(cond_prob_delta / (cond_prob_delta * len(vocab)), 10)

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
        actual_testing.append(actual_label)
        output.write("array:" + str(array_counter) + " " + actual_label + " ")
        array_counter += 1

        labels_sorted = ""
        for label in sorted(label_dist, key=label_dist.get):
            labels_sorted += (label + " " + str(label_dist[label]) + " ")

        predicted_training.append(sorted(label_dist, key=label_dist.get)[0])
        output.write(labels_sorted + "\n")

    output.write("%%%%% testing data:\n")
    formatted_testing = open(os.path.join(os.path.dirname(__file__), test_data), 'r').read().split("\n")[:-1]

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
print("Training accuracy=" + str(train_accuracy))


print("Confusion matrix for the testing data: \n")
print("row is the truth, column is the system output \n \n")
print(test_formatted)
print("\n\n")
print("Testing accuracy=" + str(test_accuracy))