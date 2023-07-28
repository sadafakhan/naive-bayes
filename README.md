# naive-bayes
```naive-bayes``` implements both multinomial and bernoulli Naive Bayes classification on text documents. 

```build_NB1.sh``` implements the multivariate Bernoulli NB model. 
```build_NB2.sh``` implements the multinomial NB model. 

Args: 
* ```training_data```: vector file for training data in the text format
* ```test_data```: vector file for test data in the text format
* ```class_prior_delta```: delta used in the add-delta smoothing when calculating the class prior P(c)
* ```cond_prob_delta```: delta used in the add-delta smoothing when calculating the conditional probability P(f|c)

Returns: 
* ```model_file```: stores the values of P(c) and P(f|c) (cf. model1 under examples)
* ```sys_output```: is the classification result on the training and test data (cf. sys1 under examples)
* ```acc_file```: confusion matrix and the accuracy for the training and the test data (cf. acc1 under examples)

To run: 

```
src/build_NB[X].sh input/train.vectors.txt input/test,vectors.txt class_prior_delta cond_prob_delta output/model_file output/sys_output > output/acc_file
```

HW3 OF LING572 (01/26/2022)

