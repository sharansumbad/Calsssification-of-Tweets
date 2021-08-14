# Calsssification-of-Tweets

A classifier is a machine learning model that is used to discriminate different objects based on certain features.A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem. The idea behind the classifier(or the approach) was taken from the reference[2] as it is easy to understand and implement we are pre-processing the training data as well as the test data by removing the special characters, extra spaces , numerical . We have also removed some english stop words and converted all the textual data to lower case. When we encounter a word not present in a city we punish(multiply by a pseudo count) that city by a factor. We use Naive bayes law: P(Posterior) = P( Likelihood)*P(Prior) to find how likely it is that a tweet belongs to a particular Region (East coast or West coast) with the maximum P(Posterior) value for a tweet will be classified as the region that tweet belongs to. We ignore the denominator of the Bayes law as it is constant for all regions.

The basic abstraction of the code can be as we calculate the prior and likelihood multiply them which gives us the probability that proportional to the Posterior probability and once we get this for both tha labels we can easily classify the document.

-The Prior probability are : P (East cost) = (Number of messages with East cost labels)/(Number of messages with East cost labels + Number of messages with West cost labels) P (West cost) = (Number of messages with West cost labels)/(Number of messages with East cost labels + Number of messages with West cost labels)

-The probability of (word | East coast) is calculated by (likelihood) frequency of the that particular word in East coast divided by total number of words in East coast and similarly for the P (word | West coast)

-The posterior probability are :(for a sentence) P (sentence | East coast) = P (East cost) * P (word 1 | East coast) * P (word 2 | East coast) ..... P (sentence | West coast) = P (West cost) * P (word 1 | West coast) * P (word 2 | West coast) .....

Based on these probability calculations we classify the document.



References
[1] https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
[2] https://www.youtube.com/watch?v=O2L2Uv9pdDA&t=626s
[3] https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
[4] https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
