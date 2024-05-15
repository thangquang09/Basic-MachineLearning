- [Ensemble Learning and Random Forests](#ensemble-learning-and-random-forests)
  - [Introduce](#introduce)
  - [Voting Classifiers](#voting-classifiers)


# Ensemble Learning and Random Forests

## Introduce

Suppose you pose a complex question to thousands of random people, then aggregate their answers. In many cases you will find that this aggregated answer is better than an expert’s answer. This is called the `wisdom of the crowd`. 

Example: Train a `group of decision tree classifiers`, each on a different random subset of the training set. You `take the predictions of all the individual trees`, and the `class gets most votes` is the ensemble's prediction.

## Voting Classifiers

When you have trained a few classifiers that each on achieving about 80% accuracy. A simple way to create an even better is aggregate the predictions of each classifiers: the class which gets the most votes is the ensemble prediction.

![Few classifiers](images/few_classifiers.png)

![Ensemble's prediction](images/ensemble_predictions.png)

The `voting classifier` often achieves a `higher accuracy than` the `best classifier` in the ensemble. In fact, even if each classifier is a `weak learner` (low accuracy), the ensemble can still be a `strong learner` (achieve high accuracy), as long as you provide a `sufficient number of weak learners` in the ensemble and they are `sufficiently diverse`.

Example: You have a slightly biased coin that has a `51% chance of coming up with heads and 49% coming up with tails`. If you toss it `1000 time`, you will get `more or less 510 heads and 490 tails`, and hence a majority of heads. If you toss the coin continuously, the probability of obtaining a majority of head will increase. It's due to `law of large number`. The `ratio of heads` gets `closer` and closer to the `probability of head` (51%).

![The law of large number](images/law_of_large_number.png)

Now that, suppose you build an ensemble containing `1000 classifiers` that are `individually correct only 51% of the time`. If you predict the majority `voted class`, you can hope for `up to 75% accuracy`. But it `only happens` if all classifiers are `perfectly independent` (Unless, they are likely to make the same types of errors). Because of that, we will train them `using very different algorithms`.

Soft voting is averaged over all the individual classifiers probability predictions (If all the classifiers have `predict_proba()` method). It often achieves higher performance than hard voting because it gives more weight to highly confident votes.






