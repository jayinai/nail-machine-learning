# Machine Learning Concepts

### <ins> Supervised vs. Unsupervised Learning?

In supervised learning the machine learns a ***function that maps between X (all the features, or input) and Y (label, or output)***, so that it can make prediction on new data - given unseen X, predict its Y. For example, given the subject line, text body, and senders' email address, we can train machines to classify whether the a new email is a spam or not.

On the other hand, in unsupervised learning there is ***no label given, so machines have to learn and discover patterns from data itself (without human supervision)***. One example is that given user related data, such as their location, browsing history, and logged times (during the past 12 hrs, 24 hrs, 3 days, 1 week, etc.), the machine would learn to cluster users into different groups (which can be useful for target marketing, for instance).

So in a nutshell - supervised: supervise/teach machines to make prediction; unsupervised: "dear machine, I trust you, discover some hidden patterns for me on your own"



### <ins> Regression vs. Classification?

Both are supervised learning problems. The difference is that in regression we are trying to predict ***continuous values*** (e.g., house prices), while in classification problem we try to make predictions on ***discrete values***: e.g., is the email spam or not (2 discrete values), is the image of a dog, cat, or fish (3 discrete values)?

A regression algorithm example is [linear regression](https://en.wikipedia.org/wiki/Linear_regression), and its classification counterpart is [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) - don't be fooled by its name, even though it's called regression, it's used to tack classification problems (e.g., spam or not)

And in reality classification algorithm such as logistic regression doesn't just output 'spam' or 'non-spam' (or in the numeric world, 1 or 0). Rather, they'd generate a probability between 0 and 1. While by default 0.5 is used as the decision threshold, you can definitely set your own threshold based on your use case (For example, mistakenly labeling a non-spam message as spam is very bad, you'd be pissed when you find an important email has been sitting in your spam folder for days. On the other hand, mistakenly labeling a spam message as non-spam is unpleasant, but is not the end of world).

### <ins> Classification: Binary vs. Multiclass?

A binary classification is where we have two possible outcomes (spam vs. non-spam), while a multiclass is where we have, well, multiple possible outputs (or classes). For example, is the sentiment of an article positive, negative, or neutral?

We can formulate a multiclass problem using binary classification: given a classification problem with N possible outcomes, a ***one-vs.-all*** method consists of N separate binary classifiers - one binary classifier for each possible outcome. For sentiment analysis we would have 3 binary classifiers:

1. Is the article's sentiment positive? No
2. Is the article's sentiment negative? No
3. Is the article's sentiment neutral? Yes

This method would work when the total number of classes is small, but becomes increasingly inefficient as the number of classes rises. That's when ***Softmax*** comes in.

While a binary classifier outputs a decimal probability between 0 and 1.0, ***Softmax assigns decimal probabilities to each class in a multi-class problem***. Those decimal probabilities must add up to 1.0. For the sentiment analysis use case, we might have these probabilities:

1. Positive: 0.27
2. Negative: 0.09
3. Neutral: 0.64

When we have a huge number of classes (e.g., 10k+), regular Softmax can start to become expensive too. That's when we can apply ***candidate sampling***, which means that Softmax calculates a probability for all the positive labels but only for a random sample of negative labels. Think of it this way:

>If we are interested in determining whether an input image is a beagle or a bloodhound, we don't have to provide probabilities for every non-doggy example.


Also, Softmax assumes that each example is a member of exactly one class. Some examples, however, can simultaneously be a member of multiple classes (for instance, you have an image that has both cat and dog in it). In that case you need to rely on a [multioutput classifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html).

Don't confuse multiclass with multioutput.

### <ins> How Does a Model Learn?

Let's say we have a linear model `y = w1x1 + b`, where `y` represents the house price (label), and `x1` is the number of bedroom (feature). `w1` and `b` are learnable parameters, weight and bias. How does the model learn?

We can initialize the learnable parameters with some random values, say, 0s. Then we need to ***train*** the model by asking it to look at different examples, use the current parameters to make predictions `y'`, and compare the difference between `y` (ground truth) and `y'` (prediction). Ideally we want the difference to be as small as possible, so we need to figure out a way to minimize the difference (or ***loss***).

By ***iteratively reducing the loss*** (and updating the learnable parameters), the model would make predictions closer and closer to the ground truth label, and therefore learns.

### <ins> How to Reduce Loss?

TL;DR: ***Gradient descent*** (among other ways)

You can find out all the math about gradient descent [here](https://en.wikipedia.org/wiki/Gradient_descent), but intuitively the ***gradient always point in the direction of the sharpest increase*** in the loss function. And the gradient descent algorithm takes a step in the direction of the ***negative gradient*** to reduce the loss.

Gradient descent algorithm often multiply the gradient by a scalar call ***learning rate***, which controls how fast the model learns.

### <ins> Learning Rate?

Finding a great learning rate is data dependant, and requires experiments. Too small of learning rate, and it would take the model forever to minimize the loss and learn well; too big of a learning rate, the model might never learns, because it might overshoot the minimum and never converges. In practice, finding a "perfect" (or near-perfect) learning rate is not essential for successful model training. The goal is to ***find a learning rate large enough that gradient descent converges efficiently, but not so large that it never converges***.

Again, it requires experiments and all machine learning frameworks have default values that are great starting point.

### <ins> Gradient Descent vs. Stochastic Gradient Descent?

In gradient descent, we use all examples to calculate the gradient in a single iteration. In practice, however, we might have too big of a dataset (e.g., billions of rows) that won't fit into the memory, or a single iteration might take too long. That's when stochastic gradient descent comes into play.


Both gradient descent (GD) and stochastic gradient descent (SGD) update parameters iteratively to minimize loss function, but ***SGD takes things to the extreme and uses only a single example per iteration***. This can be too dramatic though so people often adapt to a compromise between GD (full-batch iteration) and SGD, called ***mini-batch SGD***, where a batch can contain 10 to 1000 examples (or more).  

In practice mini-batch SGD is used most often and it usually helps the model converge faster compared to the other options.

### <ins> Generalization?

Say you've used SGD and trained a machine learning model, and the model scored perfectly (or near perfectly) on the training set. Should you call it a day and deploy it into production? Probably not.

First of all you should celebrate the fact that the model learns perfectly. That means your model is capable enough to learn well, and in fact in many projects (especially deep learning projects) one goal is to first have a model that learns perfectly on the training data, or as people would call it, to ***overfit a small training subset***.

But in real world we prefer the model to do well on the training data ***and to make accurate predictions on previously unseen data***, otherwise we can't say we have a great model because the model might mechanically memorize what the training data looks like and wouldn't be able to perform well on new data. It's the ability to make accurate prediction on unseen data that counts, and we call this ability the ***ability to generalize***.


### <ins> Overfitting vs. Underfitting, Bias-Variance Tradeoff?

Overfitting means that the machine learning model is complex (and maybe too complex) such that it even overfits the peculiarities of the data it trained on, while underfitting means the model lacks the complexity to learn well even on the training data. Ideally we want the ***model to be complex enough, but not so complex that it doesn't generalize***.


A related topic in machine learning: bias-variance tradeoff. Skipping all the math, bias means the error from erroneous assumptions of the model, while variance is the error from being overly sensitive to the peculiarities in the training data. In simplified terms, you can think like this ***high bias = too simple a model = underfitting***, and ***high variance = too complex a model = overfitting***.

We want our model to strike a good balance between bias and variance because if we reduce model complexity, the bias error might increase and if we increase its complexity, the variance error might go up. An optimal balance leads to a model that neither overfits nor underfits.


### <ins> Data Splitting: Training, Validation, Test Sets?

First of all, ***the purpose of each set***:

- Training set: to ***train*** the machine learning models
- Validation set: to ***tweak the model and tune its parameters***, and to select the best model
- Test set: ***one final sanity check*** on the performance of the model, ***NOT*** to be used to select model or tune parameters

One follow up would be: how do we split a data set into these 3 sets?

In term of ***data size***, some popular methods include a split of 70%/15%/15%, or 60%/20%/20% etc. for training/validation/test sets. I wouldn't call this strategy wrong but ***I'd rather think in the following fashion***:

Both ***validation and test sets should be large enough to yield statistically meaningful results*** and statistically meaningful could mean differently across projects. Minus the validation and test sets, we'd like to allocate ***as much data to the training set as possible***.

What's large enough though? It's really problem dependant but if you care about the 0.1% score difference between models compared to that at 1% level, you'd probably increase the size of validation/test sets by an order of magnitude. A good rule of thumb given by Andrew Ng was:

>With 10,000 examples, you will have a good chance of detecting an improvement of 0.1%


Also make sure to consider the following when you split the data:

- Are training/validation/test sets ***representative*** of the whole dataset? In other words, are they from the same distribution?
- Are there any examples in validation/test set ***duplicates*** of examples in the training set?
- For time series data, is the dataset split in a ***chronological order***?
- Test sets and validation sets "wear out" with repeated use. If possible, it's a good idea to collect more data to ***"refresh"*** the test set and validation set.


### <ins> Feature Representation?

Features usually is either one of the following two types:

- ***Numeric*** - number of room: 3, stock price: 276.89, etc.
- ***Categorical*** - street name: 'Toronto St.'

And many machine learning algorithms only work with numerical data, so we need to represent the categorical features as numeric ones.

Say we have a feature called `Country` with the following options:

`{USA, Canada, Mexico}`

While we can represent each of the value as a number (by mapping `USA` to 1, `Canada` to 2, etc.), does it mean that Canada is "better" than USA or what? What if the feature represents a person's travelled countries: how do we represent that?

This is where ***one-hot encoding*** (or multi-hot encoding, in case an example takes multiple values) comes to play.

One-hot encoding creates a binary vector of length `N`, where `N` is the number of unique values in a feature, and an example would have a value of `1` if it has this value and `0`s everywhere else (E.g., for the `country` feature, we'd create three new features `[country_usa, country_canada, country_mexico]` and a person from Canada would have a feature vector `[0, 1, 0]`).

One-hot encoding also extends to numeric data that you do not want to directly multiply by a weight, such as a postal code.

### <ins> What's a Good Feature?

Conceptually, features that help the machine learning models learn more effectively are good features, and they often present the following qualities:

- feature values appear more than N times in the feature - N is data dependant but features that only appear once and are unique to one example (such as SSN, ID etc.) are rarely a good feature
- have clear and obvious meaning (e.g., `age`: 27, `time_visited_during_past_24_hr`: 3)
- Doesn't have mixed "magic" values with actually data. If we have a feature called rating (between 1-5) and a user doesn't have a rating, instead of imputing the value with -1, a better way is to create a new feature called `has_rating` and assign boolean values accordingly.
- has a strong negative or positive correlation with the label


### <ins> Data Cleaning Techniques?

This is by no means an exhaustive list, and continues to expand:

- ***Scaling*** scale feature values from their natural range (e.g., 1 to 9999999) to a standard range (0 to 1 or -1 to +1). This helps model converge faster and learn proper weights
- ***Binning***: convert numerical into categorical, and then do one-hot encoding (e.g., zip code)
- ***Outliers***: take the log, and/or cap it at a threshold
- ***Missing values***: Impute with median, mean; Impute value from neighbours; train a classifier to impute values; create a boolean feature to indicate if the current feature has missing values
- ***Duplicate***: remove them; also if the number of duplicate is not trivial, look into the data collection pipeline and identify issues
- ***Bad feature/label***: (e.g., age=240) usually it makes sense to remove them but if they appear often also make sure to look into the data collection pipeline and fix potential issues


### <ins> Nonlinearity and Feature Crosses?

Some data are not linearly separable so we need to encode nonlinearity. This can be achieved on the algorithm level (e.g., by adding non-linear activation function such as [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), or at data level, where we perform feature crosses.

***Feature cross encodes nonlinearity in the feature space by multiplying two or more input features together***. Say we have a model `y = w1x1 + w2x2 + b`, we can create a new feature `x3` (where `x3=x1*x2`), and change the model to `y = w1x1 + w2x2 + w3x3 + b`. Now the model has 4 learnable parameters instead of 3.

In practice, machine learning models seldom cross continuous features. However, machine learning models do ***frequently cross one-hot feature vectors***. Think of feature crosses of one-hot feature vectors as ***logical conjunctions***.


### <ins> Model Complexity?

Ways to think about model complexity:

- Model complexity as ***a function of the weights*** of all the features in the model - the higher the absolute value of the weights, the more complex a model is
- Model complexity as ***a function of the total number of features*** with nonzero weights - the more features there are, the model complex a model will be

### <ins>Regularization, L1 & L2, Lasso & Ridge?

Regularization is a technique to ***combat overfitting and make the model generalize better***.

Whereas model without regularization minimize loss, model with regularization minimizes loss + complexity: `minimize(loss(data|model) + complexity)`. Regularization is a great way to reduce model complexity because it penalizes models that have weights with high absolute values.

A common regularization technique is L<sub>2</sub> regularization, which defines the regularization term as the ***sum of the squares of all the feature weights***:

L<sub>2</sub> regularization term: ||w||<sup>2</sup> = w<sub>1</sub><sup>2</sup> + w<sub>2</sub><sup>2</sup> + ... + w<sub>n</sub><sup>2</sup>

A common alternative is L1 regularization, which is the ***sum of the absolute values** of all feature weights:

L<sub>1</sub> regularization term: ||w|| = ||w<sub>1</sub>|| + ||w<sub>2</sub>|| + ... + ||w<sub>n</sub>||

And a regression ***model with L<sub>1</sub> regularization is called Lasso, and a model with L<sub>2</sub> regularization is called Ridge***.

Another interesting property: while L<sub>2</sub> regularization encourages weights to be small, it doesn't force them to exactly 0.0; ***L<sub>1</sub> can push weights to zero, thus in a way performing feature selection.*** This can come in handy when you have hi dimensional data


Also, in practice we would multiply the regularization term by a scalar called ***lambda*** (hence `minimize(loss(data|model) +  lambda * complexity`) to control the degree of penalization. Too high a lambda value and the model might become too simple and runs the risk of underfitting, while too low a lambda value the model will becoming too complex and might overfit. When lambda is zero, the model essentially doesn't have any regularization at all.


Another regularization technique is called dropout, which is almost exclusively used in neural network training. Dropout randomly selects neurons to ignore during training to reduce the complexity of neural network models. More on dropout later.


### <ins> Why Accuracy Can Be Misleading?

In classification problems, one commonly seen metric is called accuracy, which measures the percentage of correct prediction over all predictions. Sounds like a good metric, right?

Well, that's not the whole picture. Accuracy can actually be a poor or misleading metric, because ***different prediction mistakes can have different consequence, and accuracy is not robust to dataset with imbalanced labels***. For instance, say we have a 100 patients, out of whom 2 have cancer. Is a model that predicts all 100 patients as cancer free (and thus has a high accuracy of 98%) a good model?

If we use accuracy as the metric, we might be misled to believe so.


### <ins> True Positive, True Negative, False Positive, False Negative?

These terms could be intimidating for every new comers. Think of them this way

- ***True*** means the model *correctly predicts* the label, while
- ***False*** means the model *incorrectly predicts* the label

What do positive and negative mean then? Well they could mean different things in different use cases but usually positive signifies the presence of something. It could mean spam (so negative means non-spam), or raining (whereas negative means not raining).

In the case of email spam classification, combining true/false with positive/negative we have:

- ***True Positive***: the model *correctly* predicts a spam as spam
- ***True Negative***: the model *correctly* predicts a non-spam as non-spam
- ***False Positive***: the model *incorrectly* predicts a non-spam as spam (worst outcome)
- ***False Negative***: the model *incorrectly* predicts a spam as non-spam (bad)

It will take some time for the concepts to sink in. Just remember: ***True means the model is doing a job, and False bad job***; positive and negative are just codes that are mapped to labels

In practice people often use a table called [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) to summarize the above four metrics.


### <ins> Precision vs. Recall, and F1 Score?

Previously we talked about that accuracy can be misleading, and also introduced true positive (TP), true negative (TN), false positive (FP), and false negative FN). A naturally next step is two metrics called precision and recall.

Precision is defined as `Precision = TP / (TP + FP)`, and recall is defined as `Recall = TP / (TP + FN)`.

Intuitively, precision is the ratio of correct positive predictions to the total ***predicted positives***, while recall is the ratio of correct positive predictions to the ***total actual positives***.

Using the email spam example, precision means out of all the emails that ***the model thinks*** as spam, what percentages are spam, and recall means ***out of all the actual spam***, what percentage the model correctly identifies them? Try to let the ideas sink in...

Previously we talked about classification threshold, and changing the threshold would affect both precision and recall. In general, ***raising the classification threshold*** will

- reduce false positives, thus ***raising precision***
- cause the number of true positives to decrease or stay the same and will cause the number of false negatives to increase or stay the same. Therefore, ***recall will either stay constant or decrease***

In an ideal world both precision and recall scores are perfect at 1.0, but in practice there is often a tension between the two. ***Improving precision typically reduces recall and vice versa***. Since you can't get both, you have to decide, and deciding whether to use precision or recall should always be ***tied to business decisions to avoid what's more costly***.

Also, various metrics have been developed that rely on both precision and recall. For example, [F1 score (or F-score)](https://en.wikipedia.org/wiki/F-score) combines the precision and recall of a classifier into a single metric by taking their harmonic mean:

`F1 score = 2 * (precision * recall) / (precision + recall)`

The highest possible value of an F1 score is 1, indicating perfect precision and recall, and the lowest possible value is 0, if either the precision or the recall is zero.


### <ins> ROC Curve and AUC?

An [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (receiver operating characteristic curve) is a graph showing the performance of a binary classification model *at all classification thresholds*. This curve plots two parameters:

- True Positive Rate: `PTR = TP / (TP + FN)`
- False Positive Rate: `FPR = FP / (FP + TN)`

***AUC*** stands for "Area under the ROC Curve", which ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0. One way of interpreting AUC is as the ***probability that the model ranks a random positive example more highly than a random negative example***.

AUC can be desirable for the two reasons:

- AUC is ***scale-invariant***. It measures how well predictions are ranked, rather than their absolute values.
- AUC is ***classification-threshold-invariant***. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.



### <ins> Prediction Bias?

For a classification problem, ***prediction bias*** is a quantity that measures how far apart *average of predictions* and *average of observations* are. Don't confuse prediction bias with the bias term in `y=w*x + b`

For example, let's say we know that on average, 1% of all emails are spam. If we don't know anything at all about a given email, we should predict that it's 1% likely to be spam. Similarly, a good spam model should predict on average that emails are 1% likely to be spam. If instead, the model's average prediction is 20% likelihood of being spam, we can conclude that it exhibits prediction bias.

Possible root causes of prediction bias are:

- Incomplete feature set
- Noisy data set
- Buggy pipeline
- Biased training sample
- Overly strong regularization

***A good model will usually have near-zero bias***. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias. For example, a model that just predicts the mean value for all examples would be a bad model, despite having zero bias.

So our goal is not to minimize the prediction bias. We still need to minimize the loss function with respect to a specific metric we chose; examining the prediction bias is a sanity check.


### <ins> Backpropagation?

Backpropagation is the most common training algorithm for neural networks. Intuitively, what it does is to compute the gradient of the loss function with respect to each weight, one layer at a time, and propagate backward error layer by layer.

All popular deep learning frameworks (such as TensorFlow or PyTorch) handle backpropagation automatically so you don't have to, but it's still a good idea to understand what it does under the hood or even implement it on a small neuron network to really help the concept sink in.

If visuals help you better understand a concept, take a look at this [backpropagation algorithm visual explanation](https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll).

In practice, there are also a number of ways for backpropagation to go wrong. For example, ***gradients can vanish*** (when you take the product of many small terms), ***explode*** (when you multiple big terms), or you might even have ***dead units***.

Common curves include add activation function such as ReLU (for vanishing gradients), and lower the learning rate.


### <ins> Embedding?

In machine learning and natural language processing (NLP), an ***embedding*** is a low dimensional vector to represent high dimensional or unstructured data (text, images, video, etc.). An embedding, ideally, captures some of the similarities of the input by placing similar inputs close together in the embedding space. For instance, comedy movies would be will stay close in one corner of the embedding space and animations at another. This could come in handy for application such as recommendation systems.

One popular embedding example is [***word embedding***](https://en.wikipedia.org/wiki/Word_embedding), where each word is represented with a n-dimensional vector, which encodes a word's semantic meaning and words with similar meanings are closer together in the embedding space. Embeddings are lookup tables, and the closeness (or distance) between two embeddings can be computed with their dot product.

As embedding can be thought of a way to reduce dimensionality, techniques such as principal component analysis (PCA) has been used to create word embeddings. Given a set of instances like bag of words vectors, PCA tries to find highly correlated dimensions that can be collapsed into a single dimension.

Another way to obtain embeddings is to train neural networks to learn them. For example, [BERT](https://arxiv.org/abs/1810.04805), published by Google, is way to obtain pre-trained language model word embedding from text data.

How to choose embedding's dimension? One rule of thumb to is use the 4th root of the number of categories. If you have a vocabulary size of 100, a 3-D (`3 = 100**0.25`) embedding would probably do.


### <ins> Principal Component Analysis (PCA)?

Principal component analysis (PCA) is a ***dimensionality reduction*** technique that can help ***increase interpretability of data while preserving the maximum amount of information***.

PCA finds the top N principal components, along which the data vary (spread out) the most. Intuitively, the more spread out the data along a specific dimension, the more information is contained, and therefore the more important that dimension is for the pattern recognition of the dataset.

How do we choose N (number of principal component) in PCA? One way is to set N to number of total features first, and get the ***cumulative sum of the [`explained_variance_ratio_`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)***, and make a tradeoff between the ratio and number of dimensions on a case by case basis.

Keep in mind PCA will create new value vectors rather than taking the feature values from the original dataset. Also,  applying normalization to the dataset prior to PCA is important since PCA can be thought of a variance maximizing exercise.


### <ins> Static vs. Dynamic Training?

- A ***static model*** is trained offline. That is, we train the model exactly once and then use that trained model for a while.
- A ***dynamic model*** is trained online. That is, data is continually entering the system and we're incorporating that data into the model through continuous updates.

Static models are easier to build and test, while dynamic models adapt to changing data.

### <ins> Offline vs. Online Inference?

- ***Offline inference***: making all predictions in a batch, e.g., done by nightly cron job
- ***Online inference***: making prediction on demand and in real time

Usually online inference is more latency sensitive so we might need to limit the model complexity. Also online inference may require more intensive monitoring.
