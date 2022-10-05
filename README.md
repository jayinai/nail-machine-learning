# Nailing Machine Learning Interviews: Concepts

This repo is designed to help you ***prepare for the concept part of machine learning interviews***. If you have machine learning or data science related interviews coming up, then you should definitely spend some time going through the list and make sure you understand all the concepts and can explain them well. If you can, your dream job won't just be a dream anymore :)

That said, you can of course use this as a casual learning resource to get familiar with important machine learning concepts. Hope this would be helpful to you.


### <ins> Supervised vs. Unsupervised Learning?

In supervised learning the machine learns a ***function that maps between X (all the features, or input) and Y (label, or output)***, so that it can make prediction on new data - given unseen X, predict its Y. For example, given the subject line, text body, and senders' email address, we can train machines to classify whether the a new email is a spam or not.

On the other hand, in unsupervised learning there is ***no label given, so machines have to learn and discover patterns from data itself (without human supervision)***. One example is that given user related data, such as their location, browsing history, and logged times (during the past 12 hrs, 24 hrs, 3 days, 1 week, etc.), the machine would learn to cluster users into different groups (which can be useful for target marketing, for instance).

So in a nutshell - supervised: supervise/teach machines to make prediction; unsupervised: "dear machine, I trust you, discover some hidden patterns for me on your own"



### <ins> Regression vs. Classification?

Both are supervised learning problems. The difference is that in regression we are trying to predict ***continuous values*** (e.g., house prices), while in classification problem we try to make predictions on ***discrete values***: e.g., is the email spam or not (2 discrete values), is the image of a dog, cat, or fish (3 discrete values)?



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

Both ***validation and test sets should be large enough to yield statistically meaningful results*** and statistically meaningful could mean differently across projects. Lastly, minus the validation and test sets, we'd like to allocate ***as much data to the training set as possible***.

What's large enough though? It's really problem dependant but if you care about the 0.1% difference between models compared to that at 1% level, you'd probably increase the validation/test set by an order of magnitude. A rule of thumb given by Andrew Ng was 

>With 10,000 examples, you will have a good chance of detecting an improvement of 0.1%


Also make sure to consider the following when you split the data:

- Are training/validation/test sets ***representative*** of the whole dataset? In other words, are they from the same distribution?
- Are there any examples in validation/test set ***duplicates*** of examples in the training set?
- For time series data, is the dataset split in a ***chronological order***?
- Test sets and validation sets "wear out" with repeated use. If possible, it's a good idea to collect more data to ***"refresh"*** the test set and validation set.

