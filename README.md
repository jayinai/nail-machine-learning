# Nailing Machine Learning Interviews: Concepts

This repo is specifically designed to help you prepare for the the concept part of machine learning interviews.

Definitely spend some time going through the list and make sure you understand and explain all of them well :)
 

### Supervised vs. Unsupervised Learning?

In supervised learning the machine learns a ***function that maps between X (all the features, or input) and Y (label, or output)***, so that it can make prediction on new data - given unseen X, predict its Y. For example, given the subject line, text body, and senders' email address, we can train machines to classify whether the a new email is a spam or not.

On the other hand, in unsupervised learning there is ***no label given, so machines have to learn and discover patterns from data itself (without human supervision)***. One example is that given user related data, such as their location, browsing history, and logged times (during the past 12 hrs, 24 hrs, 3 days, 1 week, etc.), the machine would learn to cluster users into different groups (which can be useful for target marketing, for instance).

So in a nutshell - supervised: supervise/teach machines to make prediction; unsupervised: "dear machine, I trust you, discover some hidden patterns for me on your own"



### Regression vs. Classification?

Both are supervised learning problems. The difference is that in regression we are trying to predict ***continuous values*** (e.g., house prices), while in classification problem we try to make predictions on ***discrete values***: e.g., is the email spam or not (2 discrete values), is the image of a dog, cat, or fish (3 discrete values)?



### How Does a Model Learn?

Let's say we have a linear model `y = w1x1 + b`, where `y` represents the house price (label), and `x1` is the number of bedroom (feature). `w1` and `b` are learnable parameters, weight and bias. How does the model learn?

We can initialize the learnable parameters with some random values, say, 0s. Then we need to ***train*** the model by asking it to look at different examples, use the current parameters to make predictions `y'`, and compare the difference between `y` (ground truth) and `y'` (prediction). Ideally we want the difference to be as small as possible, so we need to figure out a way to minimize the difference (or ***loss***).

By ***iteratively reducing the loss*** (and updating the learnable parameters), the model would make predictions closer and closer to the ground truth label, and therefore learns. 

### How to Reduce Loss?

TL;DR: ***Gradient descent*** (among other ways)
 
You can find out all the math about gradient descent [here](https://en.wikipedia.org/wiki/Gradient_descent), but intuitively the ***gradient always point in the direction of the sharpest increase*** in the loss function. And the gradient descent algorithm takes a step in the direction of the ***negative gradient*** to reduce the loss. 

Gradient descent algorithm often multiply the gradient by a scalar call ***learning rate***, which controls how fast the model learns. 

### Learning Rate

Finding a great learning rate is data dependant, and requires experiments. Too small of learning rate, and it would take the model forever to minimize the loss and learn well; too big of a learning rate, the model might never learns, because it might overshoot the minimum and never converges. In practice, finding a "perfect" (or near-perfect) learning rate is not essential for successful model training. The goal is to ***find a learning rate large enough that gradient descent converges efficiently, but not so large that it never converges***.

Again, it requires experiments and all machine learning frameworks have default values that are great starting point.

### Gradient Descent vs. Stochastic Gradient Descent

In gradient descent, we use all examples to calculate the gradient in a single iteration. In practice, however, we might have too big of a dataset (e.g., billions of rows) that won't fit into the memory, or a single iteration might take too long. That's when stochastic gradient descent comes into play.


Both gradient descent (GD) and stochastic gradient descent (SGD) update parameters iteratively to minimize loss function, but ***SGD takes things to the extreme and uses only a single example per iteration***. This can be too dramatic though so people often adapt to a compromise between GD (full-batch iteration) and SGD, called ***mini-batch SGD***, where a batch can contain 10 to 1000 examples (or more).  


In practice mini-batch SGD is used most often and it usually helps the model converge faster compared to the other options.


