# Machine Learning Operationalization (MLOps)


### <ins> Static vs. Dynamic Training

- A ***static model*** is trained offline. That is, we train the model exactly once and then use that trained model for a while.
- A ***dynamic model*** is trained online. That is, data is continually entering the system and we're incorporating that data into the model through continuous updates.

Static models are easier to build and test, while dynamic models adapt to changing data.

### <ins> Offline vs. Online Inference

- ***Offline inference***: making all predictions in a batch, e.g., done by nightly cron job
- ***Online inference***: making prediction on demand and in real time

Usually online inference is more latency sensitive so we might need to limit the model complexity. Also online inference may require more intensive monitoring.

### <ins> Data Dependencies

The behavior of an ML system is dependent on the behavior and ***qualities of its input features***: As the input data for those features changes, so will your model.

In traditional software development, you focus more on code than on data. In machine learning development, although coding is still part of the job, your focus must widen to include data: you must ***continuously test, verify, and monitor your input data***.

Pay attention to the follow aspects:
- ***Reliability***: Is the dependant data always going to be available or is it coming from an unreliable source?
- ***Versioning***: Does the system that computes this data ever change, and if so how often?
- ***Necessity***: Does the usefulness of the data/feature justify the cost of including it?
- ***Correlations***: Are any features so tied together that you need additional strategies to tease them apart??
- ***Feedback Loops***: Sometimes a model can affect its own training data. For example, the results from some models, in turn, are directly or indirectly input features to that same model or other models. So are we reinforcing bias?

Feedback loops can be tricky to detect and may 'break the whole system'. For example, we may train a book-recommendation model that suggests novels its users may like based on their popularity (i.e., the number of times the books have been purchased). It, however, will only make the popular ones more popular and not so popular ones even less popular, in a way reinforcing biases.


### <ins> Human Bias

Machine learning models are not objective. It was trained by humans, and human involvement can a model's predictions susceptible to bias.
- Common [types of bias](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias)
- How to [identify bias](https://developers.google.com/machine-learning/crash-course/fairness/identifying-bias)
- How to [evaluate for bias](https://developers.google.com/machine-learning/crash-course/fairness/evaluating-for-bias)


### <ins> MLflow

[MLflow](https://www.mlflow.org/docs/latest/index.html) is a library-agnostic platform for managing the end-to-end machine learning lifecycle.

It tackles four primary functions:

- ***Tracking experiments*** to record and compare parameters and results (MLflow Tracking).
- ***Packaging ML code in a reusable, reproducible form*** in order to share with other data scientists or transfer to production (MLflow Projects).
- ***Managing and deploying models*** from a variety of ML libraries to a variety of model serving and inference platforms (MLflow Models).
- ***Providing a central model store*** to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations (MLflow Model Registry).
