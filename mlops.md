# Machine Learning Operationalization (MLOps)

Training machine learning models and getting great performance is worth celebrating. Putting it into production, however, is another story.

![ML System in real world](assets/MLSystem.svg)

### <ins> Setting up an End-To-End ML System in the Real World

These are the common steps we need to follow in order to set up an ML system in the real world:

1. ***Scoping and business problem understanding***. ML is cool but it needs to bring values to customers and align with an organization's business goals. Being able to understand the business problem and tranlating the business problem into an ML problem is a key first step. Also, ML is not a cure-all but instead is only one approach to solving any problem.
2. ***Metrics***. For any ML project setup it's usually helpful to have a **single number metric** to guide the decision process. In an *offline* setting, we can choose one metric (F1 score, NDCG, etc.) to select the best ML model, while in an *online* setting, we may need to adapt a different metric where the offline metric is only part of the equation - the goal is to make sure the product improves as a whole (e.g., highest user engagement rates). And this is also a good time to **establish a baseline** for the ML system.
3. ***Techinical requirements understanding***. Say ML is the approach we want to take, the next step is to understand the techincal requirements and other parts of the whole prodcut/system. What's the scale of the data? Do we need real-time inference or is batch workload fine? For real-time inferences, what is the latency requirment? What's the throughput? These are just a couple of example questions and answers to these questions would lead us to design the architecture of the ML system.
4. ***Architecture design***. After figuring out all the techincal requirements and details, we need to identify key components and design the arachitecture of the ML system, as well as how it interfaces with other systems.
5. ***Training data acquisition/generation***. Following the previous steps, we need to ask these questions: How do we get training data? Do we need to hire people or use a specific tool to label the data, or can we users' implict feedback as labels? No matter how good a learning algorithm is, if we don't have high quality training data, the system is doom to fail.
6. ***Feature Engineering***. This is another crucial step as good features influence the model’s ability and boost its performance. A great business understanding of the data + creativity + good engneering would help us create good features for a given ML task
7. ***Offline model training and selection***. Are we training models from scracth, or are there state of the art pre-training results we can leverage? What algorithms to use and what hyperparameters to choose (model training and parameter tuning can be fun initially but the task can become mechanical after a while; that's when tools such as [Hyperopt](https://hyperopt.github.io/hyperopt/) comes into play)?
8. ***Online testing***. After training and getting a great model offline, it's time to test it in the real world. Initially we should apply a canary deployment, where we roll out the model to only a small fraction (3-5%) of the traffic. We should monitor the performace and ramp up only when we observe better overall performace based on the online metric we definted earlier.
9. ***MLOps***. While MLOps is listed lastly, it's by no means the least important. While we may afford to do things manually at first just to see if ML would even work, to put ML into production in a similar fashion as we would for software we need to automate the process. We can leverage tools such as [MLflow](https://www.mlflow.org/docs/latest/index.html) to help manage the end-to-end machine learning lifecycle.

A couple of notes:

- While I listed all the previous steps, it doesn't mean you need to consider all of the them if you are just doing a quick and dirty prototype to check the feasibility of an ML system. Use your judgement call to skip a couple of steps here and there
-  ML is highly iterative in its nature. While some steps, such as business scoping and metrics, change less frequently, feature engineering and model training for examples can iterate very fast, which is a good thing because that's how your model improve. So don't assume the previous steps would follow a linear fashion.


### <ins> Ways to Deploy ML Models

There are different directions to think about how to deploy ML models in production:

- Do we want to deploy machine learning models as a **real-time** prediction service (email auto complete) or in a **batch** prediction mode (Netflix movie recommendation updates every N hours for a user)?
- Do we want to deploy models **on device** (some factory production lines) or as a web service?

There are different factors to consider when we choose one of the previous directions:

- **Latency**: How quickly does an application/user require the results of the model prediction?
- **Throughput**: how many query per second (QPS) are we expecting?
- **Data privacy**: Are there issues/concerns about sending data to the cloud?
- **Network connectivity**: is there limited internet/network connectivity, and is the model required to make prediction even without internet?
- **Cost**: some deployment options would be more costly then others. What's the cost expectation?

Let's take deploying models as an independent service (most commonly REST APIs) as an example and see some options:

- **On-prem vs. cloud**: you can provision virtual machines locally or on the cloud to accept REST APIs calls to your models
- **Virutal machines (VM) vs. containers**: compared to provisioning VM, you can also use lightweight service such as dock containers (which is recommended in most cases) to host your models. There are even open source solution specifically built to package ML models in containers, such as Cog, BentoML, and Truss
- **CPU vs. GPU**: you may wonder this especially for deep learning models (neural networks). But just because they are trained on GPU doesn't mean they have to be served on GPU as well. In fact, you can [serve 1 billion+ daily requests using neural network models on CPU only](https://www.youtube.com/watch?v=Nw77sEAn_Js)
- **Serverless**: Yet another way to deploy ML models is to do it in a serverless fashion so you don't even have to worry about infrastructure. It has its con though: if you don't have a stable traffic, it sometimes would take a while for the service to re-start, causing latency for real-time prediction use cases

### <ins> Ways to Speed Up Model Inference

Other things being equal, you almost always want your model inference to take less time. It will involve a tradeoff between model performance (metrics such as accuray, F1 score etc.) so it's a judgement call. If you ever want to speed up model inference time in production, here are some ways:

- **Model distillation**: once you have a large model that you've trained, you can train a smaller model that imitates the behavior of your larger one. It basically creates a much smaller model with a tolerable performace decrease - e.g., inference time reduced to 1/3 with a accuray decrease of only 0.5%. An example is DistilBERT to BERT
- **Quantization**: you execute some or potentially all of the operations in your model in a lower fidelity representation, e.g., use 8-bit integer or 16-bit floating point numbers
- **Caching**: for some of your ML models, you realize some inputs are more common than others. Instead of always calling the model every time a user makes a request, let's store the common requests in a cache
- **Horizontal scaling**: a more brute force way is to provision more containers/VMs/cores to share the workload of model inference


### <ins> Data Collection

When building a model from a user's behavior, a distinction is often made between explicit and implicit forms of data collection.

Examples of **explicit data collection** include the following:
- Asking a user to rate an item on a sliding scale.
- Asking a user to search.
- Asking a user to rank a collection of items from favorite to least favorite.
- Presenting two items to a user and asking him/her to choose the better one of them.
- Asking a user to create a list of items that he/she likes

Examples of **implicit data collection** include the following:

- Observing the items that a user views in an online store.
- Analyzing item/user viewing times.
- Keeping a record of the items that a user purchases online.
- Obtaining a list of items that a user has listened to or watched on his/her computer.
- Analyzing the user's social network and discovering similar likes and dislikes.

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
