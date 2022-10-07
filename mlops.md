# Machine Learning Operations


### <ins> Static vs. Dynamic Training?

- A ***static model*** is trained offline. That is, we train the model exactly once and then use that trained model for a while.
- A ***dynamic model*** is trained online. That is, data is continually entering the system and we're incorporating that data into the model through continuous updates.

Static models are easier to build and test, while dynamic models adapt to changing data.

### <ins> Offline vs. Online Inference?

- ***Offline inference***: making all predictions in a batch, e.g., done by nightly cron job
- ***Online inference***: making prediction on demand and in real time

Usually online inference is more latency sensitive so we might need to limit the model complexity. Also online inference may require more intensive monitoring.
