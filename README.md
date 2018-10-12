# Kickstarter

Kickstarter is a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing creative project to life.

Until now, more than $3 billion dollars have been contributed by the members in fueling creative projects.
The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis i.e if a project doesn’t meet its goal, the project owner gets nothing.
For example: if a projects’s goal is $5000. Even if it gets funded till $4999, the project won’t be a success.

If you have a project that you would like to post on Kickstarter now, can you predict whether it will be successfully funded or not? Looking into the dataset, what useful information can you extract from it, which variables are informative for your prediction and can you interpret the model?

The goal of this project is to build a classifier to predict whether a project will be successfully funded, you can use the algorithm of your choice.

**Note**:
* The target, `state` can take several values (`'canceled'`,`'live'`,`'suspended'`,`'failed'`). Here we want to convert it to a binary outcome: `1` if `successful` or `0` for all other outcomes.
* The variables `'deadline', 'state_changed_at', 'created_at', 'launched_at'` are stored in Unix time format.

## Get the data

We provided a file, `get_data.py` that you can use to download the data in the right place; in a terminal run:
```
python get_data.py
```
from within the repository. It may take a few minutes as it needs to download the dataset.

Alternatively, you can download the data manually and place it in a `data/` folder:
* Download the dataset from [here](https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter/train.zip).
* Create a new directory called `data/`
* Place it in the `data/` folder.

Since the data is quite big, it will not be tracked by the system (see the `.gitignore` file (but don't change it)).


### Start working

* It is recommended to start a new notebook in which you can load the dataset, start implementing your processing steps, your model and properly evaluate it.
* Once you're happy with the performance of your model, you can implement it as a Python class in the `model.py` file. Check the help page on K.A.T.E. for more details on how to do so.
* Run `python pickler.py` to train your new model and generate a `pickle` file that you can submit to K.A.T.E.
* Upload your `model.py`, your notebook and your `model.pickle` to K.A.T.E.

* Follow the steps described in the `ads-wiki` on how to submit a model.
* You will need to implement the class `KickModel` in `model.py` and run the `pickler.py` to train your model and save its state to a file.
* To run the pickler, use `python pickler.py`
* You need to upload both your code and the generated `model.pickle`

Your code should look something like:

File `model.py`:

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


class SimpleModel:

    def __init__(self):
        self.model = None

    def preprocess_training_data(self, path_to_csv):

        data = pd.read_csv(path_to_csv)

        y = data[data.columns[-1]]
        X = data.drop(data.columns[-1], axis=1)

        self.pca = PCA(n_components=2)
        X = self.pca.fit_transform(X)

        return X, y

    def fit(self, X, y):
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X, y)

        self.model = knn

    def preprocess_unseen_data(self, path_to_csv):

        data = pd.read_csv(path_to_csv)
        return self.pca.transform(data)

    def predict(self, X):

        return self.model.predict(X)
```