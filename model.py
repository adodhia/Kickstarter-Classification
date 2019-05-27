
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLS_TO_DROP = [
    'id', 'photo', 'slug', 'disable_communication', 'friends', 'is_starred',
    'is_backing', 'permissions', 'currency_symbol', 'currency_trailing_code',
    'currency', 'creator', 'location', 'urls', 'source_url', 'category',
    'profile'
]

LOGREG_PARAMS = {'penalty': 'l0', 'C': 0.0009}

class KickstarterModel:
    
    def __init__(self):

        self.model = LogisticRegression(**LOGREG_PARAMS)
        self.ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

    def preprocess_common(self, df):
        """Method used by both preprocess_training_data 
        and preprocess_unseen_data"""

        # Copy dataframe
        x = df.copy()

        # Replacing nans strings by empty strings
        x.name = x.name.fillna('')
        x.blurb = x.blurb.fillna('')

        # dropping useless coumns
        x.drop(COLS_TO_DROP, axis=1, inplace=True)

        # converting goal to USD
        x.goal = x.goal / x.static_usd_rate
        x.drop('static_usd_rate', axis=1, inplace=True)

        # Duration metrics
        x['duration_creation'] = x.launched_at - x.created_at
        x['duration_funding'] = x.deadline - x.launched_at

        x.drop(['created_at', 'launched_at'], axis=1, inplace=True)

        # Length of text features
        x['len_name'] = x.name.str.len()
        x['len_desc'] = x.blurb.str.len()
        x['name_word_count'] = x.name.str.split().str.len()
        x['desc_word_count'] = x.blurb.str.split().str.len()
        x['desc_avg_word_len'] = x.blurb.apply(lambda s: np.mean(
            [len(w) for w in s.split(' ')]))
        x.drop(['name', 'blurb'], axis=1, inplace=True)

        # Logs / exps
        x['len_desc_log'] = np.log1p(x['len_desc'])
        x['desc_avg_word_len'] = np.log1p(x['desc_avg_word_len'])

        # Categorical columns:
        x_cat = pd.DataFrame()
        x_cat['country'] = x.country
        x_cat['month'] = pd.to_datetime(x.deadline, unit='s').dt.month
        x.drop(['country'], axis=1, inplace=True)

        return x, x_cat

    def preprocess_training_data(self, df):

        y = df.state.apply(lambda txt: 0 if txt == 'failed' else 1)
        df.drop('state', axis=1, inplace=True)

        x, x_cat = self.preprocess_common(df)

        # One-Hot Encoding
        # Here we use fit_transform as we want to train
        # our one hot encoder
        x_cat = pd.DataFrame(self.ohe.fit_transform(x_cat))
        x = x.join(x_cat)

        x = pd.DataFrame(self.scaler.fit_transform(x),
                         columns=x.columns,
                         index=x.index)

        return x, y

    def fit(self, X, y):

        raise NotImplementedError

    def preprocess_unseen_data(self, df):

        raise NotImplementedError

    def predict(self, X):

        raise NotImplementedError
