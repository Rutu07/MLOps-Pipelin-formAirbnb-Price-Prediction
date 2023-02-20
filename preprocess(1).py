"""Feature engineers the airbnb dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


feature_columns_names = [
    "room_type",
    "accommodates",
    "bathrooms",
    "cancellation_policy",
    "cleaning_fee",
    "instant_bookable",
    "review_scores_rating",
    "bedrooms",
    "beds",
]
label_column = "price"

feature_columns_dtype = {
    "room_type": str,
    "accommodates": np.float64,
    "bathrooms": np.float64,
    "cancellation_policy": str,
    "cleaning_fee": np.bool,
    "instant_bookable": str,
    "review_scores_rating": np.float64,
    "bedrooms": np.float64,
    "beds": np.float64,
    
}
label_column_dtype = {"price": np.float64}



if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    
    # Change the bucket name
    bucket = "sagemaker-us-east-1-503936503418"
    my_region = boto3.session.Session().region_name

    base_dir = "/opt/ml/processing/"

    logger.debug("Reading data.")

    # Check the key value
    s3_client = boto3.client("s3")
    df = pd.read_csv(s3_client.get_object(Bucket=bucket, Key='airbnb.csv').get("Body"))

    logger.debug("Defining transformers.")
    numeric_features = ["accommodates","bathrooms","review_scores_rating","bedrooms","beds"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["room_type","cancellation_policy","cleaning_fee","instant_bookable"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    
    y = df.pop("price")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    # Splitting the data
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    
