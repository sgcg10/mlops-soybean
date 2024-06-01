# model/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from logger import logger

class ModelTrainer:
    def __init__(self, iterations):
        self.iterations = iterations
        self.model = LogisticRegression(max_iter=self.iterations)

    def train_classification_model(self, X_train, y_train):
        """
        Train the classification models

        Parameters:
            self: self object
            X_train: Dataframe with training data in X
            y_train: Dataframe with training data in Y (Columns)

        Returns:
            self.model: Trained model

        """
        self.model.fit(X_train, y_train)
        return self.model

    def hyperparameter_tuning(self, X_train, y_train, param_grid):
        """
        Perform hyperparameter tuning for the model using grid search cross-validation.

        Parameters:
            self: self object
            X_train: Features of the training data.
            y_train: Target labels of the training data.
            param_grid: Dictionary specifying the hyperparameter grid to search over.

        Returns:
            search.best_params_, search.best_score_ (tuple): A tuple containing the best parameters found during tuning and the corresponding
                   best cross-validation accuracy score.

        """
        search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='accuracy')
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        logger.info("Best parameters found: ", search.best_params_)
        logger.info("Best cross-validation accuracy: {:.2f}".format(search.best_score_))
        return search.best_params_, search.best_score_

    def save_model(self, file_path):
        """
        Save the trained model to a file using joblib.

        Parameters:
            self: self object
            file_path: The path to save the model file.

        """
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
