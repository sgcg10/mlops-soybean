# model/testing.py
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from logger import logger

class ModelTester:
    def evaluate_model_performance(self, X_test, y_test, model):
        """
        Evaluate the model performance
        Parameters:
            self: self object
            X_test: Dataframe with testing data in X
            y_test: Dataframe with testing data in Y (Columns)
            model: The Logistic Regression model
        Returns:
            accuracy: Accuracy of the evaluation of the model
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("Logistic Regression:")
        logger.info(classification_report(y_test, y_pred))
        return accuracy

    def calculate_confusion_matrix(self, X_test, X_train, y_train, y_test, x_label, y_label, classes, iterations):
        """
        Visualize the model comparison
        Parameters:
            self: self object
            X_test: Dataframe with testing data in X
            X_train: Dataframe with training data in X
            y_train: Dataframe with training data in Y (Columns)
            y_test: Dataframe with testing data in Y (Columns)
            x_label: The label to put in X
            y_label: The label to put in y
            classes: Classes to consider
            iterations: The number of iterations used in the Logistic Regression model
        Returns:
            confusion_matrix_figure: The rendered diagram of the confusion matrix
        """
        model = LogisticRegression(max_iter=iterations, solver='liblinear', random_state=0).fit(X_train, y_train)
        cm = confusion_matrix(y_test, model.predict(X_test))
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        confusion_matrix_figure = px.imshow(df_cm, labels=dict(x=x_label, y=y_label, color="Count"),
                        x=classes, y=classes, color_continuous_scale="Blues", title="Confusion Matrix")
        return confusion_matrix_figure
