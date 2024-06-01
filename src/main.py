# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from model.training import ModelTrainer
from model.testing import ModelTester
from logger import logger
from exception import CustomException

def read_data(file):
    """
    Read data from CSV file
    Parameters:
        file -- CSV file
    Returns:
        dataframe: the dataframe with the content of the file
    """
    return pd.read_csv(file)

try:
    # Load data (static)
    data = read_data("data/soybean_harvest.csv")

    # Initialize ModelTrainer and ModelTester
    trainer = ModelTrainer(iterations=5000)
    tester = ModelTester()

    # Train-test split
    X = pd.DataFrame(data, columns=['NLP', 'NS'])
    y = pd.Series(data['Season'], name='Season')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    }
    best_params, best_score = trainer.hyperparameter_tuning(X_train, y_train, param_grid)

    # Train Classification Model with best parameters
    model = trainer.train_classification_model(X_train, y_train)

    # Evaluate Model Performance
    accuracy = tester.evaluate_model_performance(X_test, y_test, model)

    # Calculate Confusion Matrix
    classes = ["Season1", "Season2"]
    fig = tester.calculate_confusion_matrix(X_test, X_train, y_train, y_test, "Predicted Class", "True Class", classes, trainer.iterations)
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Class", yaxis_title="True Class")
    fig.show()

except CustomException as e:
    logger.error(f"An error occurred: {e}")

except Exception as e:
    logger.exception(f"An unexpected error occurred: {e}")
