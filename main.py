import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "data"  # Directory where the dataset is located
DATA_FILE = "drug200.csv"  # Dataset filename
TARGET_COLUMN = "Drug"  # Target column name


def load_data(filepath):
    """
    Reads data from the specified CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Cleans the data by dropping rows with missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    return df.dropna()


def encode_and_scale_features(df, target_column):
    """
    Encodes categorical variables, scales numerical features, and separates features and target.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        tuple: Scaled features (X) and encoded target (y).
    """
    df_processed = df.copy()

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = label_encoder.fit_transform(df_processed[col])

    # Separate features and target
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier using GridSearchCV for hyperparameter tuning.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        tuple: The best trained model and the best parameters.
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test data.

    Args:
        model: Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        None
    """
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def visualize_decision_tree(model, feature_names, class_names):
    """
    Visualizes the trained Decision Tree model.

    Args:
        model: Trained Decision Tree model.
        feature_names (list): List of feature names.
        class_names (list): List of class names.

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        fontsize=10
    )
    plt.title("Decision Tree Visualization")
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    filepath = f"{DATA_DIR}/{DATA_FILE}"
    df = load_data(filepath)
    df = clean_data(df)
    X, y = encode_and_scale_features(df, TARGET_COLUMN)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the Decision Tree model
    best_model, best_params = train_decision_tree(X_train, y_train)
    print("\nBest Hyperparameters:", best_params)
    evaluate_model(best_model, X_test, y_test)

    # Visualize the Decision Tree
    visualize_decision_tree(
        best_model,
        feature_names=df.drop(TARGET_COLUMN, axis=1).columns.tolist(),
        class_names=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
    )
