import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Define file paths with descriptive variable names
DATA_DIR = "data"  # Assuming data files are in a folder named "data"
DATA_FILE = "drug200.csv"

def load_data():
    """
    Reads the breast cancer data from the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(f"{DATA_DIR}/{DATA_FILE}")
    return df

def clean_data(df):
    """
    Cleans the data by dropping rows with missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    return df.dropna()

def preprocess_data(df):
    """
    Preprocesses the data by encoding categorical variables and scaling numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_preprocessed = df.copy()

    # Encode categorical variables
    le = LabelEncoder()
    for col in df_preprocessed.select_dtypes(include=['object']).columns:
        df_preprocessed[col] = le.fit_transform(df_preprocessed[col])

    # Separate features and target
    features = df_preprocessed.drop('Drug', axis=1)
    target = df_preprocessed['Drug']

    # Scale numerical features (optional but often improves performance)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_clf = grid_search.best_estimator_

    # Make predictions
    y_pred = best_clf.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Print best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(best_clf, 
              filled=True, 
              rounded=True, 
              feature_names=df.columns[:-1], 
              class_names=['drugA', 'drugB', 'drugC', 'drugX','drugY'], 
              fontsize=10) 
    plt.show()