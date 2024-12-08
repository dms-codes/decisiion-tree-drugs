
# Decision Tree Classifier with Grid Search and Visualization

This project demonstrates how to build, train, and evaluate a **Decision Tree Classifier** using **GridSearchCV** for hyperparameter tuning. The dataset used is the `drug200.csv`, which contains information about drug prescriptions based on patient data. The code includes data preprocessing, model training, evaluation, and decision tree visualization.

---

## Features

- **Data Loading and Cleaning**: Reads data from a CSV file and handles missing values.
- **Preprocessing**:
  - Encodes categorical variables.
  - Scales numerical features for optimal performance.
- **Model Training**:
  - Uses `DecisionTreeClassifier` with hyperparameter tuning via `GridSearchCV`.
- **Evaluation**:
  - Calculates accuracy, classification report, and confusion matrix.
- **Visualization**:
  - Visualizes the decision tree using `matplotlib` and `plot_tree`.

---

## Requirements

To run this code, you need the following Python packages installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install the required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## File Structure

```plaintext
.
├── data/
│   └── drug200.csv       # Dataset file
├── decision_tree.py      # Main script containing the code
└── README.md             # Documentation
```

---

## Usage

### 1. Dataset

Ensure the `drug200.csv` file is located in the `data/` directory. The dataset should include the following columns:

- **Features**: Various patient-related data.
- **Target Column (`Drug`)**: The drug prescribed (e.g., `drugA`, `drugB`, `drugC`, `drugX`, `drugY`).

---

### 2. Running the Code

1. Clone this repository or download the code.
2. Navigate to the directory containing the script.
3. Run the script using:

   ```bash
   python decision_tree.py
   ```

---

## Output

### 1. Console Outputs:
- **Best Hyperparameters**: Displays the best hyperparameter combination found by `GridSearchCV`.
- **Evaluation Metrics**:
  - Accuracy
  - Classification Report
  - Confusion Matrix

### 2. Visualization:
A decision tree plot displaying:
- Feature splits
- Feature importance
- Class labels

Example:
![Decision Tree Example](https://via.placeholder.com/800x400?text=Decision+Tree+Visualization)

---

## Customization

### File Paths:
Modify the `DATA_DIR` and `DATA_FILE` constants to change the dataset path:

```python
DATA_DIR = "data"
DATA_FILE = "drug200.csv"
```

### Target Column:
Update the `TARGET_COLUMN` constant to specify the target column name:

```python
TARGET_COLUMN = "Drug"
```

### Class Labels:
Update the `class_names` parameter in the visualization function for your dataset's target labels:

```python
class_names=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
```

---

## Dependencies and Version Information

This code has been tested with the following versions:
- Python 3.8+
- scikit-learn 1.2+
- pandas 1.4+
- matplotlib 3.5+

---

## Acknowledgments

- **Dataset**: The `drug200.csv` dataset used in this project.
- **Libraries**: This project leverages the power of scikit-learn and matplotlib for machine learning and visualization.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code for any purpose.

---

## Author

Created by [Your Name]. If you have any questions, feel free to reach out! 😊
