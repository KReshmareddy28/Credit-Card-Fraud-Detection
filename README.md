# Credit Card Fraud Detection (Python + Machine Learning)

## Project overview
This educational project trains a Random Forest classifier to distinguish transactions labeled as legitimate (`Class = 0`) or fraudulent (`Class = 1`). It reports precision, recall, F1-score, ROC-AUC, a confusion matrix, and feature importance.

**Dataset warning:** `creditcard_transactions_demo.csv` is a synthetic dataset generated for this project. It is not real financial data and its performance metrics must not be used for real financial decisions.

## Files
- `main.py` — loads data, trains/evaluates the model, saves the model and charts.
- `creditcard_transactions_demo.csv` — synthetic demonstration dataset (2,000 rows).
- `predict_transaction.py` — asks for feature values and produces a demo prediction after training.
- `requirements.txt` — Python packages.
- `outputs/` — model, metrics JSON and plots are created here after running.

## Requirements
- Python 3.10 or newer recommended
- Internet is needed only to install packages, not to run the project afterward.

## Run on Windows
1. Extract the ZIP file.
2. Open Command Prompt in the extracted `credit_card_fraud_detection` folder.
3. Install packages:
   ```bash
   py -m pip install -r requirements.txt
   ```
4. Train and evaluate:
   ```bash
   py main.py
   ```
5. View `outputs/metrics.json`, `outputs/confusion_matrix.png`, and `outputs/feature_importance.png`.
6. Try a single prediction:
   ```bash
   py predict_transaction.py
   ```
   Enter numeric values when prompted. For a simple test, values such as `0` can be entered for each feature, but this is only an arbitrary demo input.

## Methodology
1. Load and inspect the dataset.
2. Separate the target (`Class`) from input features.
3. Split data into training and testing sets with stratification.
4. Train a Random Forest classifier using class weighting to help with imbalance.
5. Evaluate using precision, recall, F1-score, confusion matrix and ROC-AUC.
6. Save the trained model and visualizations.

## Project abstract
Credit card fraud can cause financial loss to customers and financial institutions. This project demonstrates a machine-learning approach to classifying transactions as legitimate or potentially fraudulent. Transaction attributes are provided to a Random Forest classifier, which learns patterns from labeled examples. The system evaluates its predictions using standard classification metrics and produces a confusion matrix and feature-importance chart. The project is designed for academic learning using synthetic data. A real deployment would require a suitable, lawfully obtained dataset, careful validation, privacy safeguards, human review and ongoing monitoring.

## Expected output
The program prints a classification report and ROC-AUC score, and saves a confusion matrix, feature-importance plot, metrics file and trained model. Exact metrics can vary with library versions. Because the dataset is synthetic, results do not represent real-world fraud detection accuracy.

## Limitations and future enhancements
- Replace the synthetic data with an approved real dataset for meaningful research.
- Tune the decision threshold based on the cost of missed fraud and false alarms.
- Compare Random Forest with Logistic Regression and Gradient Boosting.
- Add a web interface only after validating the model properly.
- Monitor model drift and protect sensitive transaction data.

## Viva questions
1. What is credit card fraud detection?
2. Why is this a classification problem?
3. What does `Class = 1` mean?
4. Why is the dataset imbalanced?
5. What are precision, recall and F1-score?
6. Why is accuracy alone insufficient for imbalanced datasets?
7. What is a confusion matrix?
8. Why is this dataset labeled synthetic?
