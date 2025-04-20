# Breast Cancer Tumor Prediction Using Random Forest

This project uses a Random Forest Regressor to predict whether a breast tumor is benign or malignant based on medical features from the Wisconsin Breast Cancer dataset.

## 📊 Features

- Loads and cleans the dataset
- Maps categorical diagnosis to numeric values (Malignant = 1, Benign = 0)
- Trains a Random Forest Regressor
- Calculates MAE (Mean Absolute Error)
- Displays feature importances
- Takes manual input for prediction
- Displays final prediction (M or B)
- Plots ROC curve to visualize model performance

## 📁 Files

- `breastCancerPred.py` - Main script
- `README.txt` - This file
- `LICENSE.txt` - MIT License

## 🧪 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

**Install requirements:**
pip install pandas numpy matplotlib scikit-learn

**🛠️ Usage**
1.Run the script:
  python breastCancerPred.py
2. Enter the values for all the features when prompted.
3. The model will predict if the tumor is Malignant (M) or Benign (B).
4. It also displays the model’s accuracy and plots the ROC curve.
