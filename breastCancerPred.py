import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the dataset
data = pd.read_csv(r'C:\Users\Private Fox\Downloads\data.csv')
# Drop unnamed columns
data = data.dropna(axis=1, how='all')  # Drop columns that are all NA
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Drop unnamed columns

# Display the shape and columns of the cleaned dataset
print("Dataset shape:", data.shape)
print("\nColumns in dataset:")
print(data.columns.tolist())

# Display the first few rows of the dataset
print(data.head())
# Display the shape of the dataset
print("Dataset shape:", data.shape)
X = data.drop(['diagnosis','id'],axis=1)
y = data['diagnosis']
print(X.describe(),X.info(),X.head())
print(y.head())
# Convert the target variable to numerical values
y = y.map({'M':1, 'B': 0})
# Display the first few rows of the target variable
print(y.head())
# Split the dataset into training and testing sets
X_train ,X_test,y_train,y_test= train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)
#load the model
def mae(X_train,y_train,X_test,y_test):
    model = RandomForestRegressor(max_depth=10,random_state=42,n_estimators=100)
    # Fit the model to the training data
    model.fit(X_train,y_train)
    predict= model.predict(X_test)
    mae= mean_absolute_error(y_test,predict)
    return mae,model

mae_value, model = mae(X_train,y_train,X_test,y_test)
print("MAE:", round(mae_value,5))

#plot the feature importances
def plot_feature_importances(model, X_train):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
# Plot the feature importances
plot_feature_importances(model, X_train)

#input the features
input_data = []
print("\nPlease enter the value for each feature:")
for feature in X.columns:
    while True:
        try:
            value = float(input(f"{feature}: "))
            input_data.append(value)
            break
        except ValueError:
            print("Please enter a valid number.")

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data], columns=X.columns)

# Make a prediction
prediction = model.predict(input_df)
# Convert the prediction to a binary value
prediction = 1 if prediction[0] > 0.5 else 0 
# Display the prediction
if prediction == 1:
    print("The model predicts that the tumor is malignant (M).")
else:
    print("The model predicts that the tumor is benign (B).")
# Display the model's accuracy
accuracy = accuracy_score(y_test, model.predict(X_test).round())
print("Model Accuracy:", round(accuracy, 4))

# Plotting the ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Get probability predictions
y_pred_proba = model.predict(X_test)

# Plot ROC curve
plot_roc_curve(y_test, y_pred_proba)
