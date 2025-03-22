import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split , KFold, StratifiedKFold,  cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Load the labels 
labels_df = pd.read_excel('labels.xlsx', header=0)
print(labels_df.columns)
labels_df.rename(columns={  # Renaming operations for smooth processing
  'Solenoid valves Opening Ratio /%': 'SV1',
  'Unnamed: 4': 'SV2',
  'Unnamed: 5': 'SV3',
  'Unnamed: 6': 'SV4',
  'Bubble'    : 'BP1',
  'Unnamed: 8':  'BP2',
  'Unnamed: 9':  'BP3',
  'Unnamed: 10': 'BP4',
  'Unnamed: 11': 'BP5',
  'Unnamed: 12': 'BP6',
  'Unnamed: 13': 'BP7',
  'Unnamed: 14': 'BV1',
  }, inplace=True)



def determine_condition(case_number):  # Function to determine if a case is normal or abnormal
    condition = labels_df.loc[labels_df['Case#'] == case_number, 'Condition'].values[0]
    if condition == 'Normal':
        return 0  # Normal
    else:
        return 1  # Abnormal


def determine_abnormality(case_number):  # Function to determine the type of abnormality
    condition = labels_df.loc[labels_df['Case#'] == case_number, 'Condition'].values[0]
    if condition == 'Fault':
        return 1  # Solenoid valve fault
    elif condition == 'Anomaly':
        return 2  # Bubble contamination

def determine_bubble_location(case_number):   # Function to determine the bubble location
    bubble_columns = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BV1']
    print(labels_df.columns)
# Inspecting the columns again to ensure they are renamed correctly
    bubble_location = labels_df.loc[labels_df['Case#'] == case_number, bubble_columns].values[0]
    for i, location in enumerate(bubble_location):
        if location == 'Yes':
            # return bubble_columns[i]
            return i + 1 # BP1 is 1, BP2 is 2, etc Return the index + 1 for the bubble location (1-based index)
    # return 'Unknown'
    return 0


def determine_failed_solenoid(case_number): # Function to determine the failed solenoid valve
    solenoid_columns = ['SV1', 'SV2', 'SV3', 'SV4']
    solenoid_status = labels_df.loc[labels_df['Case#'] == case_number, solenoid_columns].values[0]
    for i, status in enumerate(solenoid_status):
        if status != 100:  # Assuming a failed solenoid valve has an opening ratio not equal to 100%
            # return solenoid_columns[i]
            return i + 1 # SV1 is 1, SV2 is 2, etc Return the index + 1 for the failed solenoid valve (1-based index)
    # return 'Unknown'
    return 0


# def get_opening_ratio(case_number, failed_solenoid): # Function to get the opening ratio of the failed solenoid valve
#     return labels_df.loc[labels_df['Case#'] == case_number, failed_solenoid].values[0]
# Function to get the opening ratio of the failed solenoid valve
def get_opening_ratio(case_number, failed_solenoid):
    if failed_solenoid > 0:
        solenoid_columns = ['SV1', 'SV2', 'SV3', 'SV4']
        return labels_df.loc[labels_df['Case#'] == case_number, solenoid_columns[failed_solenoid - 1]].values[0]
    return 100

# Directory containing the training data 
training_data_dir = '../train'


def extract_case_number(file_name): # Function to extract the numeric part from the file name
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric part found in file name: {file_name}")

# List to store the features and labels for normal/abnormal classification
features = []
labels = []

# List to store the features and labels for abnormality classification
abnormal_features = []
abnormal_labels = []

# List to store the features and labels for bubble location classification
bubble_features = []
bubble_labels = []

# List to store the features and labels for solenoid valve fault classification
solenoid_features = []
solenoid_labels = []

# List to store the features and labels for solenoid valve opening ratio prediction
solenoid_ratio_features = []
solenoid_ratio_labels = []

# Iterate over each training data file
for file_name in os.listdir(training_data_dir):
    if file_name.endswith('.csv'):
        
        case_number = extract_case_number(file_name) # Extracting the case number from the file name
        
        # Loading the CSV file
        df = pd.read_csv(os.path.join(training_data_dir, file_name))
        
        # Extract features (mean and standard deviation of pressure sensor data)
        feature_vector = []
        for col in df.columns[1:]:  # Skip the TIME column if considering LSTM it will be used
            feature_vector.append(df[col].mean())
            feature_vector.append(df[col].std())
        
        # Append the features and label to the lists for normal/abnormal classification
        features.append(feature_vector)
        labels.append(determine_condition(case_number))
        
        # If the case is abnormal, append the features and label to the lists for abnormality classification
        if determine_condition(case_number) == 1:
            abnormal_features.append(feature_vector)
            abnormal_labels.append(determine_abnormality(case_number))
            
            # If the abnormality is bubble contamination, append the features and label to the lists for bubble location classification
            if determine_abnormality(case_number) == 2:
                bubble_features.append(feature_vector)
                bubble_labels.append(determine_bubble_location(case_number))
            
            # If the abnormality is solenoid valve fault, append the features and label to the lists for solenoid valve fault classification
            if determine_abnormality(case_number) == 1:
                solenoid_features.append(feature_vector)
                solenoid_labels.append(determine_failed_solenoid(case_number))
                
                # Append the features and label to the lists for solenoid valve opening ratio prediction
                failed_solenoid = determine_failed_solenoid(case_number)
                solenoid_ratio_features.append(feature_vector)
                solenoid_ratio_labels.append(get_opening_ratio(case_number, failed_solenoid))

# Convert features and labels to numpy arrays for normal/abnormal classification
X = np.array(features)
y = np.array(labels)


# Split the data into training and validation sets for normal/abnormal classification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model for normal/abnormal classification
model_normal_abnormal = RandomForestClassifier(n_estimators=100, random_state=42)
model_normal_abnormal.fit(X_train, y_train)


y_pred = model_normal_abnormal.predict(X_val) # Validate the model for normal/abnormal classification
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy (Normal/Abnormal): {accuracy}")


joblib.dump(model_normal_abnormal, 'old_model_normal_abnormal.pkl') # Save the trained model for normal/abnormal classification to a file
print("The trained model for normal/abnormal classification has been saved to 'model_normal_abnormal.pkl'.")


X_abnormal = np.array(abnormal_features) # Convert features and labels to numpy arrays for abnormality classification
y_abnormal = np.array(abnormal_labels)

# Split the data into training and validation sets for abnormality classification
X_train_abnormal, X_val_abnormal, y_train_abnormal, y_val_abnormal = train_test_split(X_abnormal, y_abnormal, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model for abnormality classification
model_abnormality = RandomForestClassifier(n_estimators=100, random_state=42)
model_abnormality.fit(X_train_abnormal, y_train_abnormal)


y_pred_abnormal = model_abnormality.predict(X_val_abnormal) # Validate the model for abnormality classification
accuracy_abnormal = accuracy_score(y_val_abnormal, y_pred_abnormal)
print(f"Validation Accuracy (Abnormality Classification): {accuracy_abnormal}")

# Save the trained model for abnormality classification to a file
joblib.dump(model_abnormality, 'old_model_abnormality.pkl')
print("The trained model for abnormality classification has been saved to 'model_abnormality.pkl'.")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ H.tuning +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Hyperparameter Tuning for some operations --

# Feature Scaling for normal/abnormal classification
# Convert features and labels to numpy arrays for bubble location classification
X_bubble = np.array(bubble_features)
y_bubble = np.array(bubble_labels)
scaler = StandardScaler()
X_bubble_scaled = scaler.fit_transform(X_bubble)


# Split the data into training and validation sets for normal/abnormal classification with cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)
# kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=30)
# cross_val_scores = cross_val_score(RandomForestClassifier(n_estimators=20, random_state=42), X_bubble_scaled, y, cv=kf)
# print(f"Cross-Validation Scores (Normal/Abnormal): {cross_val_scores}")
# print(f"Mean Cross-Validation Score (Normal/Abnormal): {np.mean(cross_val_scores)}")


param_grid_rf = { 
    'n_estimators': [20, 50, 100],  # Moderate number of trees
    'max_depth': [2, 1, 1],  # Limiting depth to avoid overly complex trees
    'min_samples_split': [5, 10, 20],  # Higher values to prevent overfitting
    'min_samples_leaf': [2, 5, 10],  # Higher values to prevent overfitting
    'max_features': ['sqrt', 'log2']  # Limit the number of features for splits
}

# Convert features and labels to numpy arrays for bubble location classification
grid_search_bubble = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=kf, scoring='accuracy')
grid_search_bubble.fit(X_bubble, y_bubble)
best_params_bubble = grid_search_bubble.best_params_
print(f"Best Parameters (Bubble Location Classification): {best_params_bubble}")
 
# Split the data into training and validation sets for bubble location classification
# X_train_bubble, X_val_bubble, y_train_bubble, y_val_bubble = train_test_split(X_bubble, y_bubble, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model for bubble location classification
model_bubble_location = RandomForestClassifier(**best_params_bubble, random_state=42)
model_bubble_location.fit(X_bubble_scaled, y_bubble)

# Validate the model for bubble location classification
y_pred_bubble = model_bubble_location.predict(X_bubble_scaled)
accuracy_bubble = accuracy_score(y_bubble, y_pred_bubble)
print(f"Validation Accuracy (Bubble Location Classification): {accuracy_bubble}")
# Save confusion matrix for training accuracy and validation accuracy (Bubble Location Classification)

y_pred_train_bubble_location = model_bubble_location.predict(X_bubble_scaled)
conf_matrix_train_bubble_location = confusion_matrix(y_bubble, y_pred_train_bubble_location)
sns.heatmap(conf_matrix_train_bubble_location, annot=True, fmt='d')
plt.title('Confusion Matrix - Training (Bubble Location Classification)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('conf_matrix_train_bubble_location.png')
plt.show()

# Save the trained model for bubble location classification to a file
joblib.dump(model_bubble_location, 'old_model_bubble_location.pkl')
print("The trained model for bubble location classification has been saved to 'model_bubble_location.pkl'.")

# Convert features and labels to numpy arrays for solenoid valve fault classification
X_solenoid = np.array(solenoid_features)
y_solenoid = np.array(solenoid_labels)

# Feature Scaling for solenoid valve fault classification
X_solenoid_scaled = scaler.fit_transform(X_solenoid)

# Hyperparameter tuning using GridSearchCV for solenoid valve fault classification
grid_search_solenoid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=kf, scoring='accuracy')
grid_search_solenoid.fit(X_solenoid_scaled, y_solenoid)
best_params_solenoid = grid_search_solenoid.best_params_
print(f"Best Parameters (Solenoid Valve Fault Classification): {best_params_solenoid}")

# Split the data into training and validation sets for solenoid valve fault classification
# X_train_solenoid, X_val_solenoid, y_train_solenoid, y_val_solenoid = train_test_split(X_solenoid, y_solenoid, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model for solenoid valve fault classification
model_solenoid_fault = RandomForestClassifier(n_estimators=100, random_state=42)
model_solenoid_fault.fit(X_solenoid_scaled, y_solenoid)

# Validate the model for solenoid valve fault classification
y_pred_solenoid = model_solenoid_fault.predict(X_solenoid_scaled)
accuracy_solenoid = accuracy_score(y_solenoid, y_pred_solenoid)
print(f"Validation Accuracy (Solenoid Valve Fault Classification): {accuracy_solenoid}")

# Save confusion matrix for training accuracy and validation accuracy (Solenoid Valve Fault Classification)
y_pred_train_solenoid_fault = model_solenoid_fault.predict(X_solenoid_scaled)
conf_matrix_train_solenoid_fault = confusion_matrix(y_solenoid, y_pred_train_solenoid_fault)
sns.heatmap(conf_matrix_train_solenoid_fault, annot=True, fmt='d')
plt.title('Confusion Matrix - Training (Solenoid Valve Fault Classification)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('conf_matrix_train_solenoid_fault.png')
plt.show()



# Save the trained model for solenoid valve fault classification to a file
joblib.dump(model_solenoid_fault, 'old_model_solenoid_fault.pkl')
print("The trained model for solenoid valve fault classification has been saved to 'model_solenoid_fault.pkl'.")

# Convert features and labels to numpy arrays for solenoid valve opening ratio prediction
X_ratio = np.array(solenoid_ratio_features)
y_ratio = np.array(solenoid_ratio_labels)

# Split the data into training and validation sets for solenoid valve opening ratio prediction
X_train_ratio, X_val_ratio, y_train_ratio, y_val_ratio = train_test_split(X_ratio, y_ratio, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model for solenoid valve opening ratio prediction
model_solenoid_ratio = RandomForestRegressor(n_estimators=100, random_state=42)
model_solenoid_ratio.fit(X_train_ratio, y_train_ratio)

# Validate the model for solenoid valve opening ratio prediction
y_pred_ratio = model_solenoid_ratio.predict(X_val_ratio)
mse_ratio = mean_squared_error(y_val_ratio, y_pred_ratio)
print(f"Validation Mean Squared Error (Solenoid Valve Opening Ratio Prediction): {mse_ratio}")

# Save the trained model for solenoid valve opening ratio prediction to a file
joblib.dump(model_solenoid_ratio, 'old_model_solenoid_ratio.pkl')
print("The trained model for solenoid valve opening ratio prediction has been saved to 'model_solenoid_ratio.pkl'.")

#++++++++++++++++++++++++++++++++++++++++++testing the models on new test data+++++++++++++++++++++++++++++++++++++++++++++++++++++

def test_models(test_data_dir):               
    # Load the trained models
    model_normal_abnormal = joblib.load('old_model_normal_abnormal.pkl')
    model_abnormality = joblib.load('old_model_abnormality.pkl')
    model_bubble_location = joblib.load('old_model_bubble_location.pkl')
    model_solenoid_fault = joblib.load('old_model_solenoid_fault.pkl')
    model_solenoid_ratio = joblib.load('old_model_solenoid_ratio.pkl')
    
    # List to store the results
    results = []

    # Iterate over each test data file
    for file_name in os.listdir(test_data_dir):
        if file_name.endswith('.csv'):
            # Extract the case number from the file name
            case_number = extract_case_number(file_name)
            
            # Load the CSV file
            df = pd.read_csv(os.path.join(test_data_dir, file_name))
            
            # Extract features (mean and standard deviation of pressure sensor data)
            feature_vector = []
            for col in df.columns[1:]:  # Skip the TIME column
                feature_vector.append(df[col].mean())
                feature_vector.append(df[col].std())
            
            # Predict if the condition is normal or abnormal using the trained model
            condition = model_normal_abnormal.predict([feature_vector])[0]
            condition_label = 0 if condition == 0 else 1   # 0 FOR NORMAL AND 1 FOR ABNORMAL
            # Initialize variables
            abnormality_label = 0
            bubble_location = 0
            failed_solenoid = 0
            opening_ratio = 100
            if condition_label == 1: # ABNORMAL
                # Predict the type of abnormality using the trained model for abnormality classification
                abnormality_type = model_abnormality.predict([feature_vector])[0]
                if abnormality_type == 1:
                    abnormality_label = 3 # SOLONOID VALVE FAULT
                    # Predict the failed solenoid valve
                    failed_solenoid = model_solenoid_fault.predict([feature_vector])[0]
                    # Predict the opening ratio of the failed solenoid valve
                    opening_ratio = model_solenoid_ratio.predict([feature_vector])[0]
                elif abnormality_type == 2:
                    abnormality_label = 2 # BUBBLE CONTAMINATION
                    # If bubble contamination, predict the bubble location
                    bubble_location = model_bubble_location.predict([feature_vector])[0]
                else:
                    abnormality_label = 1 # UNKNOWN ANOMALY
            # Append the result to the list
            results.append({
                'Id': case_number,
                'Task 1': condition_label,
                'Task 2': abnormality_label,
                'Task 3': bubble_location,
                'Task 4': failed_solenoid,
                'Task 5': opening_ratio
            })
            # Append the result to the list
            # results.append({'Case#': case_number, 'Condition': condition_label, 'Abnormality Type': abnormality_label, 'Bubble Location': bubble_location, 'Failed Solenoid': failed_solenoid, 'Opening Ratio': opening_ratio})

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv('newfinaltest_data_predictions_with_abnormalities.csv', index=False)
    print("The predictions for all test data have been saved to 'newfinaltest_data_predictions_with_abnormalities.csv'.")

#to test
test_models('../dataset/test/data/')
# test_models('../testdata/')