import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('c:\\Users\\HP\\Desktop\\Fake Profile\\fake_profile_detect\\fake_profiles.csv')

# Features and labels
X = df[['Followers', 'Following', 'Bio_Length', 'Has_Profile_Photo', 'Is_Private']]
y = df['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save RF model and scaler
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 2. Train SVM Model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# 3. Train ANN Model
ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)

# Save ANN model
with open('ann_model.pkl', 'wb') as f:
    pickle.dump(ann_model, f)

print("Models trained and saved!")






# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# import pickle




# # Load dataset
# df = pd.read_csv('c:\\Users\\HP\\Desktop\\Fake Profile\\fake_profile_detect\\fake_profiles.csv')

# # Features and labels
# X = df[['Followers', 'Following', 'Bio_Length', 'Has_Profile_Photo', 'Is_Private']]
# y = df['Label']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 1. Train Random Forest Model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Save RF model
# with open('rf_model.pkl', 'wb') as f:
#     pickle.dump(rf_model, f)

# # 2. Train SVM Model
# svm_model = SVC(kernel='linear', probability=True)
# svm_model.fit(X_train, y_train)

# # Save SVM model
# with open('svm_model.pkl', 'wb') as f:
#     pickle.dump(svm_model, f)

# # 3. Train ANN Model
# ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
# ann_model.fit(X_train, y_train)

# # Save ANN model
# with open('ann_model.pkl', 'wb') as f:
#     pickle.dump(ann_model, f)

# print("Models trained and saved!")
