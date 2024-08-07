import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv('data/crops.csv')

# Display the first few rows and columns of the dataset to understand its structure
print(df.head())
print(df.columns)

# Drop columns that exist, adjust according to your dataset
# Assuming 'label' is the column with target labels
if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
else:
    raise KeyError("Column 'label' not found in dataset")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler and model
scaler = StandardScaler()
model = RandomForestClassifier()

# Fit the scaler and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved.")
