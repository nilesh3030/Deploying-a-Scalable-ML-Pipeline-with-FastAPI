import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set base project directory
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean leading/trailing whitespaces in column names and string values
data.columns = [col.strip() for col in data.columns]
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].astype(str).str.strip()

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
print("Training model...")
model = train_model(X_train, y_train)

# Ensure model directory exists
os.makedirs(os.path.join(project_path, "model"), exist_ok=True)

# Save the model, encoder, and label binarizer
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
lb_path = os.path.join(project_path, "model", "lb.pkl")

save_model(model, model_path)
print(f"Model saved to {model_path}")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")
save_model(lb, lb_path)
print(f"Label binarizer saved to {lb_path}")

# Load the saved model to verify persistence
model = load_model(model_path)

# Run inference on test dataset
preds = inference(model, X_test)

# Calculate and print overall metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall Metrics - Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute and log performance on categorical slices
slice_output_file = os.path.join(project_path, "slice_output.txt")
with open(slice_output_file, "w") as f:
    f.write("Categorical Slices Performance\n")
    f.write("=" * 40 + "\n")

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        sp, sr, sfb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open(slice_output_file, "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sfb:.4f}\n", file=f)

print(f"Slice metrics successfully saved to {slice_output_file}")
