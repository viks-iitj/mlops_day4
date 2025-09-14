import os

model_files = ["saved_models/logistic_regression.joblib", "saved_models/random_forest.joblib"]

all_exist = all(os.path.exists(f) for f in model_files)

if all_exist:
    print("All model files exist.")
else:
    print("Some model files are missing.")
    missing = [f for f in model_files if not os.path.exists(f)]
    print("Missing:", missing)

