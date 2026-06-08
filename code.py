import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("/mnt/data/civic_issue_dataset.csv")


features = ["Severity_Score", "Complaint_Category", "Historical_Frequency", "Required_Resources"]
target = "Estimated_Resolution_Time_Days"


df = df.dropna(subset=features + [target])


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categorical_features = ["Complaint_Category"]
numerical_features = ["Severity_Score", "Historical_Frequency"]


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)



def predict_resolution_time(new_data):
    df_new = pd.DataFrame(new_data)
    return model.predict(df_new)


new_test_data = [{
    "Severity_Score": 5.2,
    "Complaint_Category": "Drainage",
    "Historical_Frequency": 30,
    "Required_Resources": "Workers"
}]

predicted_time = predict_resolution_time(new_test_data)
print(f"Predicted Resolution Time: {predicted_time[0]:.2f} days")
