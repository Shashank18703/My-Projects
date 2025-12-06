from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

# Model Definition (Must match the notebook)
class CarPriceModel(nn.Module):
    def __init__(self, input_dim):
        super(CarPriceModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)

        return x

# Load Artifacts
feature_columns = joblib.load('car_price_featureColumns.pkl')
input_dim = len(feature_columns)
scaler_X = joblib.load("Fitted_scaler_x.pkl")
scaler_Y = joblib.load("fitted_scaler_y.pkl")

model = CarPriceModel(input_dim)
model.load_state_dict(torch.load('carPrice_model.pth'))
model.eval()

app = Flask(__name__)

def predict_price(raw):
    '''
    raw : dict with keys like
    present_price, kms_driven, owner, age, fuel_type, seller_type, transmission
    '''
    # 1. Build feature dict in same way as notebook
    ff = {}
    ff["Presnt_Price"] = float(raw["present_price"])
    ff["Kms_Driven"] = int(raw["kms_driven"])
    ff["Owner"] = int(raw["age"])

    fuel = raw["fuel_type"]
    ff["Fuel_Type_Diesel"] = (fuel == "Diesel")
    ff["Fuel_Type_Petrol"] = (fuel == "Petrol")
    # CNG or other -> both False

    seller = raw["seller_type"]
    ff["Seller_Type_Individual"] = (seller == "Individual")

    trans = raw["transmission"]
    ff["Transmission_Manual"] = (trans == "Manual")

    # Dataframe in the correct column order
    df = pd.DataFrame([ff])
    df = df.reindex(columns = feature_columns, fill_value=False)

    # 2. Splitting the numeric vs. categorical
    numeric_cols = ["Present_Price", "Kms_Driven", "Owner", "Age"]
    X_num = df[numeric_cols].values
    X_cat = df.drop(columns = numeric_cols).values.astype(float)

    # 3. Scaling the numeric features with scaler used for scaler_X
    X_num_scaled = scaler_X.transform(X_num)

    # 4. Reconstruct full input
    X_full_scaled = np.concatenate([X_num_scaled, X_cat], axis = 1)

    # 5. Predict in Scaled Target space
    x_tensor = torch.tensor(X_full_scaled, dtype = torch.float32)
    with torch.no_grad():
        y_scaled_pred = model(x_tensor).numpy().reshape(-1, 1)

    # 6. Inverse-scale to original price units
    y_pred = scaler_Y.inverse_transform(y_scaled_pred)
    price = float(y_pred[0,0])

    return price

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    price = predict_price(data)
    return jsonify({"predicted_price" : price})

if __name__ == "__main__" :
    app.run(debug = True)
    