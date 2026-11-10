<<<<<<< HEAD
# SalesPredict — MLP Sales Forecasting Website

A Flask frontend/backend for monthly product sales forecasting using a TensorFlow/Keras MLP. It includes two supporting features: SHAP local explanations and Isolation Forest out-of-distribution (OOD) warnings.

## 1. Project structure

```text
salespredict_web/
├── app.py
├── train_model.py
├── requirements.txt
├── artifacts/                 # generated after training
├── templates/
│   ├── index.html
│   └── setup.html
└── static/
    ├── app.js
    └── style.css
```

## 2. Setup locally (Windows)

Install Python 3.10–3.12, then open Terminal/PowerShell in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place `US_Regional_Sales_Data.csv` in this folder, then train the model:

```bash
python train_model.py --data US_Regional_Sales_Data.csv
```

After successful training, run:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## 3. Dataset and target

The training script aggregates transaction rows into a monthly product panel. The target is `TargetNextMonth`, i.e. total `Order Quantity` for the next month for a product. The product identifier is read from `_ProductID` or `ProductID` (with a few common aliases). Required source columns are `OrderDate` and `Order Quantity`. The script expects the optional fields `Discount Applied`, `Unit Cost`, `Unit Price`, `OrderNumber`, `Sales Channel`, and `WarehouseCode`; missing optional fields are filled with conservative defaults.

The app asks for a product and a completed historical feature month. It uses the selected month and earlier sales history to predict the following month. Do not treat the current, incomplete month as a completed month unless the source data is complete through that date.

## 4. Model details

- MLP architecture: Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Linear).
- Optimizer: Adam, learning rate 0.001; loss: MSE; metrics: MAE.
- Batch size: 32; maximum epochs: 100; EarlyStopping and ReduceLROnPlateau.
- Chronological split is performed by forecast/target month (70/15/15), not random row splitting.
- Imputer, RobustScaler, and OneHotEncoder are fitted on training data only.
- Test metrics (MAE, RMSE, R²), loss history, test predictions, model, preprocessing pipeline, and OOD detector are saved under `artifacts/`.

## 5. Unique features

1. **SHAP explanation:** Kernel SHAP explains the selected prediction. The website displays the largest positive/negative feature contributions. SHAP is an explanation tool, not a guarantee of causality or accuracy. Kernel SHAP can take a few seconds per request.
2. **OOD detection:** Isolation Forest is fitted on transformed training features. If the selected product-month input is outside the learned distribution, the website displays a warning. It does not automatically invalidate the prediction.

## 6. Important implementation notes

- The original preprocessing notebook used `_ProductID` without showing where it was created. This training script explicitly maps `ProductID` to `_ProductID` when needed.
- The target-month calendar fields (Year, MonthNum, MonthSin, MonthCos) refer to the month being predicted.
- The script uses a zero-sales assumption for product-months without recorded transactions. Validate this assumption against the dataset/business context before reporting results.
- Do not publish `debug=True` in production. Use a production WSGI server such as Waitress or Gunicorn and HTTPS for deployment.
- The model is not trained until you run `train_model.py` against the actual CSV. The website package does not contain fabricated metrics or a fabricated trained model.
=======
# PancingMania
>>>>>>> be46e5401c80fd8f139394360a0a90ed5251bea7
