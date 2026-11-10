"""Train monthly product-sales MLP and save artifacts for the Flask website.
Run: python train_model.py --data US_Regional_Sales_Data.csv
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / 'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
CAT = ['_ProductID']
NUM = ['Year','MonthNum','MonthSin','MonthCos','SalesLag1','SalesLag2','SalesLag3',
       'SalesRollingMean3','AvgDiscount','AvgUnitCost','AvgUnitPrice',
       'TransactionCount','SalesChannelCount','WarehouseCount']
FEATURES = CAT + NUM


def load_and_prepare(path: Path):
    df = pd.read_csv(path)
    # Dataset versions may name this field ProductID or _ProductID.
    if '_ProductID' not in df.columns:
        candidates = ['ProductID', 'Product Id', 'Product_ID']
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError('Kolom ID produk tidak ditemukan. Kolom yang tersedia: ' + ', '.join(df.columns))
        df['_ProductID'] = df[found].astype(str)
    else:
        df['_ProductID'] = df['_ProductID'].astype(str)

    required = ['OrderDate','Order Quantity']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Kolom wajib tidak ditemukan: ' + ', '.join(missing))
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], dayfirst=True, errors='coerce')
    for col in ['Unit Cost','Unit Price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).str.strip(), errors='coerce')
        else:
            df[col] = np.nan
    if 'Discount Applied' in df.columns:
        df['Discount Applied'] = pd.to_numeric(df['Discount Applied'], errors='coerce')
    else:
        df['Discount Applied'] = np.nan
    if 'OrderNumber' not in df.columns: df['OrderNumber'] = np.arange(len(df)).astype(str)
    if 'Sales Channel' not in df.columns: df['Sales Channel'] = 'Unknown'
    if 'WarehouseCode' not in df.columns: df['WarehouseCode'] = 'Unknown'
    df['Order Quantity'] = pd.to_numeric(df['Order Quantity'], errors='coerce')
    valid = df.dropna(subset=['OrderDate','_ProductID','Order Quantity']).copy()
    valid['Month'] = valid['OrderDate'].dt.to_period('M')
    monthly = valid.groupby(['_ProductID','Month']).agg(
        MonthlySales=('Order Quantity','sum'), AvgDiscount=('Discount Applied','mean'),
        AvgUnitCost=('Unit Cost','mean'), AvgUnitPrice=('Unit Price','mean'),
        TransactionCount=('OrderNumber','nunique'), SalesChannelCount=('Sales Channel','nunique'),
        WarehouseCount=('WarehouseCode','nunique')).reset_index()
    products = sorted(valid['_ProductID'].astype(str).unique())
    months = pd.period_range(valid['Month'].min(), valid['Month'].max(), freq='M')
    grid = pd.MultiIndex.from_product([products, months], names=['_ProductID','Month'])
    monthly = monthly.set_index(['_ProductID','Month']).reindex(grid).reset_index()
    for c in ['MonthlySales','TransactionCount','SalesChannelCount','WarehouseCount']:
        monthly[c] = monthly[c].fillna(0)
    monthly['MonthStart'] = monthly['Month'].dt.to_timestamp()
    monthly = monthly.sort_values(['_ProductID','MonthStart']).reset_index(drop=True)
    g = monthly.groupby('_ProductID', sort=False)
    monthly['SalesLag1'] = g['MonthlySales'].shift(1)
    monthly['SalesLag2'] = g['MonthlySales'].shift(2)
    monthly['SalesLag3'] = g['MonthlySales'].shift(3)
    monthly['SalesRollingMean3'] = g['MonthlySales'].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    monthly['TargetNextMonth'] = g['MonthlySales'].shift(-1)
    monthly['TargetMonth'] = monthly['MonthStart'] + pd.DateOffset(months=1)
    # Calendar predictors describe the month being forecast, not the historical feature month.
    monthly['Year'] = monthly['TargetMonth'].dt.year
    monthly['MonthNum'] = monthly['TargetMonth'].dt.month
    monthly['MonthSin'] = np.sin(2*np.pi*monthly['MonthNum']/12)
    monthly['MonthCos'] = np.cos(2*np.pi*monthly['MonthNum']/12)
    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='US_Regional_Sales_Data.csv')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists(): path = BASE / args.data
    if not path.exists(): raise FileNotFoundError(f'Dataset tidak ditemukan: {args.data}')

    monthly = load_and_prepare(path)
    model_df = monthly.dropna(subset=['SalesLag1','SalesLag2','SalesLag3','SalesRollingMean3','TargetNextMonth']).copy()
    if model_df.empty: raise ValueError('Tidak cukup histori. Setiap produk memerlukan setidaknya 4 bulan data berurutan.')
    months = np.array(sorted(model_df['TargetMonth'].unique()))
    if len(months) < 5: raise ValueError('Periode bulanan terlalu sedikit untuk split train/validation/test.')
    train_cut = months[max(1, int(len(months)*0.70))]
    val_cut = months[max(2, int(len(months)*0.85))]
    train_df = model_df[model_df['TargetMonth'] < train_cut].copy()
    val_df = model_df[(model_df['TargetMonth'] >= train_cut) & (model_df['TargetMonth'] < val_cut)].copy()
    test_df = model_df[model_df['TargetMonth'] >= val_cut].copy()
    if min(len(train_df),len(val_df),len(test_df)) == 0:
        raise ValueError('Salah satu subset kosong. Dataset memerlukan rentang bulan yang lebih panjang.')

    try: enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError: enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    numeric_pipe = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',RobustScaler())])
    preprocessor = ColumnTransformer([('product',enc,CAT),('numeric',numeric_pipe,NUM)])
    X_train = preprocessor.fit_transform(train_df[FEATURES]).astype('float32')
    X_val = preprocessor.transform(val_df[FEATURES]).astype('float32')
    X_test = preprocessor.transform(test_df[FEATURES]).astype('float32')
    y_train = train_df['TargetNextMonth'].to_numpy(dtype='float32')
    y_val = val_df['TargetNextMonth'].to_numpy(dtype='float32')
    y_test = test_df['TargetNextMonth'].to_numpy(dtype='float32')

    model = Sequential([Input(shape=(X_train.shape[1],)), Dense(128,activation='relu'), Dropout(.2),
                        Dense(64,activation='relu'), Dropout(.2), Dense(32,activation='relu'),
                        Dropout(.2), Dense(1,activation='linear')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mse', metrics=['mae'])
    callbacks = [EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=5,min_lr=1e-5)]
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=args.epochs,batch_size=32,
                        callbacks=callbacks,verbose=1)
    pred = model.predict(X_test,verbose=0).reshape(-1)
    metrics = {'MAE':float(mean_absolute_error(y_test,pred)),
               'RMSE':float(np.sqrt(mean_squared_error(y_test,pred))),
               'R2':float(r2_score(y_test,pred)),
               'train_rows':int(len(train_df)),'validation_rows':int(len(val_df)),
               'test_rows':int(len(test_df)),'input_features':int(X_train.shape[1]),
               'train_target_period_start':str(train_df['TargetMonth'].min().date()),
               'test_target_period_start':str(test_df['TargetMonth'].min().date())}

    # OOD detector learns only from the transformed training feature distribution.
    detector = IsolationForest(n_estimators=200, contamination='auto', random_state=RANDOM_STATE)
    detector.fit(X_train)
    try: feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception: feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    bg = X_train[np.random.choice(len(X_train), size=min(25,len(X_train)), replace=False)]

    model.save(ARTIFACTS/'mlp_sales_model.keras')
    joblib.dump(preprocessor,ARTIFACTS/'preprocessor.joblib')
    joblib.dump(detector,ARTIFACTS/'ood_detector.joblib')
    np.save(ARTIFACTS/'shap_background.npy',bg)
    monthly.drop(columns=['Month'],errors='ignore').to_csv(ARTIFACTS/'monthly_product_sales.csv',index=False)
    pd.DataFrame({'actual':y_test,'predicted':pred,'target_month':test_df['TargetMonth'].astype(str).values,
                  'product_id':test_df['_ProductID'].values}).to_csv(ARTIFACTS/'test_predictions.csv',index=False)
    (ARTIFACTS/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (ARTIFACTS/'metadata.json').write_text(json.dumps({'categorical_features':CAT,'numeric_features':NUM,
        'feature_columns':FEATURES,'transformed_feature_names':feature_names,
        'products':sorted(monthly['_ProductID'].astype(str).unique().tolist()),
        'architecture':'128-64-32-1','activation':'ReLU / Linear','optimizer':'Adam','learning_rate':0.001,
        'batch_size':32,'max_epochs':args.epochs,'actual_epochs':len(history.history['loss']),
        'target':'TargetNextMonth (monthly product sales)'},indent=2))
    (ARTIFACTS/'history.json').write_text(json.dumps({k:[float(v) for v in vals] for k,vals in history.history.items()},indent=2))
    print('\nTEST METRICS')
    for k,v in metrics.items(): print(f'{k}: {v}')
    print(f'\nArtifacts tersimpan di: {ARTIFACTS}')

if __name__ == '__main__': main()
