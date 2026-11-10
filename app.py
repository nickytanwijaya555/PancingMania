from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

BASE = Path(__file__).resolve().parent
ART = BASE / 'artifacts'
app = Flask(__name__)


def load_artifacts():
    required = ['mlp_sales_model.keras','preprocessor.joblib','ood_detector.joblib','monthly_product_sales.csv','metadata.json','metrics.json','history.json']
    missing = [x for x in required if not (ART/x).exists()]
    if missing: return None, f"Artefak model belum tersedia: {', '.join(missing)}. Jalankan train_model.py terlebih dahulu."
    model = tf.keras.models.load_model(ART/'mlp_sales_model.keras')
    prep = joblib.load(ART/'preprocessor.joblib')
    detector = joblib.load(ART/'ood_detector.joblib')
    monthly = pd.read_csv(ART/'monthly_product_sales.csv',parse_dates=['MonthStart','TargetMonth'])
    metadata = json.loads((ART/'metadata.json').read_text())
    metrics = json.loads((ART/'metrics.json').read_text())
    history = json.loads((ART/'history.json').read_text())
    return {'model':model,'prep':prep,'detector':detector,'monthly':monthly,'metadata':metadata,'metrics':metrics,'history':history}, None

ARTIFACTS, LOAD_ERROR = load_artifacts()

@app.route('/')
def index():
    if ARTIFACTS is None:
        return render_template('setup.html', error=LOAD_ERROR)
    products = sorted(ARTIFACTS['monthly']['_ProductID'].astype(str).unique().tolist())
    return render_template('index.html', products=products, metrics=ARTIFACTS['metrics'], history=ARTIFACTS['history'])

@app.get('/api/months/<product_id>')
def product_months(product_id):
    if ARTIFACTS is None: return jsonify({'error':LOAD_ERROR}),503
    df = ARTIFACTS['monthly']
    rows = df[df['_ProductID'].astype(str)==str(product_id)].copy()
    # A row can be forecast only when three prior months exist.
    rows = rows.dropna(subset=['SalesLag1','SalesLag2','SalesLag3','SalesRollingMean3'])
    return jsonify([{'feature_month':d.strftime('%Y-%m'),'forecast_month':(d+pd.DateOffset(months=1)).strftime('%Y-%m')} for d in sorted(rows['MonthStart'].unique())])

@app.post('/api/predict')
def predict():
    if ARTIFACTS is None: return jsonify({'error':LOAD_ERROR}),503
    payload = request.get_json(silent=True) or {}
    product = str(payload.get('product_id',''))
    month = str(payload.get('feature_month',''))
    df = ARTIFACTS['monthly']
    rows = df[(df['_ProductID'].astype(str)==product) & (df['MonthStart'].dt.strftime('%Y-%m')==month)]
    if rows.empty: return jsonify({'error':'Kombinasi produk dan bulan tidak ditemukan.'}),404
    row = rows.iloc[-1].copy()
    required = ['SalesLag1','SalesLag2','SalesLag3','SalesRollingMean3']
    if row[required].isna().any(): return jsonify({'error':'Histori penjualan belum cukup untuk produk ini.'}),400
    target_month = pd.Timestamp(row['MonthStart']) + pd.DateOffset(months=1)
    model_input = pd.DataFrame([{
        '_ProductID':product,'Year':target_month.year,'MonthNum':target_month.month,
        'MonthSin':np.sin(2*np.pi*target_month.month/12),'MonthCos':np.cos(2*np.pi*target_month.month/12),
        'SalesLag1':row['SalesLag1'],'SalesLag2':row['SalesLag2'],'SalesLag3':row['SalesLag3'],
        'SalesRollingMean3':row['SalesRollingMean3'],'AvgDiscount':row['AvgDiscount'],
        'AvgUnitCost':row['AvgUnitCost'],'AvgUnitPrice':row['AvgUnitPrice'],
        'TransactionCount':row['TransactionCount'],'SalesChannelCount':row['SalesChannelCount'],
        'WarehouseCount':row['WarehouseCount']}])
    try:
        X = ARTIFACTS['prep'].transform(model_input[ARTIFACTS['metadata']['feature_columns']]).astype('float32')
        prediction = float(ARTIFACTS['model'].predict(X,verbose=0).reshape(-1)[0])
        prediction = max(0.0,prediction)
        is_inlier = int(ARTIFACTS['detector'].predict(X)[0]) == 1
        ood_score = float(ARTIFACTS['detector'].decision_function(X)[0])
        # SHAP KernelExplainer: explain one prediction using a compact training background.
        shap_items = []
        shap_note = None
        try:
            import shap
            background = np.load(ART/'shap_background.npy')
            explainer = shap.KernelExplainer(lambda z: ARTIFACTS['model'].predict(np.asarray(z,dtype='float32'),verbose=0).reshape(-1), background)
            values = explainer.shap_values(X, nsamples=100)
            arr = np.asarray(values)
            if arr.ndim == 3: arr = arr[0]
            arr = arr.reshape(-1)
            names = ARTIFACTS['metadata']['transformed_feature_names']
            top = np.argsort(np.abs(arr))[::-1][:8]
            shap_items = [{'feature':names[i].replace('numeric__','').replace('product__','Product '), 'impact':float(arr[i])} for i in top]
        except Exception as e:
            shap_note = 'Penjelasan SHAP belum tersedia pada sesi ini: ' + str(e)[:180]
        return jsonify({'product_id':product,'feature_month':month,'forecast_month':target_month.strftime('%Y-%m'),
                        'prediction':prediction,'prediction_rounded':int(round(prediction)),
                        'ood_status':'normal' if is_inlier else 'warning','ood_score':ood_score,
                        'shap':shap_items,'shap_note':shap_note,
                        'message':'Input masih berada dalam pola data training.' if is_inlier else 'Input berbeda dari pola data training. Gunakan prediksi dengan hati-hati.'})
    except Exception as e:
        return jsonify({'error':'Gagal menghasilkan prediksi: '+str(e)}),500

@app.get('/api/evaluation')
def evaluation():
    if ARTIFACTS is None: return jsonify({'error':LOAD_ERROR}),503
    return jsonify({'metrics':ARTIFACTS['metrics'],'history':ARTIFACTS['history']})

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)
