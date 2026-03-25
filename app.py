from flask import Flask, request, jsonify
import pandas as pd, joblib, io, base64, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
 
app = Flask(__name__)
 
df = pd.read_csv('Cellphone.csv')
FEATURES = ['ram', 'battery', 'internal mem', 'cpu core', 'RearCam', 'thickness']
X, y = df[FEATURES], df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load('model.pkl') if os.path.exists('model.pkl') else LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
ACC = round(model.score(X_test, y_test) * 100, 2)
MAE = round(mean_absolute_error(y_test, y_pred), 2)
 
def get_plot():
    plt.figure(figsize=(5,3), facecolor='#0f0f0f')
    ax = plt.gca(); ax.set_facecolor('#0f0f0f')
    ax.scatter(y_test, y_pred, color='#4f9eff', alpha=0.3, s=15)
    ax.plot([0,4500],[0,4500], color='#ff4f6d', lw=1.2, ls='--')
    ax.set_xlabel('Real', color='#888'); ax.set_ylabel('Predicted', color='#888')
    ax.tick_params(colors='#555'); [s.set_edgecolor('#222') for s in ax.spines.values()]
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=110, facecolor='#0f0f0f'); plt.close()
    buf.seek(0); return base64.b64encode(buf.read()).decode()
 
def get_table_html():
    sample = df.head(10)
    headers = ''.join(f'<th>{col}</th>' for col in sample.columns)
    rows = ''
    for _, row in sample.iterrows():
        cells = ''.join(f'<td>{val}</td>' for val in row)
        rows += f'<tr>{cells}</tr>'
    return f'''
    <div class="table-section">
      <h2>📊 Dataset Preview <span>(first 10 rows)</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{headers}</tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>'''
 
@app.route('/')
def index():
    return f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Mobile Price AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0a0a0a;color:#e0e0e0;font-family:'Inter',sans-serif;padding:2rem;max-width:900px;margin:auto}}
  h1{{font-size:1.3rem;font-weight:600;margin-bottom:.3rem}}
  p{{color:#666;font-size:.8rem;margin-bottom:1.5rem}}
  .stats{{display:flex;gap:1rem;margin-bottom:1.5rem}}
  .stat{{background:#111;border:1px solid #222;border-radius:10px;padding:.8rem 1.2rem;flex:1}}
  .stat b{{display:block;font-size:1.4rem;color:#4f9eff}}
  .stat span{{font-size:.7rem;color:#555}}
  img{{width:100%;border-radius:10px;margin-bottom:1.5rem}}
  label{{display:block;font-size:.78rem;color:#666;margin-bottom:.3rem}}
  input{{width:100%;background:#111;border:1px solid #222;border-radius:8px;padding:.5rem .8rem;color:#e0e0e0;font-size:.85rem;margin-bottom:.9rem;outline:none}}
  input:focus{{border-color:#4f9eff}}
  button{{width:100%;padding:.7rem;background:#4f9eff;border:none;border-radius:8px;color:#000;font-weight:600;cursor:pointer;font-size:.9rem}}
  button:hover{{background:#6fafff}}
  #res{{margin-top:1rem;text-align:center;font-size:1.5rem;font-weight:600;color:#4f9eff;display:none}}
 
  /* Table styles */
  .table-section{{margin-top:2.5rem}}
  .table-section h2{{font-size:1rem;font-weight:600;margin-bottom:.8rem;color:#ccc}}
  .table-section h2 span{{font-size:.75rem;color:#555;font-weight:400;margin-left:.4rem}}
  .table-wrap{{overflow-x:auto;border-radius:10px;border:1px solid #1e1e1e}}
  table{{width:100%;border-collapse:collapse;font-size:.76rem}}
  thead tr{{background:#111}}
  thead th{{padding:.6rem .9rem;text-align:left;color:#4f9eff;font-weight:600;white-space:nowrap;border-bottom:1px solid #222}}
  tbody tr{{border-bottom:1px solid #161616}}
  tbody tr:last-child{{border-bottom:none}}
  tbody tr:hover{{background:#111}}
  tbody td{{padding:.5rem .9rem;color:#aaa;white-space:nowrap}}
</style></head><body>
<h1>📱 Mobile Price Prediction</h1>
<p>Linear Regression Model</p>
<div class="stats">
  <div class="stat"><b>{ACC}%</b><span>R² Accuracy</span></div>
  <div class="stat"><b>${MAE}</b><span>MAE Error</span></div>
  <div class="stat"><b>{len(df)}</b><span>Samples</span></div>
</div>
<img src="data:image/png;base64,{get_plot()}">
<label>RAM (GB)</label><input id="ram" type="number" value="3">
<label>Battery (mAh)</label><input id="bat" type="number" value="2500">
<label>Internal Memory (GB)</label><input id="mem" type="number" value="16">
<label>CPU Cores</label><input id="cpu" type="number" value="4">
<label>Rear Camera (MP)</label><input id="cam" type="number" value="13">
<label>Thickness (mm)</label><input id="th" type="number" value="8">
<button onclick="predict()">Predict Price</button>
<div id="res"></div>
 
{get_table_html()}
 
<script>
async function predict(){{
  const d={{ram:+ram.value,battery:+bat.value,'internal mem':+mem.value,'cpu core':+cpu.value,RearCam:+cam.value,thickness:+th.value}};
  const r=await(await fetch('/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(d)}})).json();
  const el=document.getElementById('res');
  el.textContent='Predicted: $'+r.price.toLocaleString();
  el.style.display='block';
}}
</script></body></html>'''
 
@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()
    price = float(model.predict([[float(d[f]) for f in FEATURES]])[0])
    return jsonify({'price': round(max(0, price), 2)})
 
if __name__ == '__main__':
    app.run(debug=True)