#!/usr/bin/env python3
"""
K-Means Optimisation — Rapport HTML interactif
Méthodes : Random | AE+K-means++ | MiniBatch K-means
Contrôles : K · N · Features · Dimension latente (synth)
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, webbrowser, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_blobs
from scipy.spatial.distance import cdist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

K_RANGE        = [2, 3, 4, 5, 6]
N_OPTIONS      = [100, 200, 300, 500, 1000]
FEAT_OPTIONS   = [5, 10, 20]
LATENT_OPTIONS = [3, 4, 5, 6]
DEFAULT_K_IRIS  = 3
DEFAULT_K_SYNTH = 4
DEFAULT_N       = 200
DEFAULT_FEAT    = 10
DEFAULT_LATENT  = 3
MAX_SCATTER_PTS = 200

# ─── AUTOENCODER ──────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super().__init__()
        h = max(16, input_dim * 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, h//4),     nn.ReLU(),
            nn.Linear(h//4, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h//4), nn.ReLU(),
            nn.Linear(h//4, h//2),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, h),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, input_dim)
        )
    def encode(self, x): return self.encoder(x)
    def forward(self, x): z = self.encode(x); return self.decoder(z), z


def train_ae(X, latent_dim=3, epochs=100, batch_size=32):
    scaler = StandardScaler()
    X_t    = torch.FloatTensor(scaler.fit_transform(X)).to(DEVICE)
    model  = Autoencoder(X.shape[1], latent_dim).to(DEVICE)
    opt_   = optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        idx = np.random.permutation(len(X_t)); ep = 0.0
        for i in range(0, len(X_t), batch_size):
            b = X_t[idx[i:i+batch_size]]
            rec, _ = model(b); loss = crit(rec, b)
            opt_.zero_grad(); loss.backward(); opt_.step(); ep += loss.item()
        losses.append(ep / max(1, len(X_t) // batch_size))
    model.eval()
    return model, scaler, losses


# ─── UTILS ────────────────────────────────────────────────────────────────────
def encode_z(model, scaler, X):
    model.eval()
    with torch.no_grad():
        return model.encode(torch.FloatTensor(scaler.transform(X)).to(DEVICE)).cpu().numpy()

def labels_inertia(X, labels):
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]; total += float(np.sum((pts - pts.mean(0))**2))
    return total

def conv_hist(X, n_clusters, init="k-means++", seed=42, max_iter=80, tol=1e-4):
    rng = np.random.RandomState(seed)
    cents = X[rng.choice(X.shape[0], n_clusters, replace=False)].copy() \
            if init == "random" else \
            kmeans_plusplus(X, n_clusters=n_clusters, random_state=seed)[0]
    hist = []
    for _ in range(max_iter):
        lbl = np.argmin(cdist(X, cents), 1)
        hist.append(float(np.sum((X - cents[lbl])**2)))
        new = np.array([X[lbl==k].mean(0) if np.any(lbl==k) else cents[k] for k in range(n_clusters)])
        if np.linalg.norm(new - cents) < tol: cents = new; break
        cents = new
    return hist

PALETTE = ["#4e9af1","#f4a261","#2ec4b6","#e63946","#8338ec","#06d6a0","#ffbe0b","#fb5607"]
METHODS = {"Random":"#e63946", "Autoencoder":"#2ec4b6", "MiniBatch":"#8338ec"}

def r(v, n=4): return round(float(v), n)

def scatter_json(Z, labels, centroids):
    if Z.shape[1] > 2:
        pca = PCA(n_components=2); Z2 = pca.fit_transform(Z); c2 = pca.transform(centroids)
        ev = pca.explained_variance_ratio_
        xl, yl = f"PC1 ({ev[0]:.0%})", f"PC2 ({ev[1]:.0%})"
    else:
        Z2, c2, xl, yl = Z, centroids, "Dim 1", "Dim 2"
    n_c = int(labels.max()) + 1; n_per_c = MAX_SCATTER_PTS // n_c
    rng = np.random.RandomState(42)
    ds  = []
    for k in range(n_c):
        idx = np.where(labels == k)[0]
        if len(idx) > n_per_c: idx = rng.choice(idx, n_per_c, replace=False)
        ds.append({"label": f"Cluster {k+1}", "color": PALETTE[k % len(PALETTE)],
                   "pts": [{"x": r(float(Z2[i,0]),3), "y": r(float(Z2[i,1]),3)} for i in idx]})
    return {"datasets": ds,
            "centroids": [{"x": r(float(c2[i,0]),3), "y": r(float(c2[i,1]),3)} for i in range(len(c2))],
            "xLabel": xl, "yLabel": yl}

def _sil(Z, lbl): return r(silhouette_score(Z, lbl))    if len(set(lbl.tolist()))>1 else 0.0
def _dbi(Z, lbl): return r(davies_bouldin_score(Z, lbl)) if len(set(lbl.tolist()))>1 else 9.99

def pack(X, labels, km, init_method, X_orig=None, with_conv=True):
    X_el = X_orig if X_orig is not None else X
    out  = {"silhouette": _sil(X, labels), "davies_bouldin": _dbi(X, labels),
            "inertia":    r(labels_inertia(X_el, labels)),
            "scatter":    scatter_json(X, labels, km.cluster_centers_)}
    if with_conv:
        out["convergence"] = conv_hist(X, int(km.n_clusters), init_method)
    return out


# ─── IRIS GRID (ld=3 fixe) ───────────────────────────────────────────────────
def compute_iris_grid(X, k_range):
    ld = min(3, X.shape[1] - 1)
    print(f"  Iris AE (ld={ld}) ...", end=" ", flush=True)
    mdl, scl, ae_losses = train_ae(X, latent_dim=ld, epochs=100)
    Z = encode_z(mdl, scl, X); print("OK")
    by_k = {}
    for k in k_range:
        km_r  = KMeans(n_clusters=k, init="random",    n_init=1,  random_state=42).fit(X)
        km_ae = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(Z)
        km_mb = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10,
                                batch_size=64, random_state=42).fit(Z)
        by_k[str(k)] = {
            "Random":      pack(X, km_r.labels_,  km_r,  "random"),
            "Autoencoder": pack(Z, km_ae.labels_, km_ae, "k-means++", X_orig=X),
            "MiniBatch":   pack(Z, km_mb.labels_, km_mb, "k-means++", X_orig=X, with_conv=False),
        }
    return by_k, [r(v,6) for v in ae_losses]


# ─── SYNTH GRID (ld variable) ────────────────────────────────────────────────
def compute_synth_grid(k_range, n_options, feat_options, latent_options):
    by_key, ae_losses_map = {}, {}
    total = len(n_options) * len(feat_options) * len(latent_options)
    done  = 0
    for n in n_options:
        for n_feat in feat_options:
            for ld_req in latent_options:
                done += 1
                ld = min(ld_req, n_feat - 1)
                print(f"  ({done}/{total}) N={n} feat={n_feat} latent={ld_req}→{ld} ...",
                      end=" ", flush=True)
                X, _ = make_blobs(n, n_features=n_feat, centers=4, random_state=42, cluster_std=0.8)
                mdl, scl, ae_l = train_ae(X, latent_dim=ld, epochs=100)
                Z = encode_z(mdl, scl, X)
                ae_losses_map[f"{n}_{n_feat}_{ld_req}"] = [r(v,6) for v in ae_l]
                for k in k_range:
                    key   = f"{k}_{n}_{n_feat}_{ld_req}"
                    km_r  = KMeans(n_clusters=k, init="random",    n_init=1,  random_state=42).fit(X)
                    km_ae = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(Z)
                    km_mb = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10,
                                           batch_size=64, random_state=42).fit(Z)
                    by_key[key] = {
                        "Random":      pack(X, km_r.labels_,  km_r,  "random"),
                        "Autoencoder": pack(Z, km_ae.labels_, km_ae, "k-means++", X_orig=X),
                        "MiniBatch":   pack(Z, km_mb.labels_, km_mb, "k-means++", X_orig=X, with_conv=False),
                    }
                print("OK")
    return by_key, ae_losses_map


# ─── HTML TEMPLATE ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>K-Means Optimisation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#0b0d14;--bg2:#13151f;--bg3:#1a1d2b;--bg4:#20243a;
  --border:#252838;--text:#dde1f0;--muted:#6b7094;
  --blue:#4e9af1;--teal:#2ec4b6;--orange:#f4a261;
  --red:#e63946;--purple:#8338ec;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.5}
header{background:linear-gradient(120deg,#07102a,#0e1d5c 55%,#07102a);
  border-bottom:1px solid #1a2860;padding:2rem 3rem;display:flex;align-items:center;gap:1.5rem}
.h-ico{font-size:2.8rem;filter:drop-shadow(0 0 12px #4e9af1aa)}
header h1{font-size:1.75rem;font-weight:700;
  background:linear-gradient(90deg,#4e9af1,#2ec4b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
header p{color:var(--muted);font-size:.85rem;margin-top:.3rem}
.badge{display:inline-block;background:#162050;border:1px solid #2d4aaa;
  border-radius:20px;padding:.12rem .65rem;font-size:.72rem;color:#7ab2f7;
  margin-left:.35rem;-webkit-text-fill-color:#7ab2f7}
.tabs-bar{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:0 2rem;display:flex;position:sticky;top:0;z-index:100;overflow-x:auto}
.tab-btn{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:.9rem;padding:.85rem 1.3rem;border-bottom:3px solid transparent;white-space:nowrap}
.tab-btn:hover{color:var(--text)}.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue)}
.tab-panel{display:none}.tab-panel.active{display:block}
.wrap{max-width:1440px;margin:0 auto;padding:0 1.8rem 3rem}
.controls{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:1.4rem 2rem;margin:1.5rem 0;display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center}
.ctrl-group{display:flex;flex-direction:column;gap:.4rem;min-width:150px}
.ctrl-group label{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.ctrl-row{display:flex;align-items:center;gap:.8rem}
.val-badge{background:var(--bg4);border:1px solid var(--border);border-radius:6px;
  padding:.15rem .6rem;font-size:.95rem;font-weight:700;color:var(--blue);min-width:2.2rem;text-align:center}
input[type=range]{-webkit-appearance:none;appearance:none;width:150px;height:5px;
  border-radius:3px;background:var(--bg4);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;
  border-radius:50%;background:var(--blue);cursor:pointer;border:2px solid var(--bg2)}
select{background:var(--bg4);border:1px solid var(--border);color:var(--text);
  border-radius:7px;padding:.4rem .75rem;font-size:.9rem;cursor:pointer;outline:none}
.ctrl-ticks{display:flex;justify-content:space-between;width:150px;
  font-size:.65rem;color:var(--muted);padding:0 4px}
.section{background:var(--bg2);border:1px solid var(--border);
  border-radius:12px;padding:1.6rem 2rem;margin:1.2rem 0}
.sec-title{font-size:.74rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--blue);border-bottom:1px solid var(--border);padding-bottom:.6rem;margin-bottom:1.2rem}
.methods-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.2rem}
.method-block{background:var(--bg3);border-radius:10px;padding:1rem 1.3rem;border-left:4px solid var(--border)}
.method-block h3{font-size:.88rem;margin-bottom:.65rem}
.cards-row{display:flex;gap:.6rem;flex-wrap:wrap}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:.65rem .9rem;flex:1;min-width:90px}
.card-lbl{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:.15rem}
.card-val{font-size:1.25rem;font-weight:700}
.card-hint{font-size:.6rem;color:var(--muted);margin-top:.1rem}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem}
.grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1rem;margin-top:1rem}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}
.chart-box{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:1.1rem}
.chart-box h4{font-size:.7rem;color:var(--muted);font-weight:600;text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:.7rem}
.chart-wrap{position:relative}
footer{text-align:center;color:var(--muted);font-size:.78rem;
  border-top:1px solid var(--border);padding:1.6rem;margin-top:1rem}
</style>
</head>
<body>
<header>
  <div class="h-ico">⚙️</div>
  <div>
    <h1>K-Means Optimisation
      <span class="badge">AE + K-means++</span>
      <span class="badge">MiniBatch</span>
      <span class="badge">Latent 3→6D</span>
    </h1>
    <p>K · N · Features · Dimension latente &nbsp;|&nbsp; Random vs AE+K-means++ vs MiniBatch K-means</p>
  </div>
</header>
<div class="tabs-bar" id="tabs-bar"></div>
<div class="wrap" id="app"></div>
<footer id="footer"></footer>

<script>
const DATA = __DATA_JSON__;
const DEV  = "__DEVICE__";
const PAL  = ["#4e9af1","#f4a261","#2ec4b6","#e63946","#8338ec","#06d6a0","#ffbe0b","#fb5607"];
const MC   = {Random:"#e63946", Autoencoder:"#2ec4b6", MiniBatch:"#8338ec"};
Chart.defaults.animation = false;
Chart.defaults.color     = "#9aa0c0";

const GRID = {color:"#1a1d2d"}, TICK = {color:"#6b7094"};
function axes(xl="",yl=""){return{
  x:{grid:GRID,ticks:TICK,title:{display:!!xl,text:xl,color:"#6b7094"}},
  y:{grid:GRID,ticks:TICK,title:{display:!!yl,text:yl,color:"#6b7094"}}
};}
function baseOpts(title=""){return{
  responsive:true,maintainAspectRatio:false,
  plugins:{legend:{labels:{color:"#9aa0c0",boxWidth:11,padding:8}},
           title:{display:!!title,text:title,color:"#dde1f0",font:{size:11,weight:"600"}}},
};}

function mkScatter(canvas, sd, title){
  const ds=sd.datasets.map(d=>({label:d.label,data:d.pts,
    backgroundColor:d.color+"cc",pointRadius:4,borderWidth:0}));
  ds.push({label:"Centroids",data:sd.centroids,backgroundColor:"#fff",
    pointStyle:"star",pointRadius:11,borderWidth:0,showLine:false});
  return new Chart(canvas,{type:"scatter",data:{datasets:ds},options:{
    ...baseOpts(title),
    scales:{x:{...axes().x,title:{display:true,text:sd.xLabel,color:"#6b7094"}},
            y:{...axes().y,title:{display:true,text:sd.yLabel,color:"#6b7094"}}},
  }});
}
function autoLabels(datasets){
  const n=Math.max(0,...datasets.map(d=>(d.data||[]).length));
  return Array.from({length:n},(_,i)=>i+1);
}
function mkLine(canvas,datasets,title,xl="",yl="",labels=null){
  const lbl=labels||autoLabels(datasets);
  return new Chart(canvas,{type:"line",data:{labels:lbl,datasets},
    options:{...baseOpts(title),elements:{point:{radius:2,hoverRadius:5}},scales:axes(xl,yl)}});
}
function mkBar(canvas,labels,datasets,title){
  return new Chart(canvas,{type:"bar",data:{labels,datasets},
    options:{...baseOpts(title),scales:axes()}});
}

function upScatter(ch,sd,title){
  const ds=sd.datasets.map(d=>({label:d.label,data:d.pts,
    backgroundColor:d.color+"cc",pointRadius:4,borderWidth:0}));
  ds.push({label:"Centroids",data:sd.centroids,backgroundColor:"#fff",
    pointStyle:"star",pointRadius:11,borderWidth:0,showLine:false});
  ch.data.datasets=ds;
  ch.options.scales.x.title.text=sd.xLabel;
  ch.options.scales.y.title.text=sd.yLabel;
  ch.options.plugins.title.text=title;
  ch.update("none");
}
function upLine(ch,datasets,labels=null){
  ch.data.labels=labels||autoLabels(datasets);
  ch.data.datasets=datasets;
  ch.update("none");
}
function upBar(ch,labels,datasets){
  ch.data.labels=labels;ch.data.datasets=datasets;ch.update("none");
}
function setCard(id,val){const el=document.getElementById(id);if(el)el.textContent=val;}

function elbowData(ds,dsName,n,feat,ld){
  const kRange=ds.k_range,rand=[],ae=[],mb=[],silR=[],silAE=[],silMB=[];
  kRange.forEach(k=>{
    const key = dsName==="iris" ? String(k) : `${k}_${n}_${feat}_${ld}`;
    const map  = dsName==="iris" ? ds.by_k   : ds.by_k_n_feat_ld;
    if(!map[key]) return;
    rand.push(map[key].Random.inertia);
    ae.push(map[key].Autoencoder.inertia);
    mb.push(map[key].MiniBatch.inertia);
    silR.push(map[key].Random.silhouette);
    silAE.push(map[key].Autoencoder.silhouette);
    silMB.push(map[key].MiniBatch.silhouette);
  });
  return {rand,ae,mb,silR,silAE,silMB,kRange};
}
function elbowDs(ed,curK,kRange){
  const ptC=col=>kRange.map(k=>k===curK?"#fff":col);
  const ptS=()=>kRange.map(k=>k===curK?9:4);
  return [
    {label:"Random",    data:ed.rand,borderColor:"#e63946",backgroundColor:"#e6394622",
     tension:.3,fill:false,pointBackgroundColor:ptC("#e63946"),pointRadius:ptS()},
    {label:"AE+K-means++",data:ed.ae,borderColor:"#2ec4b6",backgroundColor:"#2ec4b622",
     tension:.3,fill:false,pointBackgroundColor:ptC("#2ec4b6"),pointRadius:ptS()},
    {label:"MiniBatch", data:ed.mb,borderColor:"#8338ec",backgroundColor:"#8338ec22",
     tension:.3,fill:false,pointBackgroundColor:ptC("#8338ec"),pointRadius:ptS()},
  ];
}
function silDs(ed,curK,kRange){
  const ptC=col=>kRange.map(k=>k===curK?"#fff":col);
  const ptS=()=>kRange.map(k=>k===curK?9:4);
  return [
    {label:"Random",    data:ed.silR, borderColor:"#e63946",backgroundColor:"#e6394622",
     tension:.3,fill:false,pointBackgroundColor:ptC("#e63946"),pointRadius:ptS()},
    {label:"AE+K-means++",data:ed.silAE,borderColor:"#2ec4b6",backgroundColor:"#2ec4b622",
     tension:.3,fill:false,pointBackgroundColor:ptC("#2ec4b6"),pointRadius:ptS()},
    {label:"MiniBatch", data:ed.silMB,borderColor:"#8338ec",backgroundColor:"#8338ec22",
     tension:.3,fill:false,pointBackgroundColor:ptC("#8338ec"),pointRadius:ptS()},
  ];
}

/* ── BUILD TAB ──────────────────────────────────────────────────────────── */
function buildTab(dsName,idx){
  const ds=DATA[dsName], isS=dsName==="synth";

  const controls=`
  <div class="controls">
    <div class="ctrl-group">
      <label>K — Clusters</label>
      <div class="ctrl-row">
        <input type="range" id="k-${dsName}" min="${ds.k_range[0]}" max="${ds.k_range[ds.k_range.length-1]}" value="${ds.default_k}" step="1">
        <div class="val-badge" id="kv-${dsName}">${ds.default_k}</div>
      </div>
      <div class="ctrl-ticks">${ds.k_range.map(k=>`<span>${k}</span>`).join("")}</div>
    </div>
    ${isS?`
    <div class="ctrl-group">
      <label>N — Échantillons</label>
      <select id="n-${dsName}">
        ${ds.n_options.map(n=>`<option value="${n}"${n===ds.default_n?" selected":""}>${n}</option>`).join("")}
      </select>
    </div>
    <div class="ctrl-group">
      <label>Features</label>
      <select id="feat-${dsName}">
        ${ds.feat_options.map(f=>`<option value="${f}"${f===ds.default_feat?" selected":""}>${f}</option>`).join("")}
      </select>
    </div>
    <div class="ctrl-group">
      <label>Dimension latente Z</label>
      <div class="ctrl-row">
        <input type="range" id="ld-${dsName}" min="${ds.latent_options[0]}" max="${ds.latent_options[ds.latent_options.length-1]}" value="${ds.default_latent}" step="1">
        <div class="val-badge" id="ldv-${dsName}">${ds.default_latent}D</div>
      </div>
      <div class="ctrl-ticks">${ds.latent_options.map(l=>`<span>${l}</span>`).join("")}</div>
    </div>`:""}
    <div style="color:var(--muted);font-size:.78rem;align-self:flex-end;padding-bottom:.2rem">
      ${isS?"":"Iris · N=150 · 4 features · latent=3D (fixe)"}
    </div>
  </div>`;

  const methodBlocks=[
    {key:"Random",      col:"#e63946", title:"K-means Random"},
    {key:"Autoencoder", col:"#2ec4b6", title:"AE + K-means++"},
    {key:"MiniBatch",   col:"#8338ec", title:"MiniBatch K-means"},
  ].map(({key,col,title})=>`
    <div class="method-block" style="border-left-color:${col}">
      <h3 style="color:${col}">${title}</h3>
      <div class="cards-row">
        <div class="card"><div class="card-lbl">Silhouette</div>
          <div class="card-val" id="sil-${key}-${dsName}" style="color:${col}">—</div>
          <div class="card-hint">↑ mieux</div></div>
        <div class="card"><div class="card-lbl">Davies-Bouldin</div>
          <div class="card-val" id="dbi-${key}-${dsName}" style="color:#f4a261">—</div>
          <div class="card-hint">↓ mieux</div></div>
        <div class="card"><div class="card-lbl">Inertie</div>
          <div class="card-val" id="iner-${key}-${dsName}" style="color:#8338ec">—</div></div>
      </div>
    </div>`).join("");

  return `
  <div class="tab-panel ${idx===0?"active":""}" id="tab-${idx}">
    ${controls}
    <div class="section">
      <div class="sec-title">Comparaison — Random vs AE+K-means++ vs MiniBatch</div>
      <div class="methods-row">${methodBlocks}</div>
      <div class="grid-3">
        <div class="chart-box"><h4>Clusters — K-means Random (espace X)</h4>
          <div class="chart-wrap" style="height:240px"><canvas id="sc-Random-${dsName}"></canvas></div></div>
        <div class="chart-box"><h4>Clusters — AE + K-means++ (espace Z)</h4>
          <div class="chart-wrap" style="height:240px"><canvas id="sc-Autoencoder-${dsName}"></canvas></div></div>
        <div class="chart-box"><h4>Clusters — MiniBatch K-means (espace Z)</h4>
          <div class="chart-wrap" style="height:240px"><canvas id="sc-MiniBatch-${dsName}"></canvas></div></div>
      </div>
      <div class="grid-4" style="margin-top:1rem">
        <div class="chart-box"><h4>Elbow — Inertie vs K</h4>
          <div class="chart-wrap" style="height:220px"><canvas id="elbow-${dsName}"></canvas></div></div>
        <div class="chart-box"><h4>Silhouette vs K</h4>
          <div class="chart-wrap" style="height:220px"><canvas id="sil-k-${dsName}"></canvas></div></div>
        <div class="chart-box"><h4>Convergence K-means</h4>
          <div class="chart-wrap" style="height:220px"><canvas id="conv-${dsName}"></canvas></div></div>
        <div class="chart-box"><h4>Loss AE — entraînement</h4>
          <div class="chart-wrap" style="height:220px"><canvas id="ae-loss-${dsName}"></canvas></div></div>
      </div>
    </div>
  </div>`;
}

/* ── INIT CHARTS ────────────────────────────────────────────────────────── */
function initCharts(dsName){
  const ds=DATA[dsName], isS=dsName==="synth";
  const C={};

  function curK(){ return parseInt(document.getElementById(`k-${dsName}`).value); }
  function curN(){ return isS ? parseInt(document.getElementById(`n-${dsName}`).value) : null; }
  function curF(){ return isS ? parseInt(document.getElementById(`feat-${dsName}`).value) : null; }
  function curL(){ return isS ? parseInt(document.getElementById(`ld-${dsName}`).value) : null; }

  function getData(k,n,feat,ld){
    const key = isS ? `${k}_${n}_${feat}_${ld}` : String(k);
    return isS ? ds.by_k_n_feat_ld[key] : ds.by_k[key];
  }
  function getAEL(n,feat,ld){
    if(!isS) return ds.ae_losses;
    return ds.ae_losses_by_nf_ld[`${n}_${feat}_${ld}`]||[];
  }

  const k0=ds.default_k, n0=isS?ds.default_n:null, f0=isS?ds.default_feat:null, l0=isS?ds.default_latent:null;
  const kd0=getData(k0,n0,f0,l0);
  const ed0=elbowData(ds,dsName,n0,f0,l0);
  const MKEYS=["Random","Autoencoder","MiniBatch"];

  /* scatter */
  MKEYS.forEach(m=>{
    C[`sc${m}`]=mkScatter(document.getElementById(`sc-${m}-${dsName}`),
      kd0[m].scatter, `${m} — Sil=${kd0[m].silhouette}`);
  });

  /* elbow / sil */
  C.elbow=mkLine(document.getElementById(`elbow-${dsName}`),
    elbowDs(ed0,k0,ds.k_range),"Elbow — Inertie vs K","K","Inertie",ds.k_range);
  C.silK=mkLine(document.getElementById(`sil-k-${dsName}`),
    silDs(ed0,k0,ds.k_range),"Silhouette vs K","K","Silhouette",ds.k_range);

  /* convergence (Random + AE only) */
  C.conv=mkLine(document.getElementById(`conv-${dsName}`),[
    {label:"Random",       data:kd0.Random.convergence,
     borderColor:"#e63946",backgroundColor:"#e6394622",tension:.3,fill:false},
    {label:"AE+K-means++", data:kd0.Autoencoder.convergence,
     borderColor:"#2ec4b6",backgroundColor:"#2ec4b622",tension:.3,fill:false},
  ],"Convergence K-means","Itération","Inertie");

  /* AE loss */
  C.aeLoss=mkLine(document.getElementById(`ae-loss-${dsName}`),[
    {label:"Loss AE",data:getAEL(n0,f0,l0),
     borderColor:"#4e9af1",backgroundColor:"#4e9af122",tension:.3,fill:true},
  ],"Loss AE","Epoch","MSE");

  /* seed cards */
  function seedCards(kd){
    MKEYS.forEach(m=>{
      setCard(`sil-${m}-${dsName}`,  kd[m].silhouette);
      setCard(`dbi-${m}-${dsName}`,  kd[m].davies_bouldin);
      setCard(`iner-${m}-${dsName}`, kd[m].inertia);
    });
  }
  seedCards(kd0);

  /* update */
  function update(){
    const k=curK(), n=curN(), f=curF(), ld=curL();
    document.getElementById(`kv-${dsName}`).textContent=k;
    if(isS) document.getElementById(`ldv-${dsName}`).textContent=ld+"D";

    const kd=getData(k,n,f,ld);
    if(!kd){ console.warn("missing",k,n,f,ld); return; }
    seedCards(kd);

    MKEYS.forEach(m=>
      upScatter(C[`sc${m}`], kd[m].scatter, `${m} — Sil=${kd[m].silhouette}`));

    const ed=elbowData(ds,dsName,n,f,ld);
    upLine(C.elbow, elbowDs(ed,k,ds.k_range), ds.k_range);
    upLine(C.silK,  silDs(ed,k,ds.k_range),   ds.k_range);

    upLine(C.conv,[
      {label:"Random",       data:kd.Random.convergence,
       borderColor:"#e63946",backgroundColor:"#e6394622",tension:.3,fill:false},
      {label:"AE+K-means++", data:kd.Autoencoder.convergence,
       borderColor:"#2ec4b6",backgroundColor:"#2ec4b622",tension:.3,fill:false},
    ]);
    upLine(C.aeLoss,[{label:"Loss AE",data:getAEL(n,f,ld),
      borderColor:"#4e9af1",backgroundColor:"#4e9af122",tension:.3,fill:true}]);
  }

  document.getElementById(`k-${dsName}`).addEventListener("input", update);
  if(isS){
    document.getElementById(`n-${dsName}`).addEventListener("change", update);
    document.getElementById(`feat-${dsName}`).addEventListener("change", update);
    document.getElementById(`ld-${dsName}`).addEventListener("input", update);
  }
}

/* ── BOOTSTRAP ──────────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded",()=>{
  const bar=document.getElementById("tabs-bar");
  const app=document.getElementById("app");
  const tabs=[{key:"iris",label:"Iris  (N=150, 4 feat, latent=3D)"},{key:"synth",label:"Synthétique  (N · features · latent variables)"}];
  tabs.forEach(({key,label},idx)=>{
    const btn=document.createElement("button");
    btn.className="tab-btn"+(idx===0?" active":"");
    btn.textContent=label;
    btn.onclick=()=>{
      document.querySelectorAll(".tab-btn").forEach((b,j)=>b.classList.toggle("active",j===idx));
      document.querySelectorAll(".tab-panel").forEach((p,j)=>p.classList.toggle("active",j===idx));
    };
    bar.appendChild(btn);
    app.insertAdjacentHTML("beforeend",buildTab(key,idx));
  });
  tabs.forEach(({key})=>initCharts(key));
  document.getElementById("footer").textContent=`Rapport K-Means Optimisation — Device : ${DEV}`;
});
</script>
</body>
</html>"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = {}

    print("\n" + "="*60)
    print("  IRIS  (N=150, 4 features, latent=3D fixe)")
    print("="*60)
    X_iris = load_iris().data
    by_k, ae_losses = compute_iris_grid(X_iris, K_RANGE)
    report["iris"] = {
        "k_range":   K_RANGE,
        "default_k": DEFAULT_K_IRIS,
        "ae_losses": ae_losses,
        "by_k":      by_k,
    }

    print("\n" + "="*60)
    print(f"  SYNTHÉTIQUE  ({len(N_OPTIONS)}N × {len(FEAT_OPTIONS)}feat × {len(LATENT_OPTIONS)}ld × {len(K_RANGE)}K)")
    print("="*60)
    by_k_n_feat_ld, ae_losses_by_nf_ld = compute_synth_grid(
        K_RANGE, N_OPTIONS, FEAT_OPTIONS, LATENT_OPTIONS)
    report["synth"] = {
        "k_range":            K_RANGE,
        "n_options":          N_OPTIONS,
        "feat_options":       FEAT_OPTIONS,
        "latent_options":     LATENT_OPTIONS,
        "default_k":          DEFAULT_K_SYNTH,
        "default_n":          DEFAULT_N,
        "default_feat":       DEFAULT_FEAT,
        "default_latent":     DEFAULT_LATENT,
        "ae_losses_by_nf_ld": ae_losses_by_nf_ld,
        "by_k_n_feat_ld":     by_k_n_feat_ld,
    }

    print("\n  Génération rapport.html ...")
    data_json = json.dumps(report, ensure_ascii=False)
    html = HTML.replace("__DATA_JSON__", data_json).replace("__DEVICE__", str(DEVICE))
    out  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rapport.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"  Sauvegardé : {out}  ({size_mb:.1f} MB)")
    webbrowser.open(f"file:///{out.replace(os.sep, '/')}")
