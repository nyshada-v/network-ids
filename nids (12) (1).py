"""
╔══════════════════════════════════════════════════════════════════════╗
║        AI-DRIVEN NETWORK INTRUSION DETECTION SYSTEM (NIDS)          ║
║        Hybrid ML: IsolationForest + Autoencoder + XGBoost           ║
║        Run: python nids.py                                           ║
║        Dashboard: http://localhost:5001                              ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP (run once):
  pip install scapy flask flask-socketio eventlet joblib
              tensorflow xgboost scikit-learn pandas numpy

WINDOWS: Run terminal as Administrator (Scapy needs WinPcap/Npcap)
         Download Npcap from https://npcap.com and install first.

LINUX:   sudo python nids.py

MODEL FILES NEEDED (download from Colab → Files panel):
  scaler.pkl, nzv_cols.pkl, score_stats.pkl
  isolation_forest.pkl, autoencoder.keras
  xgb_detector.pkl, attack_classifier.pkl
  attack_label_encoder.pkl

Put all .pkl / .keras files in the SAME folder as this script.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NETWORK_INTERFACE = None
FLASK_PORT        = 5001
MODEL_DIR         = "."
FLOW_TIMEOUT_SEC  = 2.0
MAX_ROWS          = 300

# ── Warmup ───────────────────────────────────────────────────────────────────
# Collects YOUR network's baseline during the first WARMUP_FLOWS flows.
# Uses MEDIAN + IQR (robust stats) so that if attack traffic sneaks into
# warmup it doesn't skew the baseline — median ignores outliers.
WARMUP_FLOWS = 60

# ── Sensitivity ──────────────────────────────────────────────────────────────
# Threshold = median + SIGMA * IQR  of YOUR normal traffic
# IQR is robust — not affected by attack outliers during warmup
# Lower SIGMA = more sensitive | Higher = stricter
SIGMA_XGB = 5.0   # XGBoost: needs strong signal to fire
SIGMA_IF  = 2.0   # IsolationForest: 2 IQR above median
SIGMA_AE  = 2.0   # AE error: 2 IQR above median

# ── Flow rate detector ────────────────────────────────────────────────────────
# DoS / flood attacks generate many flows from same src→dst in short time.
# This catches high-volume attacks even when individual flow scores are low.
RATE_WINDOW_SEC  = 10    # look at flows in last N seconds
RATE_THRESHOLD   = 20    # if same src→dst seen > N times in window = attack

# ── Adaptive baseline ────────────────────────────────────────────────────────
_baseline = {
    "xgb_vals": [],
    "if_vals":  [],
    "ae_vals":  [],
    "xgb_thresh": None,
    "if_thresh":  None,
    "ae_thresh":  None,
    "ready": False,
}
_flow_count = {"n": 0}

# Flow rate tracking — {(src,dst): [timestamps]}
_flow_rate = collections.defaultdict(list)
_flow_rate_lock = threading.Lock()

def _compute_baseline():
    """Robust baseline using median + IQR — ignores outliers from attack traffic."""
    from scipy.stats import iqr as scipy_iqr
    xgb = np.array(_baseline["xgb_vals"])
    iff = np.array(_baseline["if_vals"])
    ae  = np.array(_baseline["ae_vals"])

    # Use median + IQR (robust) instead of mean + std
    _baseline["xgb_thresh"] = float(np.median(xgb) + SIGMA_XGB * scipy_iqr(xgb))
    _baseline["if_thresh"]  = float(np.median(iff)  + SIGMA_IF  * scipy_iqr(iff))
    _baseline["ae_thresh"]  = float(np.median(ae)   + SIGMA_AE  * scipy_iqr(ae))

    # Safety floor — never set thresholds below meaningful attack levels
    _baseline["xgb_thresh"] = max(_baseline["xgb_thresh"], 0.15)
    _baseline["if_thresh"]  = max(_baseline["if_thresh"],  0.42)
    _baseline["ae_thresh"]  = max(_baseline["ae_thresh"],  0.10)

    _baseline["ready"] = True
    print("\n" + "="*60)
    print("  ✅  BASELINE LEARNED — Detection is now ACTIVE")
    print(f"  XGB threshold : {_baseline['xgb_thresh']:.4f}")
    print(f"  IF  threshold : {_baseline['if_thresh']:.4f}")
    print(f"  AE  threshold : {_baseline['ae_thresh']:.4f}")
    print(f"  Flow rate     : flag if same src→dst > {RATE_THRESHOLD}x in {RATE_WINDOW_SEC}s")
    print("="*60 + "\n")

def _check_flow_rate(src_ip, dst_ip):
    """Returns True if this src→dst pair is flooding — DoS/flood attack."""
    now = time.time()
    key = (src_ip, dst_ip)
    with _flow_rate_lock:
        # Keep only timestamps within the window
        _flow_rate[key] = [t for t in _flow_rate[key] if now - t < RATE_WINDOW_SEC]
        _flow_rate[key].append(now)
        count = len(_flow_rate[key])
    return count > RATE_THRESHOLD, count

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, math, threading, collections, random, logging, warnings
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from scapy.all import sniff, IP, TCP, UDP, get_if_list, conf

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_models(model_dir):
    print("⏳  Loading models...")
    m = {}
    files = {
        "scaler":      "scaler.pkl",
        "nzv_cols":    "nzv_cols.pkl",
        "score_stats": "score_stats.pkl",
        "if_model":    "isolation_forest.pkl",
        "xgb_det":     "xgb_detector.pkl",
        "xgb_clf":     "attack_classifier.pkl",
        "le_attack":   "attack_label_encoder.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            print(f"  ❌  Missing: {fname}  — place it in {model_dir}")
            sys.exit(1)
        m[key] = joblib.load(path)
        print(f"  ✅  {fname}")

    ae_path = os.path.join(model_dir, "autoencoder.keras")
    if not os.path.exists(ae_path):
        print(f"  ❌  Missing: autoencoder.keras")
        sys.exit(1)
    m["ae"] = tf.keras.models.load_model(ae_path)
    print(f"  ✅  autoencoder.keras")

    m["IF_MEDIAN"] = m["score_stats"]["if_median"]
    m["IF_IQR"]    = m["score_stats"]["if_iqr"]
    m["AE_MEDIAN"] = m["score_stats"]["ae_median"]
    m["AE_IQR"]    = m["score_stats"]["ae_iqr"]
    print("✅  All models loaded\n")
    return m

M = load_models(MODEL_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FLOW AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────
class FlowRecord:
    def __init__(self, key, ts):
        self.key          = key
        self.start_ts     = ts
        self.last_ts      = ts
        self.fwd_pkts     = []
        self.bwd_pkts     = []
        self.fwd_iat      = []
        self.bwd_iat      = []
        self.all_iat      = []
        self.prev_fwd_ts  = ts
        self.prev_bwd_ts  = ts
        self.prev_ts      = ts
        self.flags        = collections.Counter()
        self.init_fwd_win = 0
        self.init_bwd_win = 0
        self._fwd_win_set = False
        self._bwd_win_set = False
        self.active_start = ts
        self.active_list  = []
        self.idle_list    = []

    def add_packet(self, size, ts, direction, tcp_flags, win_size):
        gap = ts - self.prev_ts
        if gap > 1.0:
            self.idle_list.append(gap)
            self.active_list.append(ts - self.active_start)
            self.active_start = ts
        self.prev_ts = ts
        self.last_ts = ts
        self.all_iat.append(gap)

        if direction == "fwd":
            self.fwd_iat.append(ts - self.prev_fwd_ts)
            self.prev_fwd_ts = ts
            self.fwd_pkts.append((size, ts))
            if not self._fwd_win_set and win_size:
                self.init_fwd_win = win_size
                self._fwd_win_set = True
        else:
            self.bwd_iat.append(ts - self.prev_bwd_ts)
            self.prev_bwd_ts = ts
            self.bwd_pkts.append((size, ts))
            if not self._bwd_win_set and win_size:
                self.init_bwd_win = win_size
                self._bwd_win_set = True

        if tcp_flags:
            for fname, mask in [("PSH",0x08),("URG",0x20),("ACK",0x10),
                                 ("SYN",0x02),("FIN",0x01),("RST",0x04)]:
                if tcp_flags & mask:
                    self.flags[fname] += 1

    def _stats(self, lst):
        if not lst: return 0.0, 0.0, 0.0, 0.0
        a = np.array(lst, dtype=np.float64)
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    def to_feature_dict(self):
        eps  = 1e-8
        dur  = max(self.last_ts - self.start_ts, eps)
        fwd_s = [p[0] for p in self.fwd_pkts]
        bwd_s = [p[0] for p in self.bwd_pkts]
        all_s = fwd_s + bwd_s
        fwd_mean,fwd_std,fwd_min,fwd_max = self._stats(fwd_s)
        bwd_mean,bwd_std,bwd_min,bwd_max = self._stats(bwd_s)
        all_mean,all_std,all_min,all_max = self._stats(all_s)
        fim,fis,fimn,fimax = self._stats(self.fwd_iat)
        bim,bis,bimn,bimax = self._stats(self.bwd_iat)
        aim,ais,aimn,aimax = self._stats(self.all_iat)
        acm,acs,acmn,acmax = self._stats(self.active_list)
        idm,ids,idmn,idmax = self._stats(self.idle_list)
        n_fwd = len(fwd_s); n_bwd = len(bwd_s); n_pkt = n_fwd + n_bwd
        fb = sum(fwd_s); bb = sum(bwd_s)
        nsf = max(1, len(self.idle_list)+1)
        return {
            "Flow Duration":               dur*1e6,
            "Total Fwd Packets":           n_fwd,
            "Total Backward Packets":      n_bwd,
            "Total Length of Fwd Packets": fb,
            "Total Length of Bwd Packets": bb,
            "Fwd Packet Length Max":       fwd_max,
            "Fwd Packet Length Min":       fwd_min,
            "Fwd Packet Length Mean":      fwd_mean,
            "Fwd Packet Length Std":       fwd_std,
            "Bwd Packet Length Max":       bwd_max,
            "Bwd Packet Length Min":       bwd_min,
            "Bwd Packet Length Mean":      bwd_mean,
            "Bwd Packet Length Std":       bwd_std,
            "Flow Bytes/s":                (fb+bb)/dur,
            "Flow Packets/s":              n_pkt/dur,
            "Flow IAT Mean":               aim,
            "Flow IAT Std":                ais,
            "Flow IAT Max":                aimax,
            "Flow IAT Min":                aimn,
            "Fwd IAT Total":               sum(self.fwd_iat),
            "Fwd IAT Mean":                fim,
            "Fwd IAT Std":                 fis,
            "Fwd IAT Max":                 fimax,
            "Fwd IAT Min":                 fimn,
            "Bwd IAT Total":               sum(self.bwd_iat),
            "Bwd IAT Mean":                bim,
            "Bwd IAT Std":                 bis,
            "Bwd IAT Max":                 bimax,
            "Bwd IAT Min":                 bimn,
            "Fwd PSH Flags":               self.flags["PSH"],
            "Bwd PSH Flags":               0,
            "Fwd URG Flags":               0,
            "Bwd URG Flags":               0,
            "Fwd Header Length":           n_fwd*20,
            "Bwd Header Length":           n_bwd*20,
            "Fwd Packets/s":               n_fwd/dur,
            "Bwd Packets/s":               n_bwd/dur,
            "Min Packet Length":           all_min,
            "Max Packet Length":           all_max,
            "Packet Length Mean":          all_mean,
            "Packet Length Std":           all_std,
            "Packet Length Variance":      all_std**2,
            "FIN Flag Count":              self.flags["FIN"],
            "SYN Flag Count":              self.flags["SYN"],
            "RST Flag Count":              self.flags["RST"],
            "PSH Flag Count":              self.flags["PSH"],
            "ACK Flag Count":              self.flags["ACK"],
            "URG Flag Count":              self.flags["URG"],
            "CWE Flag Count":              0,
            "ECE Flag Count":              0,
            "Down/Up Ratio":               bb/(fb+eps),
            "Average Packet Size":         all_mean,
            "Avg Fwd Segment Size":        fwd_mean,
            "Avg Bwd Segment Size":        bwd_mean,
            "Fwd Avg Bytes/Bulk":          0,
            "Fwd Avg Packets/Bulk":        0,
            "Fwd Avg Bulk Rate":           0,
            "Bwd Avg Bytes/Bulk":          0,
            "Bwd Avg Packets/Bulk":        0,
            "Bwd Avg Bulk Rate":           0,
            "Subflow Fwd Packets":         n_fwd//nsf,
            "Subflow Fwd Bytes":           fb//nsf,
            "Subflow Bwd Packets":         n_bwd//nsf,
            "Subflow Bwd Bytes":           bb//nsf,
            "Init_Win_bytes_forward":      self.init_fwd_win,
            "Init_Win_bytes_backward":     self.init_bwd_win,
            "act_data_pkt_fwd":            n_fwd,
            "min_seg_size_forward":        fwd_min,
            "Active Mean":                 acm,
            "Active Std":                  acs,
            "Active Max":                  acmax,
            "Active Min":                  acmn,
            "Idle Mean":                   idm,
            "Idle Std":                    ids,
            "Idle Max":                    idmax,
            "Idle Min":                    idmn,
        }


class FlowAggregator:
    def __init__(self):
        self.flows    = {}
        self.lock     = threading.Lock()
        self.exported = collections.deque(maxlen=10000)

    def process_packet(self, pkt):
        if not pkt.haslayer(IP): return
        ip    = pkt[IP]; proto = ip.proto
        ts    = float(pkt.time)
        src_ip, dst_ip = ip.src, ip.dst
        src_port = dst_port = 0
        tcp_flags = win_size = None
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport; dst_port = pkt[TCP].dport
            tcp_flags = int(pkt[TCP].flags); win_size = pkt[TCP].window
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport; dst_port = pkt[UDP].dport

        if (src_ip, src_port) < (dst_ip, dst_port):
            key = (src_ip, dst_ip, src_port, dst_port, proto); direction = "fwd"
        else:
            key = (dst_ip, src_ip, dst_port, src_port, proto); direction = "bwd"

        with self.lock:
            if key not in self.flows:
                self.flows[key] = FlowRecord(key, ts)
            self.flows[key].add_packet(len(pkt), ts, direction, tcp_flags, win_size)

    def export_stale(self, now):
        with self.lock:
            done = [k for k,f in self.flows.items() if now - f.last_ts > FLOW_TIMEOUT_SEC]
            for k in done:
                self.exported.append((k, self.flows.pop(k).to_feature_dict()))

    def drain(self):
        out = []
        while self.exported:
            try:    out.append(self.exported.popleft())
            except: break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE PIPELINE — Adaptive Baseline
# ─────────────────────────────────────────────────────────────────────────────
def _engineer(d):
    eps = 1e-8
    if "Total Fwd Packets" in d and "Total Backward Packets" in d:
        d["fwd_bwd_pkt_ratio"]  = d["Total Fwd Packets"] / (d["Total Backward Packets"] + eps)
    if "Total Length of Fwd Packets" in d and "Total Length of Bwd Packets" in d:
        d["fwd_bwd_byte_ratio"] = d["Total Length of Fwd Packets"] / (d["Total Length of Bwd Packets"] + eps)
    for col in ["Flow Duration","Total Length of Fwd Packets",
                "Total Length of Bwd Packets","Flow Bytes/s","Flow Packets/s"]:
        if col in d:
            d[f"log1p_{col.replace('/','_per_')}"] = math.log1p(max(d[col], 0))
    if "Packet Length Mean" in d and "Packet Length Std" in d:
        d["pkt_len_cv"] = d["Packet Length Std"] / (d["Packet Length Mean"] + eps)
    if "Active Mean" in d and "Idle Mean" in d:
        d["active_idle_ratio"] = d["Active Mean"] / (d["Idle Mean"] + eps)
    if "PSH Flag Count" in d and "Total Fwd Packets" in d:
        d["psh_urg_density"] = (d.get("PSH Flag Count",0) + d.get("URG Flag Count",0)) / \
                                (d["Total Fwd Packets"] + eps)
    return d

def predict_flow(feat_dict, src_ip="?", dst_ip="?"):
    """
    Two-layer detection:
      Layer 1 — Flow rate detector: catches DoS/flood by counting
                same src→dst flows per second. Fast, catches volume attacks.
      Layer 2 — ML detector: IF + AE + XGBoost against YOUR baseline.
                Catches stealth/low-volume attacks.
    """
    _flow_count["n"] += 1

    # ── Layer 1: Flow rate check (catches DoS floods immediately) ────────────
    is_flood, flow_rate_count = _check_flow_rate(src_ip, dst_ip)
    if is_flood and _baseline["ready"]:
        print(f"  🌊 FLOOD DETECTED {src_ip}→{dst_ip} "
              f"({flow_rate_count} flows in {RATE_WINDOW_SEC}s)")

    # ── Feature prep ────────────────────────────────────────────────────────
    feat_dict = _engineer(dict(feat_dict))
    row = pd.to_numeric(pd.Series(feat_dict), errors="coerce").fillna(0)
    nzv = M["nzv_cols"]
    row = row.drop(labels=[c for c in nzv if c in row.index], errors="ignore")
    scaler = M["scaler"]
    expected = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else row.index
    row = row.reindex(expected, fill_value=0)
    X = np.clip(scaler.transform(row.values.reshape(1,-1)), -10, 10).astype(np.float32)

    # ── Raw scores ──────────────────────────────────────────────────────────
    if_score = float(-M["if_model"].score_samples(X)[0])
    ae_pred  = M["ae"].predict(X, verbose=0)
    ae_error = float(np.mean((X - ae_pred)**2))
    ae_log   = math.log1p(ae_error)
    if_norm  = (if_score - M["IF_MEDIAN"]) / M["IF_IQR"]
    ae_norm  = (ae_error - M["AE_MEDIAN"]) / M["AE_IQR"]
    X_h      = np.hstack([X, [[if_norm]], [[ae_norm]], [[ae_log]]]).astype(np.float32)
    xgb_prob = float(M["xgb_det"].predict_proba(X_h)[0, 1])

    # ── WARMUP ───────────────────────────────────────────────────────────────
    if not _baseline["ready"]:
        _baseline["xgb_vals"].append(xgb_prob)
        _baseline["if_vals"].append(if_score)
        _baseline["ae_vals"].append(ae_error)
        remaining = WARMUP_FLOWS - _flow_count["n"]
        print(f"  ⏳ WARMUP {_flow_count['n']}/{WARMUP_FLOWS} "
              f"xgb={xgb_prob:.4f} if={if_score:.4f} ae={ae_error:.4f} "
              f"({remaining} to go)")
        if _flow_count["n"] >= WARMUP_FLOWS:
            _compute_baseline()
        return {
            "is_attack":   False,
            "attack_prob": 0.0,
            "attack_type": "WARMING UP",
            "if_score":    round(if_score, 4),
            "ae_error":    round(ae_error, 6),
            "xgb_raw":     round(xgb_prob, 4),
        }

    # ── DETECTION ────────────────────────────────────────────────────────────
    xgb_alert  = xgb_prob >= _baseline["xgb_thresh"]
    if_alert   = if_score >= _baseline["if_thresh"]
    ae_alert   = ae_error >= _baseline["ae_thresh"]

    # Attack if: flood OR XGBoost fires OR both anomaly detectors agree
    is_attack  = is_flood or xgb_alert or (if_alert and ae_alert)

    # ── Display score ────────────────────────────────────────────────────────
    if is_attack:
        if is_flood:
            # Scale flood score by rate
            flood_score = min(1.0, flow_rate_count / (RATE_THRESHOLD * 3))
            display_score = 0.6 + 0.4 * flood_score
        else:
            xgb_e = max(0, (xgb_prob - _baseline["xgb_thresh"]) / max(_baseline["xgb_thresh"], 0.01))
            if_e  = max(0, (if_score  - _baseline["if_thresh"])  / max(_baseline["if_thresh"],  0.01))
            ae_e  = max(0, (ae_error  - _baseline["ae_thresh"])  / max(_baseline["ae_thresh"],  0.01))
            display_score = 0.6 + 0.4 * float(np.clip(xgb_e*0.5 + if_e*0.3 + ae_e*0.2, 0, 1))
    else:
        xgb_f = xgb_prob / max(_baseline["xgb_thresh"], 1e-8)
        if_f  = if_score  / max(_baseline["if_thresh"],  1e-8)
        ae_f  = ae_error  / max(_baseline["ae_thresh"],  1e-8)
        display_score = float(np.clip(xgb_f*0.5 + if_f*0.3 + ae_f*0.2, 0, 0.25))

    display_score = round(float(display_score), 4)

    # ── Attack type Stage 2 ──────────────────────────────────────────────────
    attack_type = "BENIGN"
    if is_attack:
        if is_flood:
            attack_type = "DoS Flood"
        else:
            try:
                atk_pred    = M["xgb_clf"].predict(X_h)[0]
                attack_type = M["le_attack"].inverse_transform([atk_pred])[0]
            except:
                attack_type = "UNKNOWN THREAT"

    # ── Terminal log ─────────────────────────────────────────────────────────
    rate_str = f" rate={flow_rate_count}" if is_flood else ""
    flags = (f"xgb={'🔴' if xgb_alert else '⚪'}{xgb_prob:.4f} "
             f"if={'🔴' if if_alert else '⚪'}{if_score:.4f} "
             f"ae={'🔴' if ae_alert else '⚪'}{ae_error:.4f}"
             f"{'🌊 FLOOD' if is_flood else ''}{rate_str}")
    print(f"  {flags} → {'🚨 '+attack_type if is_attack else '✅ BENIGN'}")

    return {
        "is_attack":   is_attack,
        "attack_prob": display_score,
        "attack_type": attack_type,
        "if_score":    round(if_score, 4),
        "ae_error":    round(ae_error, 6),
        "xgb_raw":     round(xgb_prob, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
STATE = {
    "detections":   collections.deque(maxlen=MAX_ROWS),
    "total":        0,
    "attack":       0,
    "benign":       0,
    "attack_types": collections.Counter(),
    "top_talkers":  collections.Counter(),
    "running":      False,
    "lock":         threading.Lock(),
    "alerts":       collections.deque(maxlen=50),
}

def record(result, flow_key):
    src_ip    = flow_key[0] if flow_key else "?"
    dst_ip    = flow_key[1] if flow_key else "?"
    proto_num = flow_key[4] if flow_key and len(flow_key) > 4 else 0
    proto     = "TCP" if proto_num == 6 else "UDP" if proto_num == 17 else "OTHER"
    port      = flow_key[3] if flow_key and len(flow_key) > 3 else 0

    row = {
        "time":      datetime.now().strftime("%H:%M:%S"),
        "src":       src_ip,
        "dst":       dst_ip,
        "proto":     proto,
        "port":      port,
        "is_attack": result["is_attack"],
        "label":     result["attack_type"],
        "prob":      f"{result['attack_prob']:.3f}",
        "if_score":  f"{result['if_score']:.4f}",
        "ae_error":  f"{result['ae_error']:.6f}",
        "xgb_raw":   f"{result['xgb_raw']:.4f}",
    }

    with STATE["lock"]:
        STATE["total"] += 1
        if result["is_attack"]:
            STATE["attack"] += 1
            STATE["attack_types"][result["attack_type"]] += 1
            STATE["top_talkers"][src_ip] += 1
            STATE["alerts"].appendleft({
                "time": row["time"],
                "src":  src_ip,
                "type": result["attack_type"],
                "prob": f"{result['attack_prob']:.3f}",
            })
        else:
            STATE["benign"] += 1
        STATE["detections"].appendleft(row)

    return row  # return so inference thread can emit it

# ─────────────────────────────────────────────────────────────────────────────
# 6. BACKGROUND THREADS
# ─────────────────────────────────────────────────────────────────────────────
aggregator = FlowAggregator()

def sniffer_thread(iface):
    # BPF filter — captures only real IP traffic, excludes:
    #   - loopback (127.0.0.0/8)
    #   - multicast (224.0.0.0/4)
    #   - link-local (169.254.0.0/16)
    BPF_FILTER = (
        "ip and "
        "not src net 127.0.0.0/8 and "
        "not dst net 127.0.0.0/8 and "
        "not src net 224.0.0.0/4 and "
        "not dst net 224.0.0.0/4 and "
        "not src net 169.254.0.0/16 and "
        "not dst net 169.254.0.0/16"
    )
    def cb(pkt):
        if STATE["running"]:
            aggregator.process_packet(pkt)
    print(f"🔍  Sniffing on: {iface}")
    print(f"🔍  Filter: {BPF_FILTER}")
    sniff(iface=iface, prn=cb, store=False,
          filter=BPF_FILTER,
          stop_filter=lambda _: not STATE["running"])

def inference_thread(sio_ref):
    while STATE["running"]:
        aggregator.export_stale(time.time())
        for key, feat in aggregator.drain():
            try:
                src_ip = key[0] if key else "?"
                dst_ip = key[1] if key else "?"
                result = predict_flow(feat, src_ip, dst_ip)
                row    = record(result, key)
                sio_ref.emit("detection", row)
            except Exception as e:
                print(f"  ⚠ Inference: {e}")
        time.sleep(2)

def state_push_thread(sio_ref):
    while STATE["running"]:
        with STATE["lock"]:
            payload = {
                "total":       STATE["total"],
                "attack":      STATE["attack"],
                "benign":      STATE["benign"],
                "attack_types":dict(STATE["attack_types"]),
                "top_talkers": dict(STATE["top_talkers"].most_common(5)),
                "alerts":      list(STATE["alerts"])[:5],
            }
        sio_ref.emit("state", payload)
        sio_ref.sleep(2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. FLASK + DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "nids-college-2025"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIDS — Live Network Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#060a0f;--panel:#0c1117;--border:#1a2332;--border2:#243040;
      --text:#cdd9e5;--muted:#546e7a;--cyan:#00e5ff;--green:#00e676;
      --red:#ff1744;--orange:#ff6d00;--yellow:#ffd600;
      --blue:#2979ff;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);
     display:flex;flex-direction:column;}

/* scanline */
body::before{content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,
  rgba(0,229,255,.01) 2px,rgba(0,229,255,.01) 4px);
  pointer-events:none;z-index:999;}

/* ── Header ── */
header{flex-shrink:0;display:flex;align-items:center;gap:12px;
  padding:8px 16px;background:#060a0f;border-bottom:1px solid var(--border2);}
.logo{font-size:17px;font-weight:800;letter-spacing:.06em;
  background:linear-gradient(90deg,var(--cyan),var(--blue));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.logo span{-webkit-text-fill-color:var(--red);}
.live-badge{display:flex;align-items:center;gap:6px;font-family:var(--mono);
  font-size:10px;color:var(--green);border:1px solid var(--green);
  border-radius:3px;padding:2px 8px;}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);
  box-shadow:0 0 5px var(--green);animation:blink 1.2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.threat-level{font-family:var(--mono);font-size:10px;padding:2px 10px;
  border-radius:3px;border:1px solid;}
.threat-level.low {color:var(--green); border-color:var(--green);}
.threat-level.med {color:var(--yellow);border-color:var(--yellow);}
.threat-level.high{color:var(--red);   border-color:var(--red);animation:blink .8s infinite;}
.header-ts{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--muted);}

/* ── Body layout: fixed height rows ── */
.body-wrap{flex:1;display:flex;flex-direction:column;gap:6px;padding:6px 12px;overflow:hidden;}

/* KPI row — compact */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;flex-shrink:0;}
.kpi{background:var(--panel);border:1px solid var(--border2);border-radius:6px;
  padding:8px 12px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--cyan));}
.kpi .val{font-family:var(--mono);font-size:22px;font-weight:700;
  color:var(--accent,var(--cyan));text-shadow:0 0 12px var(--accent,var(--cyan));}
.kpi label{font-size:9px;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);display:block;margin-top:2px;}

/* Middle row: charts + sidebar */
.mid-row{display:grid;grid-template-columns:200px 1fr 180px;gap:6px;flex-shrink:0;height:160px;}
.chart-card{background:var(--panel);border:1px solid var(--border2);border-radius:6px;padding:10px;overflow:hidden;}
.card-title{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin-bottom:8px;display:flex;align-items:center;gap:6px;}
.card-title::before{content:'';width:2px;height:11px;background:var(--cyan);border-radius:2px;flex-shrink:0;}

/* Alerts sidebar */
.alert-list{list-style:none;display:flex;flex-direction:column;gap:4px;overflow:hidden;}
.alert-item{background:#120808;border:1px solid #3d1010;border-radius:4px;
  padding:4px 8px;font-family:var(--mono);font-size:9px;line-height:1.4;}
.alert-item .a-type{color:var(--red);font-weight:700;}
.alert-item .a-src{color:var(--orange);}
.alert-item .a-time{color:var(--muted);display:block;}

/* Top talkers */
.talker-row{display:flex;align-items:center;gap:6px;margin-bottom:5px;}
.talker-ip{font-family:var(--mono);font-size:9px;color:var(--orange);width:100px;flex-shrink:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.talker-bar-track{flex:1;height:4px;background:var(--border2);border-radius:2px;}
.talker-bar-fill{height:100%;background:var(--red);border-radius:2px;transition:width .4s;}
.talker-count{font-family:var(--mono);font-size:9px;color:var(--muted);width:22px;text-align:right;}

/* Feed table — scrollable */
.feed-wrap{flex:1;display:flex;flex-direction:column;min-height:0;}
.feed-header{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:5px;flex-shrink:0;}
.feed-header h2{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  display:flex;align-items:center;gap:6px;}
.feed-header h2::before{content:'';width:2px;height:11px;background:var(--cyan);border-radius:2px;}
.filter-btns{display:flex;gap:4px;}
.filter-btn{font-family:var(--mono);font-size:9px;padding:2px 8px;border-radius:3px;
  cursor:pointer;border:1px solid var(--border2);background:transparent;
  color:var(--muted);transition:all .15s;}
.filter-btn.active,.filter-btn:hover{border-color:var(--cyan);color:var(--cyan);}
.table-scroll{flex:1;overflow-y:auto;overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px;}
thead tr{border-bottom:1px solid var(--border2);position:sticky;top:0;background:var(--bg);z-index:2;}
th{padding:5px 10px;text-align:left;font-size:9px;text-transform:uppercase;
  letter-spacing:.08em;color:var(--muted);font-weight:400;}
td{padding:5px 10px;border-bottom:1px solid var(--border);}
tbody tr{transition:background .12s;}
tbody tr:hover{background:#0f1820;}
tbody tr.atk-row{background:rgba(255,23,68,.04);}
.badge{display:inline-flex;align-items:center;gap:4px;padding:1px 8px;
  border-radius:3px;font-size:9px;letter-spacing:.05em;font-weight:600;}
.badge.atk{background:rgba(255,23,68,.12);color:var(--red);border:1px solid rgba(255,23,68,.3);}
.badge.ok {background:rgba(0,230,118,.08);color:var(--green);border:1px solid rgba(0,230,118,.2);}
.badge::before{content:'●';font-size:6px;}
.prob-bar{display:flex;align-items:center;gap:6px;}
.prob-bar-track{flex:1;height:3px;background:var(--border2);border-radius:2px;}
.prob-bar-fill{height:100%;border-radius:2px;transition:width .3s;}
@keyframes rowflash{0%{background:rgba(0,229,255,.1);}100%{background:transparent;}}
.flash{animation:rowflash .6s ease-out;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}

/* ── ATTACK ALERT OVERLAY ─────────────────────────────────── */
#attack-overlay{
  display:none;position:fixed;inset:0;z-index:9000;
  background:rgba(0,0,0,.85);backdrop-filter:blur(5px);
  flex-direction:column;align-items:center;justify-content:center;gap:16px;
}
#attack-overlay.active{display:flex;}
@keyframes borderPulse{
  0%,100%{box-shadow:inset 0 0 0 5px rgba(255,23,68,.9);}
  50%{box-shadow:inset 0 0 0 5px rgba(255,23,68,.15);}
}
#attack-overlay.active::before{
  content:'';position:fixed;inset:0;pointer-events:none;
  animation:borderPulse .65s infinite;
}
.ov-icon{font-size:64px;animation:iconPulse .55s infinite alternate;}
@keyframes iconPulse{0%{transform:scale(1);}100%{transform:scale(1.18);}}
.ov-title{
  font-family:var(--mono);font-size:40px;font-weight:700;letter-spacing:.1em;
  color:var(--red);text-shadow:0 0 30px var(--red),0 0 60px rgba(255,23,68,.4);
  animation:titleFlash .5s infinite alternate;
}
@keyframes titleFlash{0%{opacity:1;}100%{opacity:.65;}}
.ov-sub{font-family:var(--mono);font-size:14px;color:#ff8a80;letter-spacing:.07em;text-align:center;}
.ov-type{
  font-family:var(--mono);font-size:20px;color:var(--orange);
  background:rgba(255,109,0,.1);border:1px solid var(--orange);
  border-radius:6px;padding:7px 24px;letter-spacing:.05em;
}
.ov-src{font-family:var(--mono);font-size:12px;color:var(--muted);}
.ov-timer{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:4px;}
.ov-dismiss{
  margin-top:8px;font-family:var(--mono);font-size:10px;color:var(--muted);
  border:1px solid var(--border2);border-radius:4px;padding:4px 14px;
  cursor:pointer;background:transparent;transition:all .2s;
}
.ov-dismiss:hover{border-color:var(--cyan);color:var(--cyan);}
body.under-attack header{background:rgba(255,23,68,.07);border-bottom-color:rgba(255,23,68,.4);}
</style>
</head>
<body>

<!-- ── ATTACK ALERT OVERLAY ── -->
<div id="attack-overlay">
  <div class="ov-icon">🚨</div>
  <div class="ov-title">ATTACK DETECTED</div>
  <div class="ov-sub">Network intrusion in progress</div>
  <div class="ov-type" id="ov-type">—</div>
  <div class="ov-src"  id="ov-src">—</div>
  <div class="ov-timer" id="ov-timer"></div>
  <button class="ov-dismiss" onclick="dismissOverlay()">DISMISS (auto-clears when attack stops)</button>
</div>

<header>
  <div class="logo">NID<span>S</span></div>
  <div class="live-badge"><div class="live-dot"></div>LIVE</div>
  <div id="threat-badge" class="threat-level low">THREAT: LOW</div>
  <div class="header-ts" id="hts">--:--:--</div>
</header>

<div class="body-wrap">

  <!-- KPI row -->
  <div class="kpi-row">
    <div class="kpi" style="--accent:var(--cyan)">
      <div class="val" id="k-total">0</div><label>Flows Analysed</label>
    </div>
    <div class="kpi" style="--accent:var(--red)">
      <div class="val" id="k-attack">0</div><label>Attacks Detected</label>
    </div>
    <div class="kpi" style="--accent:var(--green)">
      <div class="val" id="k-benign">0</div><label>Benign Flows</label>
    </div>
    <div class="kpi" style="--accent:var(--yellow)">
      <div class="val" id="k-rate">0%</div><label>Attack Rate</label>
    </div>
  </div>

  <!-- Middle row: donut | timeline | alerts+talkers -->
  <div class="mid-row">
    <div class="chart-card">
      <div class="card-title">Attack Types</div>
      <canvas id="c-donut"></canvas>
    </div>
    <div class="chart-card">
      <div class="card-title">Detection Timeline</div>
      <canvas id="c-line"></canvas>
    </div>
    <div class="chart-card" style="display:flex;flex-direction:column;gap:8px;">
      <div>
        <div class="card-title">Recent Alerts</div>
        <ul class="alert-list" id="alert-list">
          <li style="color:var(--muted);font-family:var(--mono);font-size:9px">Waiting...</li>
        </ul>
      </div>
      <div>
        <div class="card-title">Top Attackers</div>
        <div id="talkers"></div>
      </div>
    </div>
  </div>

  <!-- Feed table — takes remaining space -->
  <div class="feed-wrap">
    <div class="feed-header">
      <h2>Live Alert Feed</h2>
      <div class="filter-btns">
        <button class="filter-btn active" onclick="setFilter('all',this)">ALL</button>
        <button class="filter-btn" onclick="setFilter('atk',this)">ATTACKS</button>
        <button class="filter-btn" onclick="setFilter('ok',this)">BENIGN</button>
      </div>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th>Time</th><th>Source IP</th><th>Dest IP</th><th>Proto</th>
          <th>Port</th><th>Label</th><th>Hybrid Score</th><th>XGB Raw</th><th>IF Score</th><th>AE Error</th>
        </tr></thead>
        <tbody id="feed"></tbody>
      </table>
    </div>
  </div>

</div><!-- body-wrap -->

<script>
const socket = io();
const PAL = ['#ff1744','#2979ff','#00e676','#ff6d00','#d500f9','#ffd600','#00e5ff','#ff4081'];
let filterMode = 'all';
let flowCount  = 0;
let maxTalker  = 1;

const donutChart = new Chart(document.getElementById('c-donut'),{
  type:'doughnut',
  data:{labels:[],datasets:[{data:[],borderWidth:0,hoverOffset:4}]},
  options:{cutout:'65%',responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'bottom',labels:{color:'#546e7a',
      font:{size:9,family:"'Share Tech Mono'"},boxWidth:8,padding:5}}}}
});

const lineChart = new Chart(document.getElementById('c-line'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Attack Prob',data:[],borderColor:'#ff1744',backgroundColor:'rgba(255,23,68,.08)',
     fill:true,tension:.4,pointRadius:0,borderWidth:1.5,yAxisID:'yp'},
    {label:'Flows',data:[],borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,.04)',
     fill:true,tension:.4,pointRadius:0,borderWidth:1.5,yAxisID:'yc'},
  ]},
  options:{animation:false,responsive:true,maintainAspectRatio:false,
    scales:{
      x:{ticks:{color:'#546e7a',maxTicksLimit:6,font:{size:8}},grid:{color:'#1a2332'}},
      yp:{position:'left',min:0,max:1,ticks:{color:'#ff1744',font:{size:8}},grid:{color:'#1a2332'}},
      yc:{position:'right',min:0,ticks:{color:'#00e5ff',font:{size:8}},grid:{drawOnChartArea:false}},
    },
    plugins:{legend:{labels:{color:'#546e7a',font:{size:9},boxWidth:8,padding:6}}}}
});

socket.on('state', d => {
  document.getElementById('k-total').textContent  = d.total.toLocaleString();
  document.getElementById('k-attack').textContent = d.attack.toLocaleString();
  document.getElementById('k-benign').textContent = d.benign.toLocaleString();
  const rate = d.total > 0 ? (d.attack/d.total*100).toFixed(1) : 0;
  document.getElementById('k-rate').textContent   = rate + '%';
  document.getElementById('hts').textContent      = new Date().toLocaleTimeString();

  const tb = document.getElementById('threat-badge');
  if      (rate >= 50){ tb.textContent='⚠ THREAT: HIGH'; tb.className='threat-level high'; }
  else if (rate >= 20){ tb.textContent='THREAT: MED';     tb.className='threat-level med';  }
  else                { tb.textContent='THREAT: LOW';     tb.className='threat-level low';  }

  const labels = Object.keys(d.attack_types);
  donutChart.data.labels = labels;
  donutChart.data.datasets[0].data = Object.values(d.attack_types);
  donutChart.data.datasets[0].backgroundColor = PAL.slice(0,labels.length);
  donutChart.update('none');

  const al = document.getElementById('alert-list');
  if (d.alerts && d.alerts.length > 0) {
    al.innerHTML = d.alerts.slice(0,3).map(a =>
      `<li class="alert-item">
        <span class="a-time">${a.time}</span>
        <span class="a-type">${a.type}</span> <span class="a-src">${a.src}</span>
       </li>`).join('');
  }

  const tDiv = document.getElementById('talkers');
  const talkers = Object.entries(d.top_talkers || {});
  if (talkers.length > 0) {
    maxTalker = Math.max(...talkers.map(t=>t[1]), 1);
    tDiv.innerHTML = talkers.slice(0,3).map(([ip,cnt]) =>
      `<div class="talker-row">
        <div class="talker-ip">${ip}</div>
        <div class="talker-bar-track">
          <div class="talker-bar-fill" style="width:${Math.round(cnt/maxTalker*100)}%"></div>
        </div>
        <div class="talker-count">${cnt}</div>
       </div>`).join('');
  }
});

socket.on('detection', row => {
  flowCount++;
  if (flowCount % 3 === 0) {
    const d = lineChart.data;
    if (d.labels.length > 40){ d.labels.shift(); d.datasets[0].data.shift(); d.datasets[1].data.shift(); }
    d.labels.push(row.time);
    d.datasets[0].data.push(parseFloat(row.prob));
    d.datasets[1].data.push(flowCount % 10);
    lineChart.update('none');
  }

  // ── Attack overlay logic ──────────────────────────────────────────────────
  if (row.is_attack) {
    lastAttackTime = Date.now();
    if (!overlayDismissed) {
      showOverlay(row.label, row.src, row.dst);
    }
  }

  if (filterMode === 'atk' && !row.is_attack) return;
  if (filterMode === 'ok'  &&  row.is_attack) return;

  const pct  = Math.round(parseFloat(row.prob)*100);
  const pclr = row.is_attack ? '#ff1744' : '#00e676';
  const tbody = document.getElementById('feed');
  const tr    = document.createElement('tr');
  tr.className = (row.is_attack ? 'atk-row ' : '') + 'flash';
  tr.innerHTML = `
    <td>${row.time}</td><td>${row.src}</td><td>${row.dst}</td>
    <td>${row.proto}</td><td>${row.port}</td>
    <td><span class="badge ${row.is_attack?'atk':'ok'}">${row.label}</span></td>
    <td><div class="prob-bar">
      <span style="width:32px;color:${pclr};font-size:10px">${pct}%</span>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%;background:${pclr}"></div></div>
    </div></td>
    <td style="color:var(--muted)">${row.xgb_raw}</td>
    <td>${row.if_score}</td><td>${row.ae_error}</td>`;
  tbody.prepend(tr);
  if (tbody.children.length > 100) tbody.removeChild(tbody.lastChild);
});

// ── Overlay functions ─────────────────────────────────────────────────────
let lastAttackTime   = 0;
let overlayDismissed = false;
let overlayInterval  = null;
const ATTACK_CLEAR_SEC = 15;   // seconds of no attacks before overlay clears

function showOverlay(type, src, dst) {
  const ov = document.getElementById('attack-overlay');
  document.getElementById('ov-type').textContent = '⚠ ' + type;
  document.getElementById('ov-src').textContent  = 'Source: ' + src + '  →  ' + dst;
  ov.classList.add('active');
  document.body.classList.add('under-attack');
  // start countdown timer display
  if (overlayInterval) clearInterval(overlayInterval);
  overlayInterval = setInterval(() => {
    const secSince = Math.floor((Date.now() - lastAttackTime) / 1000);
    const remaining = Math.max(0, ATTACK_CLEAR_SEC - secSince);
    const el = document.getElementById('ov-timer');
    if (remaining > 0) {
      el.textContent = `Auto-clearing in ${remaining}s after last attack flow`;
    } else {
      // attack has stopped — clear overlay
      clearOverlay();
    }
  }, 1000);
}

function clearOverlay() {
  const ov = document.getElementById('attack-overlay');
  ov.classList.remove('active');
  document.body.classList.remove('under-attack');
  overlayDismissed = false;
  if (overlayInterval) { clearInterval(overlayInterval); overlayInterval = null; }
}

function dismissOverlay() {
  overlayDismissed = true;
  clearOverlay();
  // re-enable overlay if new attack comes after 30s
  setTimeout(() => { overlayDismissed = false; }, 30000);
}

function setFilter(mode, btn) {
  filterMode = mode;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

setInterval(() => socket.emit('get_state'), 2000);
socket.emit('get_state');
</script>
</body></html>
"""

@app.route("/")
def index(): return render_template_string(HTML)

@sio.on("get_state")
def push_state():
    with STATE["lock"]:
        sio.emit("state", {
            "total":       STATE["total"],
            "attack":      STATE["attack"],
            "benign":      STATE["benign"],
            "attack_types":dict(STATE["attack_types"]),
            "top_talkers": dict(STATE["top_talkers"].most_common(5)),
            "alerts":      list(STATE["alerts"])[:5],
        })

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Auto-detect interface — prefer WiFi (en0/wlan0) over loopback
    iface = NETWORK_INTERFACE
    if iface is None:
        all_ifaces = get_if_list()
        # Priority order: en0 (Mac WiFi) → wlan0 (Linux) → eth0 → first non-loopback
        for candidate in ["en0", "en1", "wlan0", "wlan1", "eth0", "ens33"]:
            if candidate in all_ifaces:
                iface = candidate
                break
        if iface is None:
            # fallback — skip lo0/lo/loopback
            skip = {"lo0", "lo", "any", "loopback", "localhost"}
            iface = next((i for i in all_ifaces if i not in skip), all_ifaces[0])
        print(f"  Auto-selected interface: {iface}")
        print(f"  All available: {all_ifaces}")
        print(f"  To override: set NETWORK_INTERFACE = '{iface}' at top of script\n")

    STATE["running"] = True

    # Start threads
    threading.Thread(target=sniffer_thread,   args=(iface,), daemon=True).start()
    threading.Thread(target=inference_thread, args=(sio,),   daemon=True).start()
    sio.start_background_task(state_push_thread, sio)

    print(f"✅  NIDS running")
    print(f"🌐  Dashboard → http://localhost:{FLASK_PORT}")
    print(f"🛑  Press Ctrl+C to stop\n")

    try:
        sio.run(app, host="0.0.0.0", port=FLASK_PORT,
                debug=False, use_reloader=False)
    except KeyboardInterrupt:
        STATE["running"] = False
        print("\n🛑  NIDS stopped.")
