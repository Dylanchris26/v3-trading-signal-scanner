# VETRA-X LEARNING SIGNAL BOT — PRACTICE / SIGNAL ONLY
# --------------------------------------------------------
# This is a single-file research bot.
# It learns from completed predictions over time.
#
# Required environment variables:
# IQ_EMAIL, IQ_PASSWORD, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
#
# Optional:
# ACCOUNT_MODE=PRACTICE
# MAX_DAILY_SIGNALS=4
#
# IMPORTANT:
# - No automatic trading.
# - The learning model starts only after it has enough completed examples.
# - Until then, the bot uses a conservative VETRA-style market score.
# - Do not treat the displayed confidence as a guaranteed win probability.

import os, time, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# iqoptionapi is an unofficial community library.
try:
    from iqoptionapi.stable_api import IQ_Option
except Exception as e:
    raise SystemExit(f"iqoptionapi import failed: {e}")

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise SystemExit(f"scikit-learn import failed: {e}")

EMAIL = os.environ.get("IQ_EMAIL", "")
PASSWORD = os.environ.get("IQ_PASSWORD", "")
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MODE = os.environ.get("ACCOUNT_MODE", "PRACTICE").upper()

MAX_DAILY_SIGNALS = int(os.environ.get("MAX_DAILY_SIGNALS", "4"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.78"))

# A deliberately limited pair list. OTC symbols are only considered when
# fresh OTC candles are available; market-status endpoint is intentionally avoided in this build.
REGULAR_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD",
    "NZDUSD","EURJPY","GBPJPY","EURGBP","EURCAD","AUDJPY"
]
OTC_PAIRS = [
    "EURUSD-OTC","GBPUSD-OTC","USDJPY-OTC","USDCHF-OTC",
    "AUDUSD-OTC","USDCAD-OTC","NZDUSD-OTC","EURJPY-OTC",
    "GBPJPY-OTC","EURGBP-OTC","EURCAD-OTC","AUDJPY-OTC",
    "GBPCHF-OTC","NZDJPY-OTC","NZDCAD-OTC"
]

STATE_FILE = Path("vetrax_learning_state.json")
MODEL_FILE = Path("vetrax_model.json")
LOCK_FILE = Path("vetrax_signal_lock.json")

FEATURES = [
    "ret1","ret2","ret3","ret5","body","range","upper_wick","lower_wick",
    "ema9_gap","ema21_gap","ema50_gap","rsi","macd","macd_signal",
    "atr_pct","trend5","momentum5","break_high","break_low",
    "vol_ratio","hour_sin","hour_cos"
]

def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def tg(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=15,
        )
    except Exception as e:
        print("Telegram error:", e)

def connect():
    if not EMAIL or not PASSWORD:
        raise SystemExit("Missing IQ_EMAIL or IQ_PASSWORD.")
    iq = IQ_Option(EMAIL, PASSWORD)
    ok, reason = iq.connect()
    if not ok:
        raise SystemExit(f"IQ Option connection failed: {reason}")
    try:
        iq.change_balance(MODE)
    except Exception:
        pass
    print("Connected to IQ Option in", MODE, "mode.")
    return iq

def candles(iq, pair, n=120):
    try:
        data = iq.get_candles(pair, 60, n, time.time())
        if not data:
            return None
        df = pd.DataFrame(data)
        # API normally uses from/to/open/close/min/max/volume.
        if "from" in df:
            df["ts"] = pd.to_numeric(df["from"], errors="coerce")
        elif "at" in df:
            df["ts"] = pd.to_numeric(df["at"], errors="coerce")
        else:
            return None
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["min"] = pd.to_numeric(df["min"], errors="coerce")
        df["max"] = pd.to_numeric(df["max"], errors="coerce")
        if "volume" not in df:
            df["volume"] = 0
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["ts","open","close","min","max"]).sort_values("ts")
        return df
    except Exception as e:
        print(pair, "candle error:", e)
        return None

def rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def feature_row(df):
    if df is None or len(df) < 60:
        return None
    c = df["close"]
    o = df["open"]
    hi = df["max"]
    lo = df["min"]
    rng = (hi-lo).replace(0, np.nan)
    body = (c-o) / rng
    atr = (hi-lo).rolling(14).mean()
    e9, e21, e50 = ema(c,9), ema(c,21), ema(c,50)
    macd = e9-e21
    macds = ema(macd,9)
    rr = rsi(c)
    volmean = df["volume"].rolling(20).mean().replace(0,np.nan)

    x = {
        "ret1": c.pct_change(1).iloc[-1],
        "ret2": c.pct_change(2).iloc[-1],
        "ret3": c.pct_change(3).iloc[-1],
        "ret5": c.pct_change(5).iloc[-1],
        "body": body.iloc[-1],
        "range": (rng/c).iloc[-1],
        "upper_wick": ((hi-np.maximum(o,c))/rng).iloc[-1],
        "lower_wick": ((np.minimum(o,c)-lo)/rng).iloc[-1],
        "ema9_gap": (c.iloc[-1]/e9.iloc[-1])-1,
        "ema21_gap": (c.iloc[-1]/e21.iloc[-1])-1,
        "ema50_gap": (c.iloc[-1]/e50.iloc[-1])-1,
        "rsi": (rr.iloc[-1]-50)/50,
        "macd": macd.iloc[-1]/c.iloc[-1],
        "macd_signal": macds.iloc[-1]/c.iloc[-1],
        "atr_pct": atr.iloc[-1]/c.iloc[-1],
        "trend5": (e9.iloc[-1]-e21.iloc[-1])/c.iloc[-1],
        "momentum5": c.iloc[-1]/c.iloc[-6]-1,
        "break_high": c.iloc[-1]/hi.iloc[-21:-1].max()-1,
        "break_low": c.iloc[-1]/lo.iloc[-21:-1].min()-1,
        "vol_ratio": (df["volume"].iloc[-1]/volmean.iloc[-1]) if volmean.iloc[-1] else 1,
    }
    dt = datetime.fromtimestamp(float(df["ts"].iloc[-1]), timezone.utc)
    x["hour_sin"] = math.sin(2*math.pi*dt.hour/24)
    x["hour_cos"] = math.cos(2*math.pi*dt.hour/24)
    vals = np.array([x[k] for k in FEATURES], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return vals

def vetra_score(df):
    if df is None or len(df) < 60:
        return None
    c=df.close; o=df.open; hi=df["max"]; lo=df["min"]
    e9,e21,e50=ema(c,9),ema(c,21),ema(c,50)
    rr=rsi(c).iloc[-1]
    last=c.iloc[-1]; prev=c.iloc[-2]
    score_buy=0; score_sell=0

    if e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]: score_buy += 25
    if e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]: score_sell += 25
    if last > e9.iloc[-1]: score_buy += 10
    if last < e9.iloc[-1]: score_sell += 10
    if rr > 52 and rr < 72: score_buy += 12
    if rr < 48 and rr > 28: score_sell += 12
    if c.iloc[-1] > o.iloc[-1]: score_buy += 12
    if c.iloc[-1] < o.iloc[-1]: score_sell += 12
    if c.iloc[-1] > prev: score_buy += 10
    if c.iloc[-1] < prev: score_sell += 10

    recent_hi=hi.iloc[-21:-1].max()
    recent_lo=lo.iloc[-21:-1].min()
    if last > recent_hi: score_buy += 8
    if last < recent_lo: score_sell += 8

    direction="BUY" if score_buy>score_sell else "SELL"
    raw=max(score_buy,score_sell)/87
    return direction, min(.97, max(.50, raw)), score_buy, score_sell

def state():
    return load_json(STATE_FILE, {"examples":[],"signals":[],"day":"","count":0})

def model_predict(st, x):
    # Lightweight online ML. Model parameters are stored as plain JSON so the
    # learner can survive without a binary model file.
    ex=st["examples"]
    if len(ex) < 40:
        return None
    X=np.array([e["x"] for e in ex],dtype=float)
    y=np.array([e["y"] for e in ex],dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    clf=SGDClassifier(loss="log_loss", alpha=0.001, max_iter=1500,
                      random_state=42, class_weight="balanced")
    clf.fit(Xs,y)
    p=float(clf.predict_proba(scaler.transform([x]))[0,1])
    return p

def learn_from_old(st, iq):
    # Resolve predictions once their 1-minute horizon has passed.
    changed=False
    for s in st["signals"]:
        if s.get("resolved"): continue
        if time.time() < s["resolve_at"]: continue
        try:
            df=candles(iq,s["pair"],8)
            if df is None: continue
            entry=float(s["entry_price"])
            last=float(df.close.iloc[-1])
            if s["direction"]=="BUY":
                y=1 if last>entry else 0
            else:
                y=1 if last<entry else 0
            st["examples"].append({"x":s["x"],"y":y,"pair":s["pair"],
                                   "time":s["time"]})
            s["resolved"]=True
            s["result"]="WIN" if y else "LOSS"
            changed=True
        except Exception as e:
            print("learning resolution:",e)
    if changed:
        st["examples"]=st["examples"][-2000:]
        save_json(STATE_FILE,st)

def fresh_enough(df):
    if df is None or len(df)<2: return False
    ts=float(df.ts.iloc[-1])
    # Require a candle from roughly the last 3 minutes. This is a data-freshness
    # check, not a claim that the broker's trading session is open.
    return (time.time()-ts) <= 180

def already_sent(st,pair):
    now=time.time()
    return any((s["pair"]==pair and now-s["created_at"]<300) for s in st["signals"])

def main():
    if MODE!="PRACTICE":
        raise SystemExit("Safety lock: ACCOUNT_MODE must be PRACTICE.")

    st=state()
    today=datetime.now().strftime("%Y-%m-%d")
    if st.get("day")!=today:
        st["day"]=today; st["count"]=0
        st["signals"]=[]
        save_json(STATE_FILE,st)

    iq=connect()
    learn_from_old(st,iq)

    # Scan regular first. OTC is a fallback only if regular data is not fresh.
    candidates=[]
    for pair in REGULAR_PAIRS:
        df=candles(iq,pair)
        if fresh_enough(df):
            candidates.append(("REGULAR",pair,df))

    if not candidates:
        for pair in OTC_PAIRS:
            df=candles(iq,pair)
            if fresh_enough(df):
                candidates.append(("OTC",pair,df))

    if not candidates:
        print("No fresh market data. No signal.")
        return

    if st["count"]>=MAX_DAILY_SIGNALS:
        print("Daily signal limit reached.")
        return

    ranked=[]
    for market,pair,df in candidates:
        sc=vetra_score(df)
        x=feature_row(df)
        if sc and x is not None:
            direction,rule_conf,buy,sell=sc
            ml=model_predict(st,x)
            if ml is None:
                final=rule_conf
                ml_text="learning warm-up"
            else:
                ml_dir="BUY" if ml>=.5 else "SELL"
                ml_conf=max(ml,1-ml)
                # Require agreement between the market score and ML.
                if ml_dir != direction:
                    continue
                final=.55*rule_conf+.45*ml_conf
                ml_text=f"ML {ml_conf*100:.1f}%"
            ranked.append((final,market,pair,df,x,direction,rule_conf,ml_text))

    if not ranked:
        print("No setup where the model and market score agree.")
        return

    ranked.sort(reverse=True,key=lambda z:z[0])
    final,market,pair,df,x,direction,rule_conf,ml_text=ranked[0]

    if final < MIN_CONFIDENCE or already_sent(st,pair):
        print("Best setup below threshold:",pair,final)
        return

    entry=float(df.close.iloc[-1])
    now=datetime.now()
    signal_time=now.strftime("%I:%M %p").lstrip("0")
    signal={
        "pair":pair,"market":market,"direction":direction,
        "confidence":final,"entry_price":entry,"x":x.tolist(),
        "created_at":time.time(),"resolve_at":time.time()+70,
        "time":signal_time,"resolved":False
    }
    st["signals"].append(signal)
    st["signals"]=st["signals"][-100:]
    st["count"]+=1
    save_json(STATE_FILE,st)

    msg=(
        "🧠 VETRA-X LEARNING SIGNAL\n\n"
        f"PAIR: {pair} {'(OTC)' if market=='OTC' else '(REGULAR)'}\n"
        "EXPIRY: 1 MIN\n"
        f"CONFIDENCE: {final*100:.1f}%\n"
        f"ENTRY: {signal_time}\n"
        f"DIRECTION: {'🟢 BUY' if direction=='BUY' else '🔴 SELL'}\n\n"
        f"MODEL: {ml_text}\n"
        f"TODAY: {st['count']}/{MAX_DAILY_SIGNALS}\n"
        "PRACTICE / SIGNAL ONLY"
    )
    tg(msg)
    print(msg)

if __name__=="__main__":
    main()
