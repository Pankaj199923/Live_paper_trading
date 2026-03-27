import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from "recharts";

// ── Black-Scholes helpers ──────────────────────────────────
const erf = (x) => {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign=x<0?-1:1; x=Math.abs(x);
  const t=1/(1+p*x);
  const y=1-(((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t)*Math.exp(-x*x);
  return sign*y;
};
const normCDF=(x)=>(1+erf(x/Math.SQRT2))/2;
const normPDF=(x)=>Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI);

const bsPrice=(S,K,T,r,sigma,type)=>{
  if(T<=0||sigma<=0) return Math.max(0,type==='c'?S-K:K-S);
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  const d2=d1-sigma*Math.sqrt(T);
  if(type==='c') return S*normCDF(d1)-K*Math.exp(-r*T)*normCDF(d2);
  return K*Math.exp(-r*T)*normCDF(-d2)-S*normCDF(-d1);
};

const bsDelta=(S,K,T,r,sigma,type)=>{
  if(T<=0||sigma<=0) return type==='c'?(S>K?1:0):(S<K?-1:0);
  const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
  return type==='c'?normCDF(d1):normCDF(d1)-1;
};

const impliedVol=(price,S,K,T,r,type)=>{
  let sigma=0.25;
  for(let i=0;i<100;i++){
    const p=bsPrice(S,K,T,r,sigma,type);
    const d1=(Math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));
    const vega=S*normPDF(d1)*Math.sqrt(T);
    if(Math.abs(vega)<1e-10) break;
    const diff=p-price;
    if(Math.abs(diff)<1e-5) break;
    sigma=sigma-diff/vega;
    sigma=Math.max(0.01,Math.min(sigma,5));
  }
  return sigma;
};

// ── Constants ──────────────────────────────────────────────
const INITIAL_CAPITAL = 500000;
const LOT_SIZE = 50;
const STEP = 50;
const BROKERAGE = 40;
const STT = 0.000625;
const R = 0.065;
const BASE_SPOT = 24350;
const BASE_IV = 0.14;
const BASE_DTE = 7;

const indices = {
  NIFTY: { spot: 24350, step: 50, lot: 50, iv: 0.14 },
  BANKNIFTY: { spot: 52200, step: 100, lot: 15, iv: 0.18 },
  FINNIFTY: { spot: 23800, step: 50, lot: 40, iv: 0.16 },
};

const fmtINR = (n) => `₹${Math.abs(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
const fmtPts = (n) => `${n>0?"+":""}${n.toFixed(2)}`;
const uid = () => Math.random().toString(36).slice(2,7).toUpperCase();

// ── Simulate spot price (GBM) ──────────────────────────────
const nextSpot = (prev, vol=0.0002, dt=1/86400) => {
  const drift = 0;
  const z = (Math.random()-0.5)*2;
  return prev * Math.exp((drift - 0.5*vol*vol)*dt + vol*Math.sqrt(dt)*z*10);
};

// ── Compute option LTP from spot ─────────────────────────
const optionLTP = (spot, strike, dte, iv, type) => {
  const T = Math.max(dte, 0.001) / 365;
  return Math.max(0.05, bsPrice(spot, strike, T, R, iv, type === "CE" ? "c" : "p"));
};

export default function PaperTradingApp() {
  const [account, setAccount] = useState({ capital: INITIAL_CAPITAL, available: INITIAL_CAPITAL, peak: INITIAL_CAPITAL });
  const [openTrades, setOpenTrades] = useState([]);
  const [closedTrades, setClosedTrades] = useState([]);
  const [equityCurve, setEquityCurve] = useState([{ t: "Start", eq: INITIAL_CAPITAL }]);
  const [spots, setSpots] = useState({ NIFTY: 24350, BANKNIFTY: 52200, FINNIFTY: 23800 });
  const [ivs, setIvs] = useState({ NIFTY: 0.14, BANKNIFTY: 0.18, FINNIFTY: 0.16 });
  const [dte, setDte] = useState(7);
  const [tick, setTick] = useState(0);
  const [activeTab, setActiveTab] = useState("trade");
  const [aiText, setAiText] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [selectedTradeForAI, setSelectedTradeForAI] = useState(null);
  const [alerts, setAlerts] = useState([]);

  // Order form state
  const [selIndex, setSelIndex] = useState("NIFTY");
  const [selType, setSelType] = useState("CE");
  const [selAction, setSelAction] = useState("BUY");
  const [selLots, setSelLots] = useState(1);
  const [slPts, setSlPts] = useState(20);
  const [tgtPts, setTgtPts] = useState(40);
  const [customNote, setCustomNote] = useState("");

  const timerRef = useRef(null);
  const spotHistRef = useRef([]);

  // Strike based on ATM
  const atm = Math.round(spots[selIndex] / STEP) * STEP;
  const [strikeOffset, setStrikeOffset] = useState(0);
  const selStrike = atm + strikeOffset * STEP;

  const curSpot = spots[selIndex];
  const curIV = ivs[selIndex];
  const ltp = optionLTP(curSpot, selStrike, dte, curIV, selType);
  const lot = indices[selIndex]?.lot || 50;
  const qty = selLots * lot;
  const marginEst = selAction === "BUY" ? ltp * qty : selStrike * qty * 0.12 + ltp * qty;
  const rr = (tgtPts / Math.max(slPts, 1)).toFixed(1);

  // ── Live price update every 1s ─────────────────────────
  useEffect(() => {
    timerRef.current = setInterval(() => {
      setSpots(prev => {
        const next = {};
        Object.keys(prev).forEach(k => {
          next[k] = parseFloat(nextSpot(prev[k], 0.15).toFixed(2));
        });
        return next;
      });
      setIvs(prev => {
        const next = {};
        Object.keys(prev).forEach(k => {
          const delta = (Math.random() - 0.5) * 0.002;
          next[k] = Math.max(0.08, Math.min(0.45, prev[k] + delta));
        });
        return next;
      });
      setTick(t => t + 1);
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, []);

  // ── Update open trade P&L every tick ──────────────────
  useEffect(() => {
    if (openTrades.length === 0) return;
    setOpenTrades(prev => prev.map(t => {
      const liveSpot = spots[t.index] || t.spotEntry;
      const liveIV = ivs[t.index] || t.iv;
      const liveLTP = parseFloat(optionLTP(liveSpot, t.strike, t.dte, liveIV, t.type).toFixed(2));
      const pnlPts = t.action === "BUY" ? liveLTP - t.entry : t.entry - liveLTP;
      const pnl = parseFloat((pnlPts * t.qty - t.costs).toFixed(2));
      const slHit = t.action === "BUY" ? liveLTP <= t.sl : liveLTP >= t.sl;
      const tgtHit = t.action === "BUY" ? liveLTP >= t.target : liveLTP <= t.target;
      let newStatus = t.status;
      if (t.status === "OPEN") {
        if (slHit) newStatus = "SL_PENDING";
        else if (tgtHit) newStatus = "TGT_PENDING";
      }
      return { ...t, ltp: liveLTP, pnlPts: parseFloat(pnlPts.toFixed(2)), pnl, status: newStatus };
    }));
  }, [tick]);

  // ── Auto-close SL/Target hits ─────────────────────────
  useEffect(() => {
    const toClose = openTrades.filter(t => t.status === "SL_PENDING" || t.status === "TGT_PENDING");
    if (toClose.length === 0) return;
    toClose.forEach(t => {
      const reason = t.status === "SL_PENDING" ? "SL HIT" : "TARGET HIT";
      const now = new Date().toLocaleTimeString("en-IN", { hour12: false });
      const closed = { ...t, status: "CLOSED", exitLTP: t.ltp, exitTime: now, exitReason: reason };
      setOpenTrades(prev => prev.filter(x => x.id !== t.id));
      setClosedTrades(prev => [closed, ...prev]);
      setAccount(prev => {
        const newAvail = parseFloat((prev.available + t.margin + t.pnl).toFixed(2));
        const newPeak = Math.max(prev.peak, newAvail);
        return { ...prev, available: newAvail, peak: newPeak };
      });
      const timeLabel = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
      setEquityCurve(prev => {
        const newEq = account.available + t.margin + t.pnl;
        return [...prev.slice(-50), { t: timeLabel, eq: parseFloat(newEq.toFixed(0)) }];
      });
      const alertMsg = reason === "SL HIT"
        ? `⛔ SL hit on ${t.action} ${t.type} ${t.strike} — ₹${Math.abs(t.pnl).toFixed(0)} loss`
        : `🏆 Target hit on ${t.action} ${t.type} ${t.strike} — +₹${t.pnl.toFixed(0)} profit`;
      setAlerts(prev => [{ id: uid(), msg: alertMsg, time: now, type: reason }, ...prev.slice(0, 9)]);
    });
  }, [openTrades]);

  // ── Execute trade ─────────────────────────────────────
  const executeTrade = () => {
    const spot = spots[selIndex];
    const iv = ivs[selIndex];
    const premium = parseFloat(optionLTP(spot, selStrike, dte, iv, selType).toFixed(2));
    if (premium <= 0.05) { alert("LTP too low to trade"); return; }
    const margin = selAction === "BUY" ? premium * qty : selStrike * qty * 0.12 + premium * qty;
    if (margin > account.available) { alert(`Insufficient margin. Need ${fmtINR(margin)}`); return; }
    const costs = BROKERAGE * selLots + (selAction === "SELL" ? premium * qty * STT : 0);
    const sl = selAction === "BUY" ? parseFloat((premium - slPts).toFixed(2)) : parseFloat((premium + slPts).toFixed(2));
    const target = selAction === "BUY" ? parseFloat((premium + tgtPts).toFixed(2)) : parseFloat((premium - tgtPts).toFixed(2));
    const now = new Date().toLocaleTimeString("en-IN", { hour12: false });
    const trade = {
      id: uid(), time: now, index: selIndex, type: selType, action: selAction,
      strike: selStrike, lots: selLots, qty, entry: premium, ltp: premium,
      sl, target, slPts, tgtPts, margin: parseFloat(margin.toFixed(2)),
      costs: parseFloat(costs.toFixed(2)), spotEntry: spot, iv, dte,
      pnl: 0, pnlPts: 0, status: "OPEN", note: customNote,
      delta: parseFloat(bsDelta(spot, selStrike, dte/365, R, iv, selType === "CE" ? "c" : "p").toFixed(3)),
    };
    setOpenTrades(prev => [trade, ...prev]);
    setAccount(prev => ({ ...prev, available: parseFloat((prev.available - margin).toFixed(2)) }));
    setCustomNote("");
  };

  // ── Manual close ──────────────────────────────────────
  const manualClose = (id) => {
    const t = openTrades.find(x => x.id === id);
    if (!t) return;
    const now = new Date().toLocaleTimeString("en-IN", { hour12: false });
    const closed = { ...t, status: "CLOSED", exitLTP: t.ltp, exitTime: now, exitReason: "MANUAL" };
    setOpenTrades(prev => prev.filter(x => x.id !== id));
    setClosedTrades(prev => [closed, ...prev]);
    setAccount(prev => {
      const newAvail = parseFloat((prev.available + t.margin + t.pnl).toFixed(2));
      return { ...prev, available: newAvail, peak: Math.max(prev.peak, newAvail) };
    });
    const timeLabel = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
    setEquityCurve(prev => [...prev.slice(-50), { t: timeLabel, eq: parseFloat((account.available + t.margin + t.pnl).toFixed(0)) }]);
  };

  // ── AI reasoning ──────────────────────────────────────
  const generateAI = async (trade) => {
    setSelectedTradeForAI(trade);
    setActiveTab("ai");
    setAiLoading(true);
    setAiText("");
    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [{
            role: "user",
            content: `You are a senior NSE F&O trading analyst. Write a brief, sharp trade journal entry for this paper trade.

Trade: ${trade.action} ${trade.type} ${trade.strike} (${trade.index})
Entry: ₹${trade.entry} | SL: ₹${trade.sl} (${trade.slPts}pts) | Target: ₹${trade.target} (${trade.tgtPts}pts)
Spot at Entry: ₹${trade.spotEntry.toFixed(2)} | IV: ${(trade.iv*100).toFixed(1)}% | DTE: ${trade.dte}
Delta: ${trade.delta} | Lots: ${trade.lots} (${trade.qty} qty) | R:R = 1:${(trade.tgtPts/trade.slPts).toFixed(1)}
Trader Note: ${trade.note || "None"}
Status: ${trade.status === "CLOSED" ? `${trade.exitReason} @ ₹${trade.exitLTP} | P&L: ₹${trade.pnl.toFixed(0)}` : `Open | Live P&L: ₹${trade.pnl.toFixed(0)}`}

Write a sharp 5-section analysis using markdown:
**Setup** – One-line description.
**Thesis** – 2-3 sentences on why this trade. Reference numbers.
**Risk** – SL logic and max loss in ₹.
**Target** – Why this level, R:R justification.
**Score** – X/10 with one-line verdict.

Keep it tight and professional. No fluff.`
          }]
        })
      });
      const data = await resp.json();
      const text = data.content?.map(b => b.type === "text" ? b.text : "").join("") || "Error generating analysis.";
      setAiText(text);
    } catch (e) {
      setAiText("⚠️ AI analysis unavailable. Check API connection.");
    }
    setAiLoading(false);
  };

  // ── Performance stats ─────────────────────────────────
  const openPnL = openTrades.reduce((s, t) => s + t.pnl, 0);
  const closedPnL = closedTrades.reduce((s, t) => s + t.pnl, 0);
  const totalPnL = openPnL + closedPnL;
  const wins = closedTrades.filter(t => t.pnl > 0);
  const losses = closedTrades.filter(t => t.pnl < 0);
  const winRate = closedTrades.length > 0 ? (wins.length / closedTrades.length * 100).toFixed(1) : "—";
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + t.pnl, 0) / losses.length) : 0;
  const expectancy = closedTrades.length > 0
    ? ((wins.length / closedTrades.length) * avgWin - (losses.length / closedTrades.length) * avgLoss).toFixed(0)
    : "—";
  const pf = avgLoss > 0 ? (avgWin * wins.length / (avgLoss * losses.length)).toFixed(2) : "—";
  const dd = account.peak > 0 ? ((account.peak - account.available) / account.peak * 100).toFixed(1) : "0.0";
  const retPct = ((account.available - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100).toFixed(2);

  // ── UI ────────────────────────────────────────────────
  const S = {
    root: { fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)", fontSize: 12, color: "var(--color-text-primary)", background: "transparent" },
    card: { background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: "12px 14px", marginBottom: 8 },
    hdr: { background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 10, padding: "12px 16px", marginBottom: 10 },
    badge: (c) => ({ background: c === "g" ? "var(--color-background-success)" : c === "r" ? "var(--color-background-danger)" : "var(--color-background-secondary)", color: c === "g" ? "var(--color-text-success)" : c === "r" ? "var(--color-text-danger)" : "var(--color-text-secondary)", borderRadius: 4, padding: "2px 7px", fontSize: 11, fontWeight: 500 }),
    label: { fontSize: 10, color: "var(--color-text-secondary)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 2 },
    val: (clr) => ({ fontSize: 20, fontWeight: 500, color: clr || "var(--color-text-primary)" }),
    btn: (active, clr) => ({ padding: "6px 14px", borderRadius: 6, border: active ? "none" : "0.5px solid var(--color-border-secondary)", background: active ? (clr || "var(--color-text-primary)") : "transparent", color: active ? "var(--color-background-primary)" : "var(--color-text-secondary)", cursor: "pointer", fontSize: 12, fontWeight: 500 }),
    input: { width: "100%", padding: "6px 10px", borderRadius: 6, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontSize: 12, boxSizing: "border-box" },
    execBtn: { width: "100%", padding: "10px", borderRadius: 8, border: "none", background: "var(--color-text-primary)", color: "var(--color-background-primary)", fontSize: 13, fontWeight: 500, cursor: "pointer", marginTop: 8 },
    tab: (active) => ({ padding: "8px 16px", cursor: "pointer", borderBottom: active ? "2px solid var(--color-text-primary)" : "2px solid transparent", color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)", fontSize: 12, fontWeight: 500, letterSpacing: "0.05em" }),
  };

  const pnlColor = (n) => n > 0 ? "var(--color-text-success)" : n < 0 ? "var(--color-text-danger)" : "var(--color-text-secondary)";

  const SpotTicker = () => (
    <div style={{ display: "flex", gap: 12, marginBottom: 10 }}>
      {Object.entries(spots).map(([k, v]) => {
        const prev = indices[k].spot;
        const chg = ((v - prev) / prev * 100);
        return (
          <div key={k} style={{ ...S.card, flex: 1, marginBottom: 0, padding: "8px 12px" }}>
            <div style={S.label}>{k}</div>
            <div style={{ fontSize: 16, fontWeight: 500 }}>{v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
            <div style={{ fontSize: 11, color: pnlColor(chg) }}>{chg > 0 ? "▲" : "▼"} {Math.abs(chg).toFixed(2)}%</div>
          </div>
        );
      })}
      <div style={{ ...S.card, flex: 1, marginBottom: 0, padding: "8px 12px" }}>
        <div style={S.label}>NET P&L</div>
        <div style={{ fontSize: 16, fontWeight: 500, color: pnlColor(totalPnL) }}>{totalPnL >= 0 ? "+" : ""}{fmtINR(totalPnL)}</div>
        <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Open: {openTrades.length} | Closed: {closedTrades.length}</div>
      </div>
    </div>
  );

  const AccountBar = () => (
    <div style={{ ...S.hdr, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
      <div>
        <div style={S.label}>Available Capital</div>
        <div style={{ fontSize: 24, fontWeight: 500 }}>{fmtINR(account.available)}</div>
        <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Peak: {fmtINR(account.peak)} | DD: {dd}%</div>
      </div>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {[
          ["Open P&L", openPnL, pnlColor(openPnL)],
          ["Closed P&L", closedPnL, pnlColor(closedPnL)],
          ["Return", `${retPct}%`, pnlColor(parseFloat(retPct))],
          ["Win Rate", closedTrades.length > 0 ? `${winRate}%` : "—", "var(--color-text-primary)"],
        ].map(([label, val, clr]) => (
          <div key={label} style={{ textAlign: "center" }}>
            <div style={S.label}>{label}</div>
            <div style={{ ...S.val(clr), fontSize: 16 }}>{typeof val === "number" ? (val >= 0 ? "+" : "") + fmtINR(val) : val}</div>
          </div>
        ))}
      </div>
    </div>
  );

  const TradeEntry = () => {
    const tgtProgress = (() => {
      const entry = ltp;
      if (selAction === "BUY") return Math.max(0, Math.min(100, (ltp - entry) / tgtPts * 100));
      return 0;
    })();

    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr", gap: 10 }}>
        {/* Left: order form */}
        <div style={S.card}>
          <div style={{ fontWeight: 500, marginBottom: 10, fontSize: 13 }}>Order Entry</div>

          <div style={{ marginBottom: 8 }}>
            <div style={S.label}>Index</div>
            <div style={{ display: "flex", gap: 4 }}>
              {["NIFTY", "BANKNIFTY", "FINNIFTY"].map(k => (
                <button key={k} onClick={() => setSelIndex(k)} style={{ ...S.btn(selIndex === k), padding: "4px 8px", fontSize: 11 }}>{k}</button>
              ))}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
            <div>
              <div style={S.label}>Action</div>
              <div style={{ display: "flex", gap: 4 }}>
                {["BUY", "SELL"].map(a => (
                  <button key={a} onClick={() => setSelAction(a)} style={{ ...S.btn(selAction === a, a === "BUY" ? "#16a34a" : "#dc2626"), flex: 1, padding: "5px 0" }}>{a}</button>
                ))}
              </div>
            </div>
            <div>
              <div style={S.label}>Type</div>
              <div style={{ display: "flex", gap: 4 }}>
                {["CE", "PE"].map(t => (
                  <button key={t} onClick={() => setSelType(t)} style={{ ...S.btn(selType === t), flex: 1, padding: "5px 0" }}>{t}</button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={S.label}>Strike  (ATM: {atm})</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button onClick={() => setStrikeOffset(s => s - 1)} style={{ ...S.btn(false), padding: "4px 10px" }}>−</button>
              <div style={{ flex: 1, textAlign: "center", fontWeight: 500, fontSize: 15 }}>{selStrike}</div>
              <button onClick={() => setStrikeOffset(s => s + 1)} style={{ ...S.btn(false), padding: "4px 10px" }}>+</button>
            </div>
            <div style={{ fontSize: 10, color: "var(--color-text-secondary)", textAlign: "center", marginTop: 2 }}>
              {selStrike === atm ? "ATM" : selStrike > atm ? `OTM +${(selStrike - atm)}` : `ITM −${atm - selStrike}`}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 8 }}>
            <div>
              <div style={S.label}>Lots</div>
              <input type="number" min={1} max={50} value={selLots} onChange={e => setSelLots(+e.target.value)} style={S.input} />
            </div>
            <div>
              <div style={S.label}>SL pts</div>
              <input type="number" min={1} value={slPts} onChange={e => setSlPts(+e.target.value)} style={S.input} />
            </div>
            <div>
              <div style={S.label}>Tgt pts</div>
              <input type="number" min={1} value={tgtPts} onChange={e => setTgtPts(+e.target.value)} style={S.input} />
            </div>
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={S.label}>DTE</div>
            <input type="range" min={1} max={30} value={dte} onChange={e => setDte(+e.target.value)} style={{ width: "100%" }} />
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)", textAlign: "center" }}>{dte} days to expiry</div>
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={S.label}>Note</div>
            <input placeholder="Trade reason..." value={customNote} onChange={e => setCustomNote(e.target.value)} style={S.input} />
          </div>

          {/* Preview */}
          <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: 10, marginBottom: 8 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                ["LTP", `₹${ltp.toFixed(2)}`, pnlColor(0)],
                ["Qty", qty, "var(--color-text-primary)"],
                ["Margin", fmtINR(marginEst), "var(--color-text-warning)"],
                ["R:R", `1:${rr}`, parseFloat(rr) >= 2 ? "var(--color-text-success)" : parseFloat(rr) >= 1.5 ? "var(--color-text-warning)" : "var(--color-text-danger)"],
                ["SL", `₹${(selAction === "BUY" ? ltp - slPts : ltp + slPts).toFixed(2)}`, "var(--color-text-danger)"],
                ["Target", `₹${(selAction === "BUY" ? ltp + tgtPts : ltp - tgtPts).toFixed(2)}`, "var(--color-text-success)"],
              ].map(([l, v, c]) => (
                <div key={l}>
                  <div style={S.label}>{l}</div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: c }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          <button onClick={executeTrade} style={S.execBtn}>
            🚀 Execute Paper Trade
          </button>

          {/* Quick templates */}
          <div style={{ marginTop: 10 }}>
            <div style={{ ...S.label, marginBottom: 6 }}>Quick Templates</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
              {[
                ["ATM CE Buy", () => { setSelType("CE"); setSelAction("BUY"); setStrikeOffset(0); }],
                ["ATM PE Buy", () => { setSelType("PE"); setSelAction("BUY"); setStrikeOffset(0); }],
                ["Sell ATM CE", () => { setSelType("CE"); setSelAction("SELL"); setStrikeOffset(0); }],
                ["Sell ATM PE", () => { setSelType("PE"); setSelAction("SELL"); setStrikeOffset(0); }],
              ].map(([label, fn]) => (
                <button key={label} onClick={fn} style={{ ...S.btn(false), padding: "5px 6px", fontSize: 11 }}>{label}</button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: open positions */}
        <div>
          <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 6 }}>
            Open Positions ({openTrades.length})
          </div>
          {openTrades.length === 0 ? (
            <div style={{ ...S.card, textAlign: "center", color: "var(--color-text-secondary)", padding: "24px 0" }}>
              No open positions — execute a trade
            </div>
          ) : openTrades.map(t => {
            const progTgt = t.action === "BUY"
              ? Math.max(0, Math.min(100, (t.ltp - t.entry) / t.tgtPts * 100))
              : Math.max(0, Math.min(100, (t.entry - t.ltp) / t.tgtPts * 100));
            const progSL = t.action === "BUY"
              ? Math.max(0, Math.min(100, (t.entry - t.ltp) / t.slPts * 100))
              : Math.max(0, Math.min(100, (t.ltp - t.entry) / t.slPts * 100));

            return (
              <div key={t.id} style={{ ...S.card, borderLeft: `3px solid ${pnlColor(t.pnl)}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                  <div>
                    <span style={{ fontWeight: 500, fontSize: 13 }}>{t.index} {t.strike} {t.type}</span>
                    <span style={{ ...S.badge(t.action === "BUY" ? "g" : "r"), marginLeft: 6 }}>{t.action}</span>
                    <span style={{ color: "var(--color-text-secondary)", fontSize: 10, marginLeft: 6 }}>#{t.id} · {t.time}</span>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 17, fontWeight: 500, color: pnlColor(t.pnl) }}>{t.pnl >= 0 ? "+" : ""}{fmtINR(t.pnl)}</div>
                    <div style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{fmtPts(t.pnlPts)} pts</div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 6 }}>
                  <span>E: ₹{t.entry}</span>
                  <span>LTP: <b style={{ color: "var(--color-text-primary)" }}>₹{t.ltp}</b></span>
                  <span style={{ color: "var(--color-text-danger)" }}>SL: ₹{t.sl}</span>
                  <span style={{ color: "var(--color-text-success)" }}>T: ₹{t.target}</span>
                  <span>Δ {t.delta}</span>
                </div>
                {/* Progress bars */}
                <div style={{ marginBottom: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--color-text-secondary)", marginBottom: 2 }}>
                    <span>To target {progTgt.toFixed(0)}%</span>
                    <span>To SL {progSL.toFixed(0)}%</span>
                  </div>
                  <div style={{ height: 3, background: "var(--color-border-tertiary)", borderRadius: 2, marginBottom: 2 }}>
                    <div style={{ height: "100%", width: `${progTgt}%`, background: "var(--color-text-success)", borderRadius: 2, transition: "width 0.5s" }} />
                  </div>
                  <div style={{ height: 2, background: "var(--color-border-tertiary)", borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${progSL}%`, background: "var(--color-text-danger)", borderRadius: 2, transition: "width 0.5s" }} />
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <button onClick={() => manualClose(t.id)} style={{ ...S.btn(false), fontSize: 11, padding: "3px 10px" }}>✕ Close</button>
                  <button onClick={() => generateAI(t)} style={{ ...S.btn(false), fontSize: 11, padding: "3px 10px" }}>🧠 AI Analysis ↗</button>
                  {t.note && <span style={{ fontSize: 10, color: "var(--color-text-secondary)", alignSelf: "center" }}>📝 {t.note}</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const AnalyticsTab = () => (
    <div>
      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 10 }}>
        {[
          ["Total P&L", totalPnL >= 0 ? "+"+fmtINR(totalPnL) : "−"+fmtINR(-totalPnL), pnlColor(totalPnL)],
          ["Win Rate", closedTrades.length > 0 ? `${winRate}%` : "—", "var(--color-text-primary)"],
          ["Profit Factor", pf, pf !== "—" && parseFloat(pf) >= 1.5 ? "var(--color-text-success)" : "var(--color-text-danger)"],
          ["Expectancy", expectancy !== "—" ? (parseFloat(expectancy) >= 0 ? "+₹" : "−₹") + Math.abs(expectancy) : "—", pnlColor(parseFloat(expectancy) || 0)],
          ["Avg Win", avgWin ? "+"+fmtINR(avgWin) : "—", "var(--color-text-success)"],
          ["Avg Loss", avgLoss ? "−"+fmtINR(avgLoss) : "—", "var(--color-text-danger)"],
          ["Max DD", `${dd}%`, parseFloat(dd) > 10 ? "var(--color-text-danger)" : parseFloat(dd) > 5 ? "var(--color-text-warning)" : "var(--color-text-success)"],
          ["Return", `${retPct}%`, pnlColor(parseFloat(retPct))],
        ].map(([l, v, c]) => (
          <div key={l} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "10px 12px" }}>
            <div style={S.label}>{l}</div>
            <div style={{ fontSize: 18, fontWeight: 500, color: c }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 10 }}>
        {/* Equity curve */}
        <div style={S.card}>
          <div style={{ fontWeight: 500, marginBottom: 8, fontSize: 13 }}>Equity Curve</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={equityCurve} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <XAxis dataKey="t" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={v => `${(v/1000).toFixed(0)}k`} domain={["auto", "auto"]} />
              <Tooltip formatter={(v) => fmtINR(v)} contentStyle={{ fontSize: 11 }} />
              <ReferenceLine y={INITIAL_CAPITAL} stroke="#888" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="eq" stroke="var(--color-text-primary)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* P&L per trade */}
        <div style={S.card}>
          <div style={{ fontWeight: 500, marginBottom: 8, fontSize: 13 }}>P&L Per Trade</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={closedTrades.slice().reverse().slice(-15)} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
              <XAxis dataKey="id" tick={{ fontSize: 9 }} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={v => `${(v/1000).toFixed(0)}k`} />
              <Tooltip formatter={(v) => fmtINR(v)} contentStyle={{ fontSize: 11 }} />
              <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
                {closedTrades.slice().reverse().slice(-15).map((t, i) => (
                  <Cell key={i} fill={t.pnl >= 0 ? "#16a34a" : "#dc2626"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const HistoryTab = () => (
    <div>
      <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 6 }}>Closed Trades ({closedTrades.length})</div>
      {closedTrades.length === 0 ? (
        <div style={{ ...S.card, textAlign: "center", color: "var(--color-text-secondary)", padding: "24px 0" }}>No closed trades yet</div>
      ) : closedTrades.slice(0, 20).map(t => (
        <div key={t.id} style={{ ...S.card, borderLeft: `3px solid ${pnlColor(t.pnl)}`, padding: "8px 12px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 }}>
            <div>
              <span style={{ fontWeight: 500 }}>{t.index} {t.strike} {t.type} {t.action}</span>
              <span style={{ ...S.badge(t.exitReason === "TARGET HIT" ? "g" : t.exitReason === "SL HIT" ? "r" : ""), marginLeft: 6 }}>
                {t.exitReason === "TARGET HIT" ? "🏆" : t.exitReason === "SL HIT" ? "⛔" : "✋"} {t.exitReason}
              </span>
              <span style={{ color: "var(--color-text-secondary)", fontSize: 10, marginLeft: 8 }}>#{t.id} · {t.exitTime}</span>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>E:₹{t.entry} → X:₹{t.exitLTP} ({fmtPts(t.pnlPts)}pts)</span>
              <span style={{ fontSize: 16, fontWeight: 500, color: pnlColor(t.pnl) }}>{t.pnl >= 0 ? "+" : ""}{fmtINR(t.pnl)}</span>
              <button onClick={() => generateAI(t)} style={{ ...S.btn(false), fontSize: 10, padding: "2px 8px" }}>AI ↗</button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const AITab = () => (
    <div style={S.card}>
      {!selectedTradeForAI ? (
        <div style={{ textAlign: "center", color: "var(--color-text-secondary)", padding: "40px 0" }}>
          Click "AI Analysis" on any trade to generate reasoning
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontWeight: 500, fontSize: 13 }}>AI Trade Journal — #{selectedTradeForAI.id}</div>
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>
              {selectedTradeForAI.action} {selectedTradeForAI.type} {selectedTradeForAI.strike} ({selectedTradeForAI.index})
              · Entry ₹{selectedTradeForAI.entry} · R:R 1:{(selectedTradeForAI.tgtPts / selectedTradeForAI.slPts).toFixed(1)}
              {selectedTradeForAI.status === "CLOSED" ? ` · ${selectedTradeForAI.exitReason} @ ₹${selectedTradeForAI.exitLTP}` : " · Open"}
            </div>
          </div>
          {aiLoading ? (
            <div style={{ color: "var(--color-text-secondary)", padding: "20px 0" }}>
              <div style={{ marginBottom: 8 }}>⏳ Claude is analysing this trade…</div>
              <div style={{ height: 4, background: "var(--color-border-tertiary)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: "60%", background: "var(--color-text-primary)", borderRadius: 2, animation: "none" }} />
              </div>
            </div>
          ) : (
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, fontSize: 13 }}>
              {aiText.split("\n").map((line, i) => {
                if (line.startsWith("**") && line.endsWith("**")) {
                  return <div key={i} style={{ fontWeight: 500, marginTop: 12, marginBottom: 4, color: "var(--color-text-primary)" }}>{line.replace(/\*\*/g, "")}</div>;
                }
                if (line.startsWith("**")) {
                  const parts = line.split("**").filter(Boolean);
                  return <div key={i} style={{ marginBottom: 4 }}>{parts.map((p, j) => j % 2 === 0 ? <b key={j}>{p}</b> : p)}</div>;
                }
                if (line.startsWith("- ")) {
                  return <div key={i} style={{ paddingLeft: 12, marginBottom: 2 }}>· {line.slice(2)}</div>;
                }
                return <div key={i} style={{ marginBottom: 4, color: "var(--color-text-secondary)" }}>{line}</div>;
              })}
            </div>
          )}
        </>
      )}
    </div>
  );

  const AlertsTab = () => (
    <div>
      <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 6 }}>Live Alerts</div>
      {alerts.length === 0 ? (
        <div style={{ ...S.card, textAlign: "center", color: "var(--color-text-secondary)", padding: "24px 0" }}>No alerts yet — alerts fire when SL or target hits</div>
      ) : alerts.map(a => (
        <div key={a.id} style={{ ...S.card, borderLeft: `3px solid ${a.type === "SL HIT" ? "var(--color-text-danger)" : a.type === "TARGET HIT" ? "var(--color-text-success)" : "var(--color-text-secondary)"}`, padding: "8px 12px", marginBottom: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 12 }}>{a.msg}</span>
            <span style={{ fontSize: 10, color: "var(--color-text-secondary)" }}>{a.time}</span>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ ...S.root, padding: "8px 0" }}>
      {/* Header */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div>
            <span style={{ fontSize: 18, fontWeight: 500 }}>Paper Trading</span>
            <span style={{ fontSize: 11, color: "var(--color-text-secondary)", marginLeft: 10 }}>Simulated · Real prices · Zero risk</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#16a34a" }} />
            <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>Live · 1s updates</span>
          </div>
        </div>
        <AccountBar />
        <SpotTicker />
      </div>

      {/* Nav tabs */}
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 10 }}>
        {[
          ["trade", "Trade Entry"],
          ["analytics", "Analytics"],
          ["history", "History"],
          ["ai", "AI Journal"],
          ["alerts", `Alerts ${alerts.length > 0 ? `(${alerts.length})` : ""}`],
        ].map(([k, label]) => (
          <div key={k} onClick={() => setActiveTab(k)} style={S.tab(activeTab === k)}>{label}</div>
        ))}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={() => {
            if (!confirm("Reset all paper trades?")) return;
            setOpenTrades([]); setClosedTrades([]); setAlerts([]);
            setAccount({ capital: INITIAL_CAPITAL, available: INITIAL_CAPITAL, peak: INITIAL_CAPITAL });
            setEquityCurve([{ t: "Start", eq: INITIAL_CAPITAL }]);
          }} style={{ ...S.btn(false), fontSize: 10, padding: "3px 8px", color: "var(--color-text-danger)", borderColor: "var(--color-border-danger)" }}>
            Reset
          </button>
        </div>
      </div>

      {activeTab === "trade" && <TradeEntry />}
      {activeTab === "analytics" && <AnalyticsTab />}
      {activeTab === "history" && <HistoryTab />}
      {activeTab === "ai" && <AITab />}
      {activeTab === "alerts" && <AlertsTab />}

      {/* Footer */}
      <div style={{ marginTop: 12, fontSize: 10, color: "var(--color-text-tertiary)", textAlign: "center", borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 8 }}>
        Simulated prices via Black-Scholes · Real Anthropic AI reasoning · Lot sizes: NIFTY 50 · BANKNIFTY 15 · FINNIFTY 40
      </div>
    </div>
  );
}
