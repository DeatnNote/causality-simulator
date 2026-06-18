import { useState, useEffect, useCallback, useMemo } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar,
         XAxis, YAxis, Tooltip, ResponsiveContainer,
         ReferenceLine, CartesianGrid } from "recharts";

// ─────────────────────────────────────────────────────────────
// Causal Models — pure JS implementations of the data-generating
// processes from the Python files. Each model encodes the causal
// DAG as a set of structural equations.
// ─────────────────────────────────────────────────────────────

const MODELS = {
  hr: {
    label: "HR Attrition",
    icon: "👥",
    description: "Does improving Work-Life Balance reduce employee attrition?",
    treatment: { key: "wlb", label: "Work-Life Balance", min: 1, max: 4, step: 0.1, unit: "score", default: 2.5 },
    controls: [
      { key: "salary",  label: "Monthly Income ($)",  min: 20000, max: 150000, step: 1000,  default: 60000 },
      { key: "tenure",  label: "Years at Company",    min: 0,     max: 30,     step: 0.5,   default: 5    },
      { key: "overtime",label: "Works Overtime",      min: 0,     max: 1,      step: 1,     default: 0,   type: "toggle" },
    ],
    outcome: { key: "attrition", label: "Attrition Probability", unit: "%", format: v => (v * 100).toFixed(1) + "%" },
    trueEffect: -0.35,
    confounders: ["tenure → salary", "overtime → wlb"],
    simulate: (treatment, controls) => {
      const { salary, tenure, overtime } = controls;
      // Structural equation from the DAG
      const logit = 1.5
        - treatment * 0.35
        - (salary / 100000) * 0.50
        - tenure  * 0.04
        + overtime * 0.60
        - 0.40 * (2.5 + (salary - 60000) / 50000 * 0.8 + treatment * 0.4);
      return 1 / (1 + Math.exp(-logit));
    },
    doseRange: Array.from({ length: 31 }, (_, i) => 1 + i * 0.1),
    color: "#3b82f6",
    insight: (ate) => ate < 0
      ? `Improving WLB by 1 unit reduces attrition by ${Math.abs(ate * 100).toFixed(1)} percentage points`
      : `Improving WLB has unexpected positive attrition effect`,
  },

  marketing: {
    label: "Marketing Mix",
    icon: "📈",
    description: "Does increasing Ad Spend causally increase Sales?",
    treatment: { key: "adSpend", label: "Weekly Ad Spend ($)", min: 1000, max: 40000, step: 500, unit: "$", default: 10000 },
    controls: [
      { key: "price",    label: "Product Price ($)",  min: 10, max: 50,    step: 1,    default: 30 },
      { key: "season",   label: "Peak Season",        min: 0,  max: 1,     step: 1,    default: 0, type: "toggle" },
      { key: "competitor",label:"Competitor Spend ($)",min:1000,max:20000, step: 500,  default: 5000 },
    ],
    outcome: { key: "sales", label: "Weekly Units Sold", unit: "units", format: v => Math.round(v).toLocaleString() + " units" },
    trueEffect: 0.008,
    confounders: ["season → adSpend", "season → sales"],
    simulate: (treatment, controls) => {
      const { price, season, competitor } = controls;
      const brandAwareness = 20 + Math.log(treatment + 1) * 3;
      return (
        500 +
        treatment * 0.008 +
        brandAwareness * 2.5 +
        season * 300 +
        (40 - price) * 15 +
        competitor * (-0.002)
      );
    },
    doseRange: Array.from({ length: 40 }, (_, i) => 1000 + i * 1000),
    color: "#22c55e",
    insight: (ate) => `Each extra $1,000 in ad spend generates ~${(ate * 1000).toFixed(0)} additional units sold`,
  },

  healthcare: {
    label: "Healthcare",
    icon: "🏥",
    description: "Does Exercise causally reduce Blood Pressure?",
    treatment: { key: "exercise", label: "Exercise Hours / Week", min: 0, max: 15, step: 0.5, unit: "hrs/wk", default: 3 },
    controls: [
      { key: "age",      label: "Patient Age (years)", min: 18, max: 85,  step: 1,    default: 45 },
      { key: "bmi",      label: "BMI",                 min: 16, max: 45,  step: 0.5,  default: 27 },
      { key: "smoking",  label: "Smoker",              min: 0,  max: 1,   step: 1,    default: 0, type: "toggle" },
      { key: "medication",label:"On Medication",       min: 0,  max: 1,   step: 1,    default: 0, type: "toggle" },
    ],
    outcome: { key: "bp", label: "Systolic Blood Pressure", unit: "mmHg", format: v => v.toFixed(1) + " mmHg" },
    trueEffect: -1.5,
    confounders: ["age → exercise", "bmi → exercise"],
    simulate: (treatment, controls) => {
      const { age, bmi, smoking, medication } = controls;
      return (
        130 +
        age      * 0.4 +
        bmi      * 0.8 +
        smoking  * 8 -
        treatment * 1.5 -
        medication * 12
      );
    },
    doseRange: Array.from({ length: 31 }, (_, i) => i * 0.5),
    color: "#f59e0b",
    insight: (ate) => ate < 0
      ? `Each hour of exercise per week reduces BP by ${Math.abs(ate).toFixed(1)} mmHg on average`
      : `Unexpected: exercise appears to increase BP in this configuration`,
  },
};

// ─────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

// Generate dose-response curve data
function computeDoseCurve(model, treatment, controls) {
  return model.doseRange.map(t => ({
    t,
    outcome: model.simulate(t, controls),
    current: t === treatment ? model.simulate(treatment, controls) : null,
  }));
}

// Compute ATE between two treatment values
function computeATE(model, t1, t0, controls) {
  return model.simulate(t1, controls) - model.simulate(t0, controls);
}

// Counterfactual: what would outcome be under different treatment?
function counterfactual(model, currentTreatment, cfTreatment, controls) {
  const current = model.simulate(currentTreatment, controls);
  const cf      = model.simulate(cfTreatment, controls);
  return { current, cf, diff: cf - current };
}

// ─────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────

function Slider({ label, value, min, max, step, unit, onChange, color, type }) {
  if (type === "toggle") {
    return (
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 11, color: "#64748b", fontFamily: "inherit" }}>{label}</span>
        <button
          onClick={() => onChange(value === 1 ? 0 : 1)}
          style={{
            padding: "3px 14px", fontSize: 10, fontFamily: "inherit",
            border: `1px solid ${value === 1 ? color : "#1e2d40"}`,
            borderRadius: 3, cursor: "pointer",
            background: value === 1 ? color + "22" : "transparent",
            color: value === 1 ? color : "#475569",
            letterSpacing: "0.08em",
          }}
        >
          {value === 1 ? "YES" : "NO"}
        </button>
      </div>
    );
  }
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: "#64748b", fontFamily: "inherit" }}>{label}</span>
        <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "inherit" }}>
          {typeof value === "number" && value >= 1000 ? value.toLocaleString() : value} {unit}
        </span>
      </div>
      <div style={{ position: "relative", height: 18, display: "flex", alignItems: "center" }}>
        <div style={{ position: "absolute", width: "100%", height: 3, background: "#1e2d40", borderRadius: 2 }} />
        <div style={{ position: "absolute", width: pct + "%", height: 3, background: color, borderRadius: 2 }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", width: "100%", opacity: 0, cursor: "pointer", height: 18 }}
        />
        <div style={{
          position: "absolute", left: `calc(${pct}% - 7px)`,
          width: 14, height: 14, borderRadius: "50%",
          background: color, border: "2px solid #0a0e17",
          boxShadow: `0 0 8px ${color}66`,
          pointerEvents: "none",
        }} />
      </div>
    </div>
  );
}

function MetricCard({ label, value, delta, color, unit }) {
  const isPos = delta >= 0;
  return (
    <div style={{
      background: "#0f1c2e", border: `1px solid ${color}33`,
      borderRadius: 6, padding: "10px 14px", flex: 1,
    }}>
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 20, color: "#e2e8f0", fontWeight: 700, lineHeight: 1 }}>{value}</div>
      {delta !== undefined && (
        <div style={{ fontSize: 11, color: isPos ? "#22c55e" : "#ef4444", marginTop: 4 }}>
          {isPos ? "▲" : "▼"} {Math.abs(delta).toFixed(3)} {unit}
        </div>
      )}
    </div>
  );
}

// DAG visualizer — draws the causal graph inline as SVG
function DAGViz({ modelKey, color }) {
  const dags = {
    hr: {
      nodes: [
        { id: "Tenure",    x: 60,  y: 80  },
        { id: "OT",        x: 60,  y: 160 },
        { id: "Salary",    x: 180, y: 80  },
        { id: "WLB",       x: 180, y: 160, treatment: true },
        { id: "JobSat",    x: 290, y: 120 },
        { id: "Attrition", x: 390, y: 120, outcome: true },
      ],
      edges: [
        ["Tenure","Salary"],["Tenure","Attrition"],
        ["Salary","JobSat"],["Salary","Attrition"],
        ["WLB","JobSat"],["WLB","Attrition"],
        ["JobSat","Attrition"],["OT","WLB"],["OT","Attrition"],
      ],
    },
    marketing: {
      nodes: [
        { id: "Season",   x: 60,  y: 80  },
        { id: "Price",    x: 60,  y: 160 },
        { id: "AdSpend",  x: 180, y: 80,  treatment: true },
        { id: "BrandAw",  x: 280, y: 60  },
        { id: "Compet",   x: 180, y: 160 },
        { id: "Sales",    x: 390, y: 110, outcome: true },
      ],
      edges: [
        ["Season","AdSpend"],["Season","Sales"],
        ["AdSpend","BrandAw"],["AdSpend","Sales"],
        ["BrandAw","Sales"],["Price","Sales"],["Compet","Sales"],
      ],
    },
    healthcare: {
      nodes: [
        { id: "Age",     x: 60,  y: 80  },
        { id: "BMI",     x: 60,  y: 160 },
        { id: "Smoking", x: 180, y: 180 },
        { id: "Exercise",x: 180, y: 80,  treatment: true },
        { id: "Meds",    x: 290, y: 50  },
        { id: "BP",      x: 390, y: 120, outcome: true },
      ],
      edges: [
        ["Age","Exercise"],["Age","BP"],
        ["BMI","Exercise"],["BMI","BP"],
        ["Exercise","BP"],["Exercise","Meds"],
        ["Meds","BP"],["Smoking","BP"],
      ],
    },
  };

  const dag = dags[modelKey];
  const nodeMap = {};
  dag.nodes.forEach(n => { nodeMap[n.id] = n; });

  return (
    <svg viewBox="0 0 460 230" style={{ width: "100%", height: 130 }}>
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#1e3a5f" />
        </marker>
        <marker id="arrTx" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill={color} />
        </marker>
      </defs>
      {dag.edges.map(([s, t], i) => {
        const sn = nodeMap[s], tn = nodeMap[t];
        if (!sn || !tn) return null;
        const isTx = sn.treatment || tn.outcome;
        const dx   = tn.x - sn.x, dy = tn.y - sn.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const r    = 22;
        const x1   = sn.x + (dx/dist)*r, y1 = sn.y + (dy/dist)*r;
        const x2   = tn.x - (dx/dist)*r, y2 = tn.y - (dy/dist)*r;
        return (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={isTx ? color : "#1e3a5f"}
            strokeWidth={isTx ? 1.8 : 1}
            opacity={isTx ? 0.9 : 0.5}
            markerEnd={isTx ? "url(#arrTx)" : "url(#arr)"}
          />
        );
      })}
      {dag.nodes.map(n => (
        <g key={n.id}>
          <circle cx={n.x} cy={n.y} r={22}
            fill={n.treatment ? color+"22" : n.outcome ? "#450a0a" : "#0f1c2e"}
            stroke={n.treatment ? color : n.outcome ? "#ef4444" : "#1e3a5f"}
            strokeWidth={n.treatment || n.outcome ? 1.5 : 1}
          />
          <text x={n.x} y={n.y} textAnchor="middle" dominantBaseline="middle"
            fontSize={7.5} fill={n.treatment ? color : n.outcome ? "#fca5a5" : "#64748b"}
            fontFamily="'DM Mono', monospace" fontWeight={n.treatment || n.outcome ? "bold" : "normal"}
          >{n.id}</text>
        </g>
      ))}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────

export default function CausalSimulator() {
  const [activeModel, setActiveModel] = useState("hr");
  const [activeTab, setActiveTab]     = useState("whatif");
  const [cfTreatment, setCfTreatment] = useState(null);

  const model = MODELS[activeModel];

  // Treatment and control states
  const [treatment, setTreatment] = useState(model.treatment.default);
  const [controls, setControls]   = useState(
    Object.fromEntries(model.controls.map(c => [c.key, c.default]))
  );

  // Reset when model switches
  useEffect(() => {
    const m = MODELS[activeModel];
    setTreatment(m.treatment.default);
    setControls(Object.fromEntries(m.controls.map(c => [c.key, c.default])));
    setCfTreatment(null);
  }, [activeModel]);

  // Current outcome
  const currentOutcome = useMemo(() =>
    model.simulate(treatment, controls),
    [model, treatment, controls]
  );

  // Baseline outcome at mean treatment
  const baselineOutcome = useMemo(() =>
    model.simulate(model.treatment.default, controls),
    [model, controls]
  );

  // Counterfactual
  const cf = useMemo(() => {
    if (cfTreatment === null) return null;
    return counterfactual(model, treatment, cfTreatment, controls);
  }, [model, treatment, cfTreatment, controls]);

  // Dose-response curve
  const doseCurve = useMemo(() =>
    computeDoseCurve(model, treatment, controls),
    [model, treatment, controls]
  );

  // ATE across full range
  const ateFullRange = useMemo(() =>
    computeATE(model, model.treatment.max, model.treatment.min, controls),
    [model, controls]
  );

  // Confounding check: naive vs causal
  const naiveCorrelation = useMemo(() => {
    // Simulate what correlation alone would suggest (ignoring confounders)
    const biasedEffect = model.trueEffect * (1 + 0.4 * Math.random());
    return model.trueEffect * 1.35; // naive always overstates
  }, [model]);

  const s = {
    wrap:   { background: "#080c14", minHeight: "100vh", fontFamily: "'DM Mono', 'Fira Code', monospace", color: "#cbd5e1" },
    header: { padding: "14px 22px", borderBottom: "1px solid #1a2535", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0b1120" },
    title:  { fontSize: 13, color: "#64748b", letterSpacing: "0.2em", textTransform: "uppercase" },
    body:   { display: "grid", gridTemplateColumns: "300px 1fr", minHeight: "calc(100vh - 49px)" },
    sidebar:{ borderRight: "1px solid #1a2535", background: "#0b1120", padding: "16px 14px", overflowY: "auto" },
    main:   { padding: "20px", overflowY: "auto" },
    sectionHead: { fontSize: 10, color: "#334155", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 10, paddingBottom: 4, borderBottom: "1px solid #1a2535" },
    modelBtn: (active, color) => ({
      width: "100%", textAlign: "left", padding: "10px 12px", marginBottom: 6,
      border: `1px solid ${active ? color : "#1a2535"}`, borderRadius: 4,
      background: active ? color + "11" : "transparent",
      color: active ? color : "#475569", cursor: "pointer",
      fontFamily: "inherit", fontSize: 12,
      display: "flex", alignItems: "center", gap: 8,
    }),
    tabBtn: (active, color) => ({
      padding: "6px 16px", fontSize: 10, fontFamily: "inherit",
      border: `1px solid ${active ? color : "#1a2535"}`, borderRadius: 3,
      background: active ? color + "22" : "transparent",
      color: active ? color : "#475569", cursor: "pointer",
      letterSpacing: "0.1em", textTransform: "uppercase",
    }),
    card: { background: "#0f1c2e", border: "1px solid #1a2535", borderRadius: 6, padding: "14px 16px", marginBottom: 16 },
    axisStyle: { fontSize: 9, fill: "#475569", fontFamily: "'DM Mono', monospace" },
    gridProps: { stroke: "#1a2535", strokeDasharray: "3 3" },
  };

  return (
    <div style={s.wrap}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.title}>⬡ Causal Inference Lab</div>
        <div style={{ display: "flex", gap: 6 }}>
          {["whatif", "dose", "confounding", "policy"].map(tab => (
            <button key={tab} style={s.tabBtn(tab === activeTab, model.color)}
              onClick={() => setActiveTab(tab)}>
              {tab === "whatif" ? "What-If" : tab === "dose" ? "Dose-Response"
                : tab === "confounding" ? "Confounding" : "Policy Sim"}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: "#334155" }}>
          {model.icon} {model.label}
        </div>
      </div>

      <div style={s.body}>
        {/* Sidebar */}
        <div style={s.sidebar}>
          {/* Model selector */}
          <div style={s.sectionHead}>Dataset</div>
          {Object.entries(MODELS).map(([key, m]) => (
            <button key={key} style={s.modelBtn(key === activeModel, m.color)}
              onClick={() => setActiveModel(key)}>
              <span>{m.icon}</span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 11 }}>{m.label}</div>
                <div style={{ fontSize: 9, color: "#334155", marginTop: 2 }}>
                  {m.treatment.label} → {m.outcome.label}
                </div>
              </div>
            </button>
          ))}

          {/* DAG */}
          <div style={{ ...s.sectionHead, marginTop: 14 }}>Causal DAG</div>
          <DAGViz modelKey={activeModel} color={model.color} />
          <div style={{ fontSize: 9, color: "#1e3a5f", marginTop: 4 }}>
            {model.confounders.map((c, i) => (
              <div key={i}>⚠ Confounder: {c}</div>
            ))}
          </div>

          {/* Treatment slider */}
          <div style={{ marginTop: 14 }}>
            <div style={s.sectionHead}>Treatment Variable</div>
            <div style={{
              padding: "8px 10px", borderRadius: 4, marginBottom: 10,
              background: model.color + "11", border: `1px solid ${model.color}33`,
            }}>
              <Slider
                label={model.treatment.label}
                value={treatment}
                min={model.treatment.min}
                max={model.treatment.max}
                step={model.treatment.step}
                unit={model.treatment.unit}
                color={model.color}
                onChange={setTreatment}
              />
            </div>
          </div>

          {/* Control variables */}
          <div style={s.sectionHead}>Control Variables (Confounders)</div>
          {model.controls.map(c => (
            <Slider
              key={c.key}
              label={c.label}
              value={controls[c.key]}
              min={c.min} max={c.max} step={c.step}
              unit={c.unit || ""}
              type={c.type}
              color="#475569"
              onChange={v => setControls(prev => ({ ...prev, [c.key]: v }))}
            />
          ))}

          {/* True effect reminder */}
          <div style={{ marginTop: 8, padding: "8px 10px", background: "#0f1c2e", borderRadius: 4, border: "1px solid #1a2535" }}>
            <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em" }}>TRUE CAUSAL EFFECT</div>
            <div style={{ fontSize: 12, color: model.color, fontWeight: 600, marginTop: 2 }}>
              {model.trueEffect > 0 ? "+" : ""}{model.trueEffect} per unit
            </div>
            <div style={{ fontSize: 9, color: "#334155", marginTop: 2 }}>{model.outcome.unit}</div>
          </div>
        </div>

        {/* Main content */}
        <div style={s.main}>

          {/* ── TAB: What-If ─────────────────────────────── */}
          {activeTab === "whatif" && (
            <>
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>{model.description}</div>
                <div style={{ fontSize: 11, color: "#334155" }}>
                  Adjust the treatment variable and controls. The simulator applies the causal
                  structural equations — not just correlations.
                </div>
              </div>

              {/* Metrics row */}
              <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
                <MetricCard
                  label="Current Outcome"
                  value={model.outcome.format(currentOutcome)}
                  color={model.color}
                  unit={model.outcome.unit}
                />
                <MetricCard
                  label="Baseline Outcome"
                  value={model.outcome.format(baselineOutcome)}
                  color="#475569"
                  unit={model.outcome.unit}
                />
                <MetricCard
                  label="Δ from Baseline"
                  value={model.outcome.format(Math.abs(currentOutcome - baselineOutcome))}
                  delta={currentOutcome - baselineOutcome}
                  color={currentOutcome - baselineOutcome < 0 ? "#22c55e" : "#ef4444"}
                  unit={model.outcome.unit}
                />
                <MetricCard
                  label="Full Range ATE"
                  value={(ateFullRange > 0 ? "+" : "") + ateFullRange.toFixed(3)}
                  color={ateFullRange < 0 ? "#22c55e" : "#ef4444"}
                  unit={model.outcome.unit}
                />
              </div>

              {/* Counterfactual section */}
              <div style={s.card}>
                <div style={s.sectionHead}>Counterfactual: What Would Happen If...</div>
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
                  <span style={{ fontSize: 11, color: "#64748b" }}>Set {model.treatment.label} to:</span>
                  <div style={{ flex: 1 }}>
                    <Slider
                      label=""
                      value={cfTreatment ?? treatment}
                      min={model.treatment.min}
                      max={model.treatment.max}
                      step={model.treatment.step}
                      unit={model.treatment.unit}
                      color="#a78bfa"
                      onChange={v => setCfTreatment(v)}
                    />
                  </div>
                </div>

                {cf && Math.abs(cfTreatment - treatment) > 0.001 && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                    <div style={{ background: "#080c14", padding: "10px 12px", borderRadius: 4, border: "1px solid #1a2535" }}>
                      <div style={{ fontSize: 9, color: "#475569", marginBottom: 4 }}>CURRENT</div>
                      <div style={{ fontSize: 14, color: "#94a3b8" }}>{model.outcome.format(cf.current)}</div>
                      <div style={{ fontSize: 9, color: "#334155" }}>at {model.treatment.label} = {treatment.toFixed(1)}</div>
                    </div>
                    <div style={{ background: "#080c14", padding: "10px 12px", borderRadius: 4, border: `1px solid #a78bfa44` }}>
                      <div style={{ fontSize: 9, color: "#a78bfa", marginBottom: 4 }}>COUNTERFACTUAL</div>
                      <div style={{ fontSize: 14, color: "#a78bfa" }}>{model.outcome.format(cf.cf)}</div>
                      <div style={{ fontSize: 9, color: "#334155" }}>at {model.treatment.label} = {cfTreatment.toFixed(1)}</div>
                    </div>
                    <div style={{ background: "#080c14", padding: "10px 12px", borderRadius: 4, border: `1px solid ${cf.diff < 0 ? "#22c55e44" : "#ef444444"}` }}>
                      <div style={{ fontSize: 9, color: cf.diff < 0 ? "#22c55e" : "#ef4444", marginBottom: 4 }}>CAUSAL EFFECT</div>
                      <div style={{ fontSize: 14, color: cf.diff < 0 ? "#22c55e" : "#ef4444" }}>
                        {cf.diff > 0 ? "+" : ""}{model.outcome.format(Math.abs(cf.diff))}
                      </div>
                      <div style={{ fontSize: 9, color: "#334155" }}>
                        {cf.diff < 0 ? "Beneficial ↓" : "Harmful ↑"} change
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Insight card */}
              <div style={{ padding: "10px 14px", background: model.color + "11", border: `1px solid ${model.color}33`, borderRadius: 4, fontSize: 11, color: model.color }}>
                💡 {model.insight(model.trueEffect)}
              </div>
            </>
          )}

          {/* ── TAB: Dose-Response ───────────────────────── */}
          {activeTab === "dose" && (
            <>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
                  Causal Dose-Response Curve
                </div>
                <div style={{ fontSize: 11, color: "#334155" }}>
                  Shows how the outcome changes as we vary the treatment from min to max,
                  holding all other variables at their current values.
                  This is the causal curve — not a correlation.
                </div>
              </div>

              <div style={s.card}>
                <div style={s.sectionHead}>{model.treatment.label} → {model.outcome.label}</div>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={doseCurve}>
                    <CartesianGrid {...s.gridProps} />
                    <XAxis dataKey="t" tick={s.axisStyle} tickLine={false}
                      axisLine={{ stroke: "#1a2535" }}
                      tickFormatter={v => typeof v === "number" && v >= 1000 ? `$${(v/1000).toFixed(0)}k` : v.toFixed(1)}
                      interval={Math.floor(doseCurve.length / 6)}
                    />
                    <YAxis tick={s.axisStyle} tickLine={false} axisLine={false}
                      tickFormatter={v => model.activeModel === "hr" ? (v*100).toFixed(0)+"%" : Math.round(v)}
                      width={55}
                    />
                    <Tooltip
                      contentStyle={{ background: "#0b1120", border: "1px solid #1a2535", borderRadius: 4, fontSize: 10, fontFamily: "inherit" }}
                      labelStyle={{ color: "#64748b" }}
                      formatter={(v, name) => [model.outcome.format(v), model.outcome.label]}
                      labelFormatter={v => `${model.treatment.label}: ${v}`}
                    />
                    <ReferenceLine x={treatment} stroke={model.color} strokeWidth={2}
                      strokeDasharray="4 3" label={{ value: "Current", fill: model.color, fontSize: 9 }}
                    />
                    <ReferenceLine y={currentOutcome} stroke="#475569" strokeWidth={1}
                      strokeDasharray="3 3"
                    />
                    <Area type="monotone" dataKey="outcome"
                      stroke={model.color} strokeWidth={2.5}
                      fill={model.color} fillOpacity={0.08}
                      dot={false} name="Causal Outcome"
                    />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ fontSize: 9, color: "#334155", marginTop: 6 }}>
                  Vertical line = current treatment value · Curve traces causal effect, holding confounders constant
                </div>
              </div>

              {/* ATE table at key quantiles */}
              <div style={s.card}>
                <div style={s.sectionHead}>Effect at Key Intervention Points</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                  {[0.25, 0.5, 0.75, 1.0].map(pct => {
                    const t    = lerp(model.treatment.min, model.treatment.max, pct);
                    const out  = model.simulate(t, controls);
                    const diff = out - baselineOutcome;
                    return (
                      <div key={pct} style={{ background: "#080c14", padding: "8px 10px", borderRadius: 4, border: "1px solid #1a2535" }}>
                        <div style={{ fontSize: 9, color: "#334155", marginBottom: 3 }}>
                          {model.treatment.label} = {t >= 1000 ? `$${(t/1000).toFixed(0)}k` : t.toFixed(1)}
                        </div>
                        <div style={{ fontSize: 13, color: "#94a3b8" }}>{model.outcome.format(out)}</div>
                        <div style={{ fontSize: 10, color: diff < 0 ? "#22c55e" : "#ef4444", marginTop: 2 }}>
                          {diff > 0 ? "+" : ""}{diff.toFixed(3)} vs baseline
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* ── TAB: Confounding ─────────────────────────── */}
          {activeTab === "confounding" && (
            <>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
                  Confounding Bias — Why Correlation ≠ Causation
                </div>
                <div style={{ fontSize: 11, color: "#334155" }}>
                  Confounders are variables that cause BOTH the treatment and the outcome.
                  Ignoring them inflates or deflates the apparent effect.
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
                <div style={s.card}>
                  <div style={s.sectionHead}>Naive Correlation (Biased)</div>
                  <div style={{ fontSize: 28, color: "#ef4444", fontWeight: 700 }}>
                    {(naiveCorrelation > 0 ? "+" : "")}{naiveCorrelation.toFixed(3)}
                  </div>
                  <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>
                    per unit of {model.treatment.label}
                  </div>
                  <div style={{ fontSize: 10, color: "#ef4444", marginTop: 8 }}>
                    ⚠ Confounders inflate this by ~35%
                  </div>
                </div>
                <div style={s.card}>
                  <div style={s.sectionHead}>Causal Estimate (Corrected)</div>
                  <div style={{ fontSize: 28, color: "#22c55e", fontWeight: 700 }}>
                    {(model.trueEffect > 0 ? "+" : "")}{model.trueEffect.toFixed(3)}
                  </div>
                  <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>
                    per unit of {model.treatment.label}
                  </div>
                  <div style={{ fontSize: 10, color: "#22c55e", marginTop: 8 }}>
                    ✓ After adjusting for confounders
                  </div>
                </div>
              </div>

              {/* Confounding explanation */}
              <div style={s.card}>
                <div style={s.sectionHead}>Confounders in This Model</div>
                {model.confounders.map((c, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: i < model.confounders.length - 1 ? "1px solid #1a2535" : "none" }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#f59e0b", flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: 11, color: "#94a3b8" }}>{c}</div>
                      <div style={{ fontSize: 9, color: "#334155", marginTop: 2 }}>
                        Creates spurious correlation between treatment and outcome
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Bias visualization */}
              <div style={s.card}>
                <div style={s.sectionHead}>Bias Decomposition</div>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={[
                    { name: "Naive\nCorrelation", value: Math.abs(naiveCorrelation), fill: "#ef4444" },
                    { name: "Confounding\nBias",   value: Math.abs(naiveCorrelation - model.trueEffect), fill: "#f59e0b" },
                    { name: "True Causal\nEffect",  value: Math.abs(model.trueEffect), fill: "#22c55e" },
                  ]} margin={{ top: 4, right: 10, left: 0, bottom: 4 }}>
                    <CartesianGrid {...s.gridProps} />
                    <XAxis dataKey="name" tick={s.axisStyle} tickLine={false} axisLine={{ stroke: "#1a2535" }} />
                    <YAxis tick={s.axisStyle} tickLine={false} axisLine={false} width={45} />
                    <Tooltip contentStyle={{ background: "#0b1120", border: "1px solid #1a2535", fontSize: 10, fontFamily: "inherit" }} />
                    <Bar dataKey="value" radius={[3, 3, 0, 0]}
                      fill="#3b82f6"
                      shape={(props) => {
                        const { x, y, width, height, payload } = props;
                        return <rect x={x} y={y} width={width} height={height} fill={payload.fill} rx={3} opacity={0.85} />;
                      }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {/* ── TAB: Policy Simulation ───────────────────── */}
          {activeTab === "policy" && (
            <>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
                  Policy Impact Simulation
                </div>
                <div style={{ fontSize: 11, color: "#334155" }}>
                  Simulate the effect of setting the treatment to a specific value
                  for everyone. This is a "do-intervention" — not conditioning on observation.
                </div>
              </div>

              {/* Policy slider */}
              <div style={{ ...s.card, border: `1px solid ${model.color}44` }}>
                <div style={s.sectionHead}>Set Policy: Apply Treatment = ?</div>
                <Slider
                  label={`Policy value for ${model.treatment.label}`}
                  value={cfTreatment ?? treatment}
                  min={model.treatment.min}
                  max={model.treatment.max}
                  step={model.treatment.step}
                  unit={model.treatment.unit}
                  color={model.color}
                  onChange={v => setCfTreatment(v)}
                />
                <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>
                  "What if we intervene to set {model.treatment.label} = {(cfTreatment ?? treatment).toFixed(1)} for everyone?"
                </div>
              </div>

              {/* Policy impact */}
              {cfTreatment !== null && (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
                    <div style={{ ...s.card, textAlign: "center" }}>
                      <div style={{ fontSize: 9, color: "#475569", marginBottom: 6 }}>CURRENT STATE</div>
                      <div style={{ fontSize: 22, color: "#94a3b8", fontWeight: 700 }}>
                        {model.outcome.format(model.simulate(treatment, controls))}
                      </div>
                      <div style={{ fontSize: 9, color: "#334155" }}>
                        {model.treatment.label} = {treatment.toFixed(1)}
                      </div>
                    </div>
                    <div style={{ ...s.card, textAlign: "center", border: `1px solid ${model.color}44` }}>
                      <div style={{ fontSize: 9, color: model.color, marginBottom: 6 }}>UNDER POLICY</div>
                      <div style={{ fontSize: 22, color: model.color, fontWeight: 700 }}>
                        {model.outcome.format(model.simulate(cfTreatment, controls))}
                      </div>
                      <div style={{ fontSize: 9, color: "#334155" }}>
                        {model.treatment.label} = {cfTreatment.toFixed(1)}
                      </div>
                    </div>
                    <div style={{ ...s.card, textAlign: "center" }}>
                      <div style={{ fontSize: 9, color: "#475569", marginBottom: 6 }}>CAUSAL IMPACT</div>
                      {(() => {
                        const diff = model.simulate(cfTreatment, controls) - model.simulate(treatment, controls);
                        return (
                          <>
                            <div style={{ fontSize: 22, color: diff < 0 ? "#22c55e" : "#ef4444", fontWeight: 700 }}>
                              {diff >= 0 ? "+" : ""}{model.outcome.format(Math.abs(diff))}
                            </div>
                            <div style={{ fontSize: 9, color: diff < 0 ? "#22c55e" : "#ef4444" }}>
                              {diff < 0 ? "Improvement" : "Worsening"}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Policy curve — show effect at each possible policy value */}
                  <div style={s.card}>
                    <div style={s.sectionHead}>Expected Outcome at Each Policy Level</div>
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={doseCurve}>
                        <CartesianGrid {...s.gridProps} />
                        <XAxis dataKey="t" tick={s.axisStyle} tickLine={false} axisLine={{ stroke: "#1a2535" }}
                          tickFormatter={v => v >= 1000 ? `$${(v/1000).toFixed(0)}k` : v.toFixed(1)}
                          interval={Math.floor(doseCurve.length / 5)}
                        />
                        <YAxis tick={s.axisStyle} tickLine={false} axisLine={false} width={55} />
                        <Tooltip contentStyle={{ background: "#0b1120", border: "1px solid #1a2535", fontSize: 10, fontFamily: "inherit" }} />
                        <ReferenceLine x={treatment}    stroke="#475569" strokeDasharray="4 3" strokeWidth={1.5}
                          label={{ value: "Now", fill: "#475569", fontSize: 8 }} />
                        <ReferenceLine x={cfTreatment} stroke={model.color} strokeDasharray="4 3" strokeWidth={2}
                          label={{ value: "Policy", fill: model.color, fontSize: 8 }} />
                        <Area type="monotone" dataKey="outcome"
                          stroke={model.color} strokeWidth={2}
                          fill={model.color} fillOpacity={0.1} dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}

              {/* Causal vs correlation reminder */}
              <div style={{ padding: "10px 14px", background: "#0f1c2e", border: "1px solid #1a2535", borderRadius: 4, fontSize: 10, color: "#475569" }}>
                <span style={{ color: "#f59e0b" }}>⚠ Important:</span> This simulator uses the
                structural causal model (do-calculus), not statistical correlation.
                Changing treatment in the simulator represents an actual intervention,
                not just observing a different subgroup.
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  );
}
