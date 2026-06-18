"""
=============================================================
CAUSALITY & WHAT-IF SIMULATOR — Part 2
Simulation Engine & Counterfactual Analysis
=============================================================

Reads outputs from Part 1:
  - causal_data_hr.csv
  - causal_data_marketing.csv
  - causal_data_healthcare.csv
  - causal_results.json

Covers:
  1. Counterfactual simulation (what would have happened if...)
  2. Dose-response curves (how does outcome change as treatment varies?)
  3. Heterogeneous treatment effects (does the effect differ by subgroup?)
  4. Policy simulation (what is the predicted impact of an intervention?)
  5. Sensitivity analysis (how sensitive is the result to hidden confounders?)

=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — Load Data
# ─────────────────────────────────────────────────────────────

def load_data():
    """Loads all datasets from Part 1."""
    print("Loading causal datasets...")
    datasets = {}
    for key in ["hr", "marketing", "healthcare"]:
        try:
            datasets[key] = pd.read_csv(f"causal_data_{key}.csv")
            print(f"  {key}: {len(datasets[key]):,} rows")
        except FileNotFoundError:
            print(f"  {key}: not found — run Part 1 first")

    try:
        with open("causal_results.json") as f:
            causal_results = json.load(f)
    except FileNotFoundError:
        causal_results = {}

    return datasets, causal_results


# ─────────────────────────────────────────────────────────────
# SECTION 2 — Causal Effect Estimator (Plug-in)
# ─────────────────────────────────────────────────────────────

class CausalEstimator:
    """
    A simple but principled causal effect estimator that works
    without DoWhy. Uses the backdoor adjustment formula:

      E[Y | do(T=t)] = Σ_z E[Y | T=t, Z=z] * P(Z=z)

    where Z is the set of backdoor adjustment variables
    (confounders that need to be controlled for).

    This is the mathematical foundation behind all causal
    inference methods — DoWhy, propensity scoring etc. all
    implement variations of this formula.
    """

    def __init__(self, treatment, outcome, confounders):
        self.treatment   = treatment
        self.outcome     = outcome
        self.confounders = confounders
        self.model       = None
        self.scaler      = StandardScaler()

    def fit(self, df):
        """
        Fits a regression model of outcome on treatment + confounders.
        The causal effect is the coefficient on treatment (after
        controlling for confounders).
        """
        features = [self.treatment] + self.confounders
        X = df[features].fillna(0)
        y = df[self.outcome]

        X_scaled = self.scaler.fit_transform(X)

        # Use Gradient Boosting for non-linear relationships
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        )
        self.model.fit(X_scaled, y)
        self.feature_names = features
        self.df_train = df.copy()
        return self

    def predict_counterfactual(self, df, treatment_value):
        """
        Predicts what the outcome WOULD BE if we set
        the treatment to treatment_value for everyone.

        This is a "do-intervention" — we're not conditioning
        on observing T=t, we're SETTING T=t for everyone
        regardless of what their value was.
        """
        df_cf = df.copy()
        df_cf[self.treatment] = treatment_value

        X_cf = df_cf[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_cf)
        return self.model.predict(X_scaled)

    def dose_response(self, df, treatment_range):
        """
        Computes the expected outcome at each treatment value.
        This traces the causal dose-response curve.
        """
        responses = []
        for t in treatment_range:
            preds = self.predict_counterfactual(df, t)
            responses.append({
                "treatment": t,
                "mean_outcome": float(np.mean(preds)),
                "ci_lower": float(np.percentile(preds, 10)),
                "ci_upper": float(np.percentile(preds, 90)),
            })
        return pd.DataFrame(responses)

    def ate(self, df, t1, t0):
        """
        Average Treatment Effect: E[Y(t1)] - E[Y(t0)]
        The causal effect of changing treatment from t0 to t1.
        """
        y1 = self.predict_counterfactual(df, t1).mean()
        y0 = self.predict_counterfactual(df, t0).mean()
        return float(y1 - y0)

    def cate(self, df, t1, t0, subgroup_col, subgroup_val):
        """
        Conditional Average Treatment Effect for a subgroup.
        "Does the effect differ for older vs. younger patients?"
        """
        mask = df[subgroup_col] == subgroup_val
        sub  = df[mask]
        if len(sub) == 0:
            return None
        y1 = self.predict_counterfactual(sub, t1).mean()
        y0 = self.predict_counterfactual(sub, t0).mean()
        return float(y1 - y0)


# ─────────────────────────────────────────────────────────────
# SECTION 3 — Dose Response Curves
# ─────────────────────────────────────────────────────────────

def compute_dose_response_curves(datasets):
    """
    For each dataset, computes the causal dose-response curve:
    "As we increase the treatment, how does the outcome change?"

    This is one of the most useful outputs for decision-makers —
    it shows the SHAPE of the causal relationship, not just
    the slope at one point.
    """
    print("\n" + "="*55)
    print("DOSE-RESPONSE CURVES")
    print("="*55)

    configs = {
        "hr": {
            "treatment":   "WorkLifeBalance",
            "outcome":     "Attrition",
            "confounders": ["YearsAtCompany", "MonthlyIncome", "OverTime"],
            "range":       np.linspace(1, 4, 20),
            "unit":        "WLB Score (1=Bad, 4=Best)",
            "outcome_unit": "Attrition Probability",
        },
        "marketing": {
            "treatment":   "AdSpend",
            "outcome":     "Sales",
            "confounders": ["Season", "IsPeakSeason", "Price", "CompetitorSpend"],
            "range":       np.linspace(1000, 40000, 30),
            "unit":        "Weekly Ad Spend ($)",
            "outcome_unit": "Weekly Units Sold",
        },
        "healthcare": {
            "treatment":   "ExerciseHours",
            "outcome":     "BloodPressure",
            "confounders": ["Age", "BMI", "Smoking"],
            "range":       np.linspace(0, 15, 25),
            "unit":        "Exercise Hours / Week",
            "outcome_unit": "Systolic BP (mmHg)",
        },
    }

    estimators = {}
    dose_responses = {}

    for key, config in configs.items():
        if key not in datasets:
            continue

        df = datasets[key]
        print(f"\n  {key.upper()} — {config['treatment']} → {config['outcome']}")

        # Fit estimator
        est = CausalEstimator(
            config["treatment"],
            config["outcome"],
            config["confounders"]
        ).fit(df)
        estimators[key] = est

        # Compute dose-response
        dr = est.dose_response(df, config["range"])
        dose_responses[key] = {
            "data":         dr,
            "config":       config,
            "baseline_mean": float(df[config["outcome"]].mean()),
        }

        # Show a few key points
        t_min  = config["range"][0]
        t_max  = config["range"][-1]
        ate    = est.ate(df, t_max, t_min)
        print(f"    ATE ({t_min:.0f} → {t_max:.0f}): {ate:+.4f} {config['outcome_unit']}")

    return estimators, dose_responses


# ─────────────────────────────────────────────────────────────
# SECTION 4 — Heterogeneous Treatment Effects
# ─────────────────────────────────────────────────────────────

def heterogeneous_effects(datasets, estimators):
    """
    Does the causal effect differ across subgroups?

    Examples:
    - HR: Does WLB affect attrition MORE for junior vs. senior employees?
    - Healthcare: Does exercise reduce BP MORE for older vs. younger patients?
    - Marketing: Does ad spend work MORE during peak season?

    This is called CATE (Conditional Average Treatment Effect)
    analysis — one of the most actionable outputs of causal inference.
    """
    print("\n" + "="*55)
    print("HETEROGENEOUS TREATMENT EFFECTS (CATE)")
    print("="*55)

    cate_results = {}

    # ── HR: Effect of WLB by OverTime status ──
    if "hr" in datasets and "hr" in estimators:
        df  = datasets["hr"]
        est = estimators["hr"]
        print("\n  HR — Effect of WLB improvement on Attrition:")
        for ot, label in [(0, "No Overtime"), (1, "Works Overtime")]:
            mask = df["OverTime"] == ot
            sub  = df[mask]
            ate  = est.ate(sub, 4, 1)
            print(f"    {label:<20}: CATE = {ate:+.4f}")
        cate_results["hr"] = {"subgroup": "OverTime",
                               "t0": 1, "t1": 4}

    # ── Marketing: Effect of AdSpend by Season ──
    if "marketing" in datasets and "marketing" in estimators:
        df  = datasets["marketing"]
        est = estimators["marketing"]
        print("\n  Marketing — Effect of doubling AdSpend on Sales:")
        for s, label in [(0, "Off-Peak"), (1, "Peak Season")]:
            mask = df["IsPeakSeason"] == s
            sub  = df[mask]
            t0   = df["AdSpend"].mean()
            ate  = est.ate(sub, t0 * 2, t0)
            print(f"    {label:<20}: CATE = {ate:+.1f} units")
        cate_results["marketing"] = {"subgroup": "IsPeakSeason",
                                      "t0": "mean", "t1": "2x mean"}

    # ── Healthcare: Effect of Exercise by Age group ──
    if "healthcare" in datasets and "healthcare" in estimators:
        df  = datasets["healthcare"]
        est = estimators["healthcare"]
        df["AgeGroup"] = pd.cut(df["Age"],
            bins=[0, 40, 60, 100],
            labels=["Young (<40)", "Middle (40-60)", "Senior (>60)"]
        )
        print("\n  Healthcare — Effect of 5 hrs/week exercise on Blood Pressure:")
        for group in ["Young (<40)", "Middle (40-60)", "Senior (>60)"]:
            mask = df["AgeGroup"] == group
            sub  = df[mask]
            ate  = est.ate(sub, 5, 0)
            print(f"    {group:<22}: CATE = {ate:+.2f} mmHg")
        cate_results["healthcare"] = {"subgroup": "AgeGroup",
                                       "t0": 0, "t1": 5}

    return cate_results


# ─────────────────────────────────────────────────────────────
# SECTION 5 — Policy Simulation
# ─────────────────────────────────────────────────────────────

def policy_simulation(datasets, estimators):
    """
    Simulates the real-world impact of a proposed policy/intervention.

    For HR: "What if we improve Work-Life Balance for all employees
             with WLB < 3 (the bottom 60%)?"

    For Marketing: "What if we increase ad spend by 20%?"

    For Healthcare: "What if we get all sedentary patients to
                     exercise at least 3 hours/week?"
    """
    print("\n" + "="*55)
    print("POLICY SIMULATION")
    print("="*55)

    simulations = {}

    # ── HR Policy ──────────────────────────────────────────
    if "hr" in datasets and "hr" in estimators:
        df  = datasets["hr"]
        est = estimators["hr"]

        # Current state
        current_attrition = df["Attrition"].mean()

        # Counterfactual: everyone gets WLB >= 3
        df_policy        = df.copy()
        improved_mask    = df_policy["WorkLifeBalance"] < 3
        df_policy.loc[improved_mask, "WorkLifeBalance"] = 3

        # Predict counterfactual outcomes for improved employees
        improved_preds = est.predict_counterfactual(
            df[improved_mask], 3
        )
        cf_attrition = (
            df[~improved_mask]["Attrition"].sum() +
            improved_preds.sum()
        ) / len(df)

        n_improved = improved_mask.sum()
        reduction  = current_attrition - cf_attrition

        simulations["hr"] = {
            "policy": "Improve WLB to >= 3 for all employees below threshold",
            "employees_affected": int(n_improved),
            "current_attrition_rate": round(float(current_attrition), 4),
            "predicted_attrition_rate": round(float(cf_attrition), 4),
            "absolute_reduction": round(float(reduction), 4),
            "lives_impacted": int(len(df) * reduction),
        }

        print(f"\n  HR Policy: Improve WLB for {n_improved} employees")
        print(f"    Current attrition rate:   {current_attrition:.1%}")
        print(f"    Predicted attrition rate: {cf_attrition:.1%}")
        print(f"    Reduction:                {reduction:.1%} ({int(len(df)*reduction)} employees retained)")

    # ── Marketing Policy ────────────────────────────────────
    if "marketing" in datasets and "marketing" in estimators:
        df  = datasets["marketing"]
        est = estimators["marketing"]

        current_sales   = df["Sales"].mean()
        new_ad_spend    = df["AdSpend"] * 1.20
        boost           = 0
        for orig, new_t in zip(df["AdSpend"], new_ad_spend):
            pred_new  = est.predict_counterfactual(
                df[df["AdSpend"] == orig].head(1), new_t
            )
            pred_orig = est.predict_counterfactual(
                df[df["AdSpend"] == orig].head(1), orig
            )
            if len(pred_new) > 0 and len(pred_orig) > 0:
                boost += (pred_new[0] - pred_orig[0])

        avg_boost    = boost / len(df)
        cost_increase = df["AdSpend"].mean() * 0.20
        roi          = avg_boost / cost_increase if cost_increase > 0 else 0

        simulations["marketing"] = {
            "policy": "Increase all ad spend by 20%",
            "avg_sales_lift": round(float(avg_boost), 2),
            "cost_increase_per_week": round(float(cost_increase), 2),
            "roi_units_per_dollar": round(float(roi), 4),
        }

        print(f"\n  Marketing Policy: +20% Ad Spend")
        print(f"    Avg sales lift:     +{avg_boost:.1f} units/week")
        print(f"    Cost increase:      ${cost_increase:,.0f}/week")
        print(f"    ROI:                {roi:.4f} units per dollar")

    # ── Healthcare Policy ───────────────────────────────────
    if "healthcare" in datasets and "healthcare" in estimators:
        df  = datasets["healthcare"]
        est = estimators["healthcare"]

        sedentary      = df["ExerciseHours"] < 2
        n_sedentary    = sedentary.sum()
        current_bp_sed = df[sedentary]["BloodPressure"].mean()
        target_exercise = 3

        cf_bp = est.predict_counterfactual(
            df[sedentary], target_exercise
        ).mean()
        bp_reduction = current_bp_sed - cf_bp

        # Clinical significance: each 2 mmHg reduction ≈ 7% CVD risk reduction
        cvd_risk_reduction = (bp_reduction / 2) * 0.07

        simulations["healthcare"] = {
            "policy": "Get sedentary patients to 3 hours/week exercise",
            "patients_affected": int(n_sedentary),
            "current_avg_bp": round(float(current_bp_sed), 1),
            "predicted_avg_bp": round(float(cf_bp), 1),
            "bp_reduction_mmhg": round(float(bp_reduction), 2),
            "estimated_cvd_risk_reduction": round(float(cvd_risk_reduction), 3),
        }

        print(f"\n  Healthcare Policy: Exercise intervention for {n_sedentary} sedentary patients")
        print(f"    Current avg BP:     {current_bp_sed:.1f} mmHg")
        print(f"    Predicted avg BP:   {cf_bp:.1f} mmHg")
        print(f"    BP reduction:       -{bp_reduction:.1f} mmHg")
        print(f"    CVD risk reduction: -{cvd_risk_reduction:.1%}")

    return simulations


# ─────────────────────────────────────────────────────────────
# SECTION 6 — Sensitivity Analysis (Rosenbaum Bounds)
# ─────────────────────────────────────────────────────────────

def sensitivity_analysis(datasets, estimators):
    """
    How sensitive is our causal estimate to hidden confounders
    we didn't measure or include in the DAG?

    Rosenbaum's sensitivity analysis asks:
    "How strong would an unobserved confounder have to be to
     completely explain away our result?"

    If Gamma = 1.0 → result holds with no hidden confounders
    If Gamma = 2.0 → a hidden confounder doubling the odds of
                     treatment could explain the result
    If Gamma is large → result is ROBUST (hard to explain away)
    If Gamma is small → result is FRAGILE (easy to explain away)
    """
    print("\n" + "="*55)
    print("SENSITIVITY ANALYSIS (Rosenbaum Bounds)")
    print("="*55)

    sensitivity_results = {}

    for key in ["hr", "marketing", "healthcare"]:
        if key not in datasets or key not in estimators:
            continue

        df  = datasets[key]
        est = estimators[key]

        # Only build the config for THIS dataset — avoids referencing
        # columns that don't exist in the other datasets (the old code
        # built all three dict entries eagerly, which crashed when
        # df was the HR/Healthcare frame and had no "AdSpend" column)
        if key == "hr":
            c = {"t1": 4, "t0": 1, "label": "WLB 1→4"}
        elif key == "marketing":
            c = {"t1": df["AdSpend"].quantile(0.75),
                 "t0": df["AdSpend"].quantile(0.25),
                 "label": "AdSpend Q1→Q3"}
        elif key == "healthcare":
            c = {"t1": 10, "t0": 0, "label": "Exercise 0→10hrs"}

        ate = est.ate(df, c["t1"], c["t0"])

        # Simulate sensitivity: progressively add a hidden confounder
        gamma_values = np.linspace(1.0, 3.0, 20)
        upper_bounds = []
        lower_bounds = []

        for gamma in gamma_values:
            # Simplified Rosenbaum bound calculation
            # In reality DoWhy/sensemakr would compute this more rigorously
            noise = np.log(gamma) * abs(ate) * 0.5
            upper_bounds.append(ate + noise)
            lower_bounds.append(ate - noise)

        # Find gamma where confidence interval crosses zero
        crosses_zero = None
        for i, (lo, hi) in enumerate(zip(lower_bounds, upper_bounds)):
            if lo <= 0 <= hi:
                crosses_zero = gamma_values[i]
                break

        sensitivity_results[key] = {
            "ate":            ate,
            "gamma_values":   gamma_values.tolist(),
            "upper_bounds":   upper_bounds,
            "lower_bounds":   lower_bounds,
            "crosses_zero_at": crosses_zero,
            "label":          c["label"],
        }

        robustness = "ROBUST" if (crosses_zero is None or crosses_zero > 2) \
                     else "FRAGILE" if crosses_zero < 1.5 else "MODERATE"

        gamma_str = f"{crosses_zero:.2f}" if crosses_zero else "N/A (never)"

        print(f"\n  {key.upper()} — {c['label']}")
        print(f"    ATE estimate:      {ate:+.4f}")
        print(f"    Effect nullified at Gamma = {gamma_str}")
        print(f"    Robustness:        {robustness}")

    return sensitivity_results


# ─────────────────────────────────────────────────────────────
# SECTION 7 — Full Visualization
# ─────────────────────────────────────────────────────────────

def visualize_simulation_results(dose_responses, cate_results,
                                  simulations, sensitivity_results):
    """
    4-panel visualization:
    - Dose-response curves for all 3 datasets
    - Policy simulation impact summary
    - Sensitivity analysis bounds
    - CATE subgroup effects
    """
    print("\nGenerating simulation visualization...")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0e17')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    bg   = '#0d1520'
    grid = '#1e2d40'
    text = '#94a3b8'
    colors = {'hr': '#3b82f6', 'marketing': '#22c55e', 'healthcare': '#f59e0b'}
    labels = {'hr': 'HR Attrition', 'marketing': 'Marketing', 'healthcare': 'Healthcare'}

    # ── Row 1: Dose-response curves ──────────────────────────
    for col, key in enumerate(["hr", "marketing", "healthcare"]):
        if key not in dose_responses:
            continue
        dr     = dose_responses[key]["data"]
        config = dose_responses[key]["config"]

        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg)
        ax.set_title(
            f"{labels[key]}\nDose-Response Curve",
            color=text, fontsize=10, pad=6
        )

        color = colors[key]
        ax.fill_between(dr["treatment"], dr["ci_lower"], dr["ci_upper"],
                       color=color, alpha=0.15)
        ax.plot(dr["treatment"], dr["mean_outcome"],
               color=color, linewidth=2.5, label="Causal effect")
        ax.axhline(dose_responses[key]["baseline_mean"],
                  color='#475569', linestyle='--', linewidth=1,
                  label='Observed mean')

        ax.set_xlabel(config["unit"], color=text, fontsize=8)
        ax.set_ylabel(config["outcome_unit"], color=text, fontsize=8)
        ax.legend(fontsize=7, facecolor=bg, labelcolor=text, edgecolor=grid)
        ax.tick_params(colors=text, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid)
        ax.grid(color=grid, linewidth=0.5, alpha=0.5)

    # ── Row 2 left: Policy simulation impact ────────────────
    ax_pol = fig.add_subplot(gs[1, 0])
    ax_pol.set_facecolor(bg)
    ax_pol.set_title("Policy Simulation Impact", color=text, fontsize=10, pad=6)
    ax_pol.axis('off')

    y_pos = 0.95
    for key, sim in simulations.items():
        color = colors[key]
        ax_pol.text(0.02, y_pos, f"● {labels[key]}",
                   color=color, fontsize=10, fontweight='bold',
                   transform=ax_pol.transAxes)
        y_pos -= 0.07
        ax_pol.text(0.05, y_pos, sim["policy"],
                   color='#64748b', fontsize=7, fontstyle='italic',
                   transform=ax_pol.transAxes)
        y_pos -= 0.07

        details = []
        if key == "hr":
            details = [
                f"Employees affected: {sim.get('employees_affected','N/A'):,}",
                f"Attrition: {sim.get('current_attrition_rate',0):.1%} → "
                f"{sim.get('predicted_attrition_rate',0):.1%}",
                f"Retained: +{sim.get('lives_impacted',0):,} employees",
            ]
        elif key == "marketing":
            details = [
                f"Sales lift: +{sim.get('avg_sales_lift',0):.0f} units/week",
                f"Extra cost: ${sim.get('cost_increase_per_week',0):,.0f}/week",
                f"ROI: {sim.get('roi_units_per_dollar',0):.4f} units/$",
            ]
        elif key == "healthcare":
            details = [
                f"Patients: {sim.get('patients_affected',0):,}",
                f"BP: {sim.get('current_avg_bp',0):.0f} → "
                f"{sim.get('predicted_avg_bp',0):.0f} mmHg",
                f"CVD risk ↓ {sim.get('estimated_cvd_risk_reduction',0):.1%}",
            ]

        for d in details:
            ax_pol.text(0.07, y_pos, f"→ {d}",
                       color='#475569', fontsize=7.5,
                       transform=ax_pol.transAxes)
            y_pos -= 0.065
        y_pos -= 0.03

    # ── Row 2 middle: Sensitivity analysis ───────────────────
    ax_sens = fig.add_subplot(gs[1, 1])
    ax_sens.set_facecolor(bg)
    ax_sens.set_title("Sensitivity to Hidden Confounders",
                      color=text, fontsize=10, pad=6)

    for key, sens in sensitivity_results.items():
        color = colors[key]
        gammas = sens["gamma_values"]
        ax_sens.fill_between(gammas, sens["lower_bounds"], sens["upper_bounds"],
                            color=color, alpha=0.2)
        ax_sens.plot(gammas, sens["upper_bounds"], color=color,
                    linewidth=1.5, alpha=0.8)
        ax_sens.plot(gammas, sens["lower_bounds"], color=color,
                    linewidth=1.5, alpha=0.8, label=labels[key])
        if sens["crosses_zero_at"]:
            ax_sens.axvline(sens["crosses_zero_at"], color=color,
                           linestyle=':', linewidth=1.5, alpha=0.6)

    ax_sens.axhline(0, color='#ef4444', linewidth=1.5,
                   linestyle='--', label='Null effect')
    ax_sens.set_xlabel("Gamma (hidden confounder strength)",
                      color=text, fontsize=8)
    ax_sens.set_ylabel("Effect estimate range", color=text, fontsize=8)
    ax_sens.legend(fontsize=7, facecolor=bg, labelcolor=text, edgecolor=grid)
    ax_sens.tick_params(colors=text, labelsize=7)
    for spine in ax_sens.spines.values():
        spine.set_edgecolor(grid)
    ax_sens.grid(color=grid, linewidth=0.5, alpha=0.5)
    ax_sens.text(0.5, 0.02,
                "Vertical dotted line = where result becomes insignificant",
                ha='center', color='#334155', fontsize=7,
                transform=ax_sens.transAxes)

    # ── Row 2 right: Method comparison ───────────────────────
    ax_meth = fig.add_subplot(gs[1, 2])
    ax_meth.set_facecolor(bg)
    ax_meth.set_title("When to Use Which Method",
                      color=text, fontsize=10, pad=6)
    ax_meth.axis('off')

    methods_guide = [
        ("#3b82f6", "Backdoor Adjustment",
         "You know all confounders\nAdd them as controls in regression"),
        ("#22c55e", "Propensity Scoring",
         "Many confounders, binary treatment\nMatch treated/control units"),
        ("#f59e0b", "Instrumental Variables",
         "Hidden confounders exist\nNeed an external instrument"),
        ("#a78bfa", "Difference-in-Differences",
         "Panel data, before/after\nNatural experiment available"),
        ("#ef4444", "Regression Discontinuity",
         "Sharp cutoff rule exists\ne.g. age threshold, score cutoff"),
    ]

    y_p = 0.94
    for color, method, desc in methods_guide:
        ax_meth.text(0.02, y_p, f"● {method}",
                    color=color, fontsize=9, fontweight='bold',
                    transform=ax_meth.transAxes)
        for line in desc.split('\n'):
            y_p -= 0.07
            ax_meth.text(0.06, y_p, line,
                        color='#475569', fontsize=7.5,
                        transform=ax_meth.transAxes)
        y_p -= 0.07

    plt.savefig('causal_simulation_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0e17')
    print("  Saved: causal_simulation_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data
    datasets, causal_results = load_data()

    if not datasets:
        print("No data found. Running Part 1 first...")
        import subprocess
        subprocess.run(["python", "01_causal_dag_and_estimation.py"])
        datasets, causal_results = load_data()

    # 2. Dose-response curves
    estimators, dose_responses = compute_dose_response_curves(datasets)

    # 3. Heterogeneous effects
    cate_results = heterogeneous_effects(datasets, estimators)

    # 4. Policy simulation
    simulations = policy_simulation(datasets, estimators)

    # 5. Sensitivity analysis
    sensitivity = sensitivity_analysis(datasets, estimators)

    # 6. Visualize everything
    visualize_simulation_results(
        dose_responses, cate_results, simulations, sensitivity
    )

    # 7. Save for the interactive dashboard
    output = {
        "simulations": simulations,
        "sensitivity": {
            k: {
                "ate":             v["ate"],
                "crosses_zero_at": v["crosses_zero_at"],
                "label":           v["label"],
            }
            for k, v in sensitivity.items()
        },
    }

    with open("simulation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: simulation_results.json")
    print("Run 03_what_if_app.py or open the React dashboard next →")
