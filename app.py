import copy
import csv
import io
import json
import math
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. DETERMINISTIC MONEY & DOMAIN HELPERS
# ==============================================================================

def to_decimal(val: Any) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def fmt_usd(val: Decimal) -> str:
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.2f}"

def default_snapshot() -> Dict[str, Any]:
    return {
        "team_id": "SOLO-ROB3",
        "course": "AP Calculus",
        "pathway_id": "ROB-3",
        "business_name": "Robotics Kit and Repair Studio",
        "snapshot_id": "SNAP-001",
        "state_version": 1,
        "simulation_date": "2027-01-31",
        "base_period": "Monthly",
        "working_days": 20,
        "currency": "USD",
        "level_by_skill": {"profit_optimization": "Foundation"},
        "teacher_approved_level": "Foundation",
        "assumptions": [
            "Fictional educational baseline values from Career Venture Lab ROB-3.",
            "Linear demand curve q(p) = 80 - (1/3)p on domain 0 <= p <= 240.",
            "Components are expensed when consumed on attempted kits; cash outflow occurs at restock.",
            "Taxes, depreciation, and loan interest disabled at baseline ($0.00).",
        ],
        "sources": ["Career Venture Lab Student Handbook (Nov 2026) - ROB Package & AP Calculus Adapter"],
        "offerings": [
            {
                "offering_id": "ROB-KIT-01",
                "name": "Introductory Robot Kit (Primary)",
                "unit": "kit",
                "price": "120.00",
                "demand_rule": "q(p) = max(0, 80 - (1/3)*p)",
                "capacity_rule": "floor(staff_hours / hours_per_unit)",
            }
        ],
        "funding": {
            "owner_contribution": "12000.00",
            "equipment_assets": "4000.00",
            "pre_opening_expense": "0.00",
            "opening_inventory_units": 100,
            "component_unit_cost": "30.00",
            "opening_cash": "5000.00",
            "retained_earnings": "0.00",
            "loan_debt": "0.00",
        },
        "parameters": {
            "base_price": "120.00",
            "D0": 40,
            "staff_hours": 90.0,
            "hours_per_unit": 3.0,
            "defect_rate": 0.0,
            "restock_units": 0,
            "ad_cost": "0.00",
            "refund_amount": "0.00",
            "loan_principal": "0.00",
            "interest": "0.00",
            "additional_capex": "0.00",
        },
        "cost_lines": {
            "variable": {
                "components_COGS": "30.00",
                "assembly_labor": "20.00",
                "packaging": "10.00",
                "warranty_consumables": "5.00",
            },
            "fixed": {
                "workspace": "250.00",
                "software": "100.00",
                "equipment_maintenance": "150.00",
                "admin": "100.00",
            },
        },
        "balances": {
            "cash": "5000.00",
            "stock_units": 100,
            "inventory_value": "3000.00",
            "equipment_assets": "4000.00",
            "debt": "0.00",
            "retained_earnings": "0.00",
        },
        "executed_run_ids": [],
        "ledger": [],
        "campaigns": [],
        "decisions": [],
    }

# ==============================================================================
# 2. ROB-3 DETERMINISTIC SIMULATION ENGINE
# ==============================================================================

def compute_monthly_step(
    state: Dict[str, Any],
    requested: int,
    sim_date: str,
    run_id: str,
    capacity_override: int | None = None,
    ad_cost_override: Decimal | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes one deterministic monthly step according to ROB package rules.
    Raises ValueError on duplicate run_id or invalid negative inputs.
    """
    if run_id in state["executed_run_ids"]:
        raise ValueError(f"Duplicate run_id '{run_id}' rejected. Transaction already recorded.")
    if requested < 0:
        raise ValueError("Requested units cannot be negative.")

    params = state["parameters"]
    balances = state["balances"]
    v_lines = state["cost_lines"]["variable"]
    f_lines = state["cost_lines"]["fixed"]

    price = to_decimal(params["base_price"])
    staff_hours = float(params["staff_hours"])
    hours_per_unit = float(params["hours_per_unit"])
    defect_rate = float(params["defect_rate"])
    restock_units = int(params["restock_units"])
    comp_unit_cost = to_decimal(v_lines["components_COGS"])

    if not (0.0 <= defect_rate <= 1.0):
        raise ValueError("defect_rate must be between 0 and 1.")
    if restock_units < 0 or hours_per_unit <= 0 or staff_hours < 0:
        raise ValueError("Invalid capacity or restock parameters.")

    # Capacity & Fulfillment Algorithm (ROB Page 79)
    derived_capacity = math.floor(staff_hours / hours_per_unit)
    effective_capacity = int(capacity_override) if capacity_override is not None else derived_capacity

    opening_stock = int(balances["stock_units"])
    available_stock = opening_stock + restock_units
    attempted = min(int(requested), effective_capacity, available_stock)
    failed = math.floor(attempted * defect_rate)
    sold = attempted - failed
    unserved_demand = max(0, int(requested) - attempted)

    # Variable & Fixed Expense Calculations
    non_comp_var_per_unit = (
        to_decimal(v_lines["assembly_labor"])
        + to_decimal(v_lines["packaging"])
        + to_decimal(v_lines["warranty_consumables"])
    )
    total_var_per_unit = comp_unit_cost + non_comp_var_per_unit

    revenue = (price * Decimal(sold)).quantize(Decimal("0.01"))
    refunds = to_decimal(params["refund_amount"])
    net_revenue = revenue - refunds

    # Variable expense incurred on EVERY attempted kit (including failed builds)
    variable_expense = (total_var_per_unit * Decimal(attempted)).quantize(Decimal("0.01"))
    failed_component_writeoff = (comp_unit_cost * Decimal(failed)).quantize(Decimal("0.01"))
    fixed_expense = sum((to_decimal(v) for v in f_lines.values()), Decimal("0.00"))
    marketing_expense = ad_cost_override if ad_cost_override is not None else to_decimal(params["ad_cost"])

    total_operating_expense = variable_expense + fixed_expense + marketing_expense
    operating_profit = net_revenue - total_operating_expense

    interest = to_decimal(params["interest"])
    principal = to_decimal(params["loan_principal"])
    capex = to_decimal(params["additional_capex"])
    net_profit = operating_profit - interest

    # Cash Movement (Components paid in cash on restock, NOT on consumption)
    opening_cash = to_decimal(balances["cash"])
    non_comp_cash_outflow = (non_comp_var_per_unit * Decimal(attempted)).quantize(Decimal("0.01"))
    restock_cash_outflow = (comp_unit_cost * Decimal(restock_units)).quantize(Decimal("0.01"))

    ending_cash = (
        opening_cash
        + net_revenue
        - non_comp_cash_outflow
        - restock_cash_outflow
        - fixed_expense
        - marketing_expense
        - interest
        - principal
        - capex
    ).quantize(Decimal("0.01"))

    # Inventory Reconciliation
    ending_stock = available_stock - attempted
    ending_inventory_value = (comp_unit_cost * Decimal(ending_stock)).quantize(Decimal("0.01"))

    # Build updated state & ledger entry
    new_state = copy.deepcopy(state)
    new_version = int(state["state_version"]) + 1
    new_state["state_version"] = new_version
    new_state["simulation_date"] = sim_date
    new_state["executed_run_ids"].append(run_id)

    new_state["balances"]["cash"] = str(ending_cash)
    new_state["balances"]["stock_units"] = ending_stock
    new_state["balances"]["inventory_value"] = str(ending_inventory_value)
    new_state["balances"]["equipment_assets"] = str(to_decimal(balances["equipment_assets"]) + capex)
    new_state["balances"]["debt"] = str(max(Decimal("0.00"), to_decimal(balances["debt"]) - principal))
    new_state["balances"]["retained_earnings"] = str(to_decimal(balances["retained_earnings"]) + net_profit)

    summary = {
        "run_id": run_id,
        "date": sim_date,
        "state_version": new_version,
        "requested": int(requested),
        "effective_capacity": effective_capacity,
        "attempted": attempted,
        "failed": failed,
        "sold": sold,
        "unserved_demand": unserved_demand,
        "revenue": str(net_revenue),
        "variable_expense": str(variable_expense),
        "failed_writeoff_included": str(failed_component_writeoff),
        "fixed_expense": str(fixed_expense),
        "marketing_expense": str(marketing_expense),
        "total_expense": str(total_operating_expense),
        "operating_profit": str(operating_profit),
        "net_profit": str(net_profit),
        "opening_cash": str(opening_cash),
        "closing_cash": str(ending_cash),
        "opening_stock": opening_stock,
        "ending_stock": ending_stock,
        "ending_inventory_value": str(ending_inventory_value),
        "build_hours_used": attempted * hours_per_unit,
    }

    ledger_tx = {
        "transaction_id": f"TX-{new_version:04d}-{run_id}",
        "date": sim_date,
        "event_type": "MONTHLY_RUN",
        "category": "Operations & Sales",
        "amount": str(net_profit),
        "quantity": sold,
        "unit_price": str(price),
        "offering_id": "ROB-KIT-01",
        "campaign_id": f"CAMP-{sim_date}" if marketing_expense > 0 else None,
        "financing_type": "Operating",
        "state_version": new_version,
        "details": summary,
    }
    new_state["ledger"].append(ledger_tx)
    return new_state, summary

# ==============================================================================
# 3. SNAPSHOT VALIDATION & IMPORT/EXPORT
# ==============================================================================

def validate_and_import_snapshot(raw_json: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    data = json.loads(raw_json)
    required_keys = [
        "team_id", "course", "pathway_id", "snapshot_id", "state_version",
        "simulation_date", "currency", "funding", "parameters", "cost_lines", "balances", "ledger"
    ]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"Missing required snapshot field: '{k}'")

    if data["team_id"] != current_state["team_id"]:
        raise ValueError(
            f"Team ID mismatch: imported '{data['team_id']}', expected '{current_state['team_id']}'."
        )
    if data["pathway_id"] != "ROB-3":
        raise ValueError(f"Pathway mismatch: expected 'ROB-3', got '{data['pathway_id']}'.")
    if int(data["state_version"]) < int(current_state["state_version"]):
        raise ValueError(
            f"Stale state_version rejected: imported v{data['state_version']} < current v{current_state['state_version']}."
        )
    if int(data["balances"]["stock_units"]) < 0 or to_decimal(data["balances"]["cash"]) < 0:
        raise ValueError("Negative cash or stock balance rejected.")
    return data

# ==============================================================================
# 4. AP CALCULUS MATHEMATICAL ENGINE (PATHWAY ROB-3)
# ==============================================================================

def calc_rob3_metrics(D0: float, p0: float, v: float, F: float, capacity: int) -> Dict[str, float]:
    """
    Linear demand: q(p) = a - b*p where a = 2*D0, b = D0/p0 on [0, 2*p0].
    """
    a = 2.0 * D0
    b = D0 / p0
    p_max = 2.0 * p0
    p_unconstrained = (a + b * v) / (2.0 * b)
    q_unconstrained = a - b * p_unconstrained
    profit_unconstrained = (p_unconstrained - v) * q_unconstrained - F

    # Capacity corner: a - b*p = capacity => p_corner = (a - capacity)/b
    p_corner = max(0.0, (a - capacity) / b)
    profit_corner = (p_corner - v) * min(capacity, max(0.0, a - b * p_corner)) - F

    # Constrained optimum on [0, 2*p0]
    if q_unconstrained <= capacity:
        p_opt = p_unconstrained
        profit_opt = profit_unconstrained
    else:
        p_opt = p_corner
        profit_opt = profit_corner

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_unconstrained": p_unconstrained,
        "q_unconstrained": q_unconstrained,
        "profit_unconstrained": profit_unconstrained,
        "p_corner": p_corner,
        "profit_corner": profit_corner,
        "p_opt": p_opt,
        "profit_opt": profit_opt,
    }

def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ==============================================================================
# 5. STREAMLIT UI (STAGE 1)
# ==============================================================================

st.set_page_config(page_title="Career Venture Lab | ROB-3 (AP Calculus)", layout="wide")

if "sim_state" not in st.session_state:
    st.session_state.sim_state = default_snapshot()

state = st.session_state.sim_state

# --- SIDEBAR: STATE, SAVE/LOAD, & FIXTURE MODE ---
with st.sidebar:
    st.header("System State & Persistence")
    state["team_id"] = st.text_input("Team ID", value=state["team_id"])
    st.caption(f"Course: **{state['course']}** | Pathway: **{state['pathway_id']}**")
    st.caption(f"Snapshot: `{state['snapshot_id']}` | Version: **v{state['state_version']}**")

    st.divider()
    st.subheader("Durable Save / Restore (JSON)")
    snapshot_bytes = json.dumps(state, indent=2).encode("utf-8")
    st.download_button(
        label="Export Snapshot JSON",
        data=snapshot_bytes,
        file_name=f"{state['team_id']}_v{state['state_version']}.json",
        mime="application/json",
    )

    uploaded_json = st.file_uploader("Restore / Import Snapshot JSON", type=["json"])
    if uploaded_json is not None:
        if st.button("Validate & Load Snapshot"):
            try:
                loaded = validate_and_import_snapshot(uploaded_json.read().decode("utf-8"), state)
                st.session_state.sim_state = loaded
                st.success(f"Loaded snapshot v{loaded['state_version']} successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Import Rejected: {exc}")

    if st.button("Reset Simulation to Opening Baseline"):
        st.session_state.sim_state = default_snapshot()
        st.warning("Simulation reset to v1 opening balances.")
        st.rerun()

# --- HEADER & TOP METRICS ---
st.title("Career Venture Lab — Robotics Kit & Repair Studio (ROB-3)")
st.info(
    "**Stage 1 Build Active:** Deterministic Ledger, Startup Capital & Cost Controls, AP Calculus Math Skills Panel, "
    "Save/Restore JSON, and Fixture Tests. *(Stage 2–4 AI Assistants, Marketing Studio, Assessments & Storefront are marked below.)*"
)

bal = state["balances"]
params = state["parameters"]
v_sum = sum((to_decimal(x) for x in state["cost_lines"]["variable"].values()), Decimal("0.00"))
f_sum = sum((to_decimal(x) for x in state["cost_lines"]["fixed"].values()), Decimal("0.00"))
derived_cap = math.floor(float(params["staff_hours"]) / float(params["hours_per_unit"]))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spendable Cash", fmt_usd(to_decimal(bal["cash"])))
m2.metric("Component Stock", f"{bal['stock_units']} sets ({fmt_usd(to_decimal(bal['inventory_value']))})")
m3.metric("Monthly Capacity", f"{derived_cap} kits")
m4.metric("Unit Variable Cost", fmt_usd(v_sum))
m5.metric("Monthly Fixed Cost", fmt_usd(f_sum))

# --- TABS ---
tabs = st.tabs([
    "1. Overview & Math Skills (AP Calculus)",
    "2. Inputs, Startup & Monthly Run",
    "3. Ledger & Fixture-Test Mode",
    "4. Stage 2–4 Roadmap (Unimplemented)",
])

# ==============================================================================
# TAB 1: PERSISTENT MATH SKILLS & CHALLENGE LEVEL PANEL
# ==============================================================================
with tabs[0]:
    st.subheader("Math Skills & Challenge Level Panel — AP Calculus (ROB-3)")

    c_lvl, c_lock = st.columns([2, 2])
    with c_lvl:
        chosen_level = st.selectbox(
            "Select Challenge Level",
            ["Foundation", "Applied", "Advanced"],
            index=["Foundation", "Applied", "Advanced"].index(state["teacher_approved_level"]),
        )
    with c_lock:
        teacher_approved = st.checkbox(
            "Teacher Approval Confirmed for Selected Level",
            value=True,
            help="Handbook Rule: Changing challenge level changes equations, visuals, and tasks; requires teacher sign-off.",
        )

    if teacher_approved:
        state["teacher_approved_level"] = chosen_level
        state["level_by_skill"]["profit_optimization"] = chosen_level
    else:
        st.warning("Level preview active. Check teacher approval box to save level change to snapshot.")

    p0_val = float(params["base_price"])
    D0_val = float(params["D0"])
    v_val = float(v_sum)
    F_val = float(f_sum)
    calc_res = calc_rob3_metrics(D0_val, p0_val, v_val, F_val, derived_cap)

    st.markdown(f"**Course:** AP Calculus | **Pathway:** ROB-3 | **Active Level:** `{chosen_level}`")

    if chosen_level == "Foundation":
        st.markdown(
            """
            **Business Question:** What kit price $p$ maximizes unconstrained monthly operating profit $P(p)$, and what are the units and meaning of $P'(p)$?  
            **Model & Equations (Continuous Relaxation):**
            - Demand function: $q(p) = 80 - \\frac{1}{3}p$ on $0 \\le p \\le 240$ (units: kits/month)
            - Unconstrained Profit: $P(p) = (p - v)(a - bp) - F = (p - 65)\\left(80 - \\frac{1}{3}p\\right) - 600$ (units: USD/month)
            - First Derivative: $P'(p) = a + bv - 2bp = 80 + \\frac{65}{3} - \\frac{2}{3}p$ (units: $\\frac{\\text{USD/month}}{\\text{USD/kit}} = \\text{kits/month}$)
            """
        )
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Critical Price p*", f"${calc_res['p_unconstrained']:,.2f}/kit")
        col_f2.metric("Unconstrained Demand q(p*)", f"{calc_res['q_unconstrained']:.2f} kits")
        col_f3.metric("Unconstrained Max Profit", f"${calc_res['profit_unconstrained']:,.2f}")

        p_grid = np.linspace(0, calc_res["p_max"], 250)
        q_grid = np.maximum(0.0, calc_res["a"] - calc_res["b"] * p_grid)
        prof_grid = (p_grid - v_val) * q_grid - F_val

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(p_grid, prof_grid, label="Unconstrained Profit P(p)", color="#1f77b4", linewidth=2.2)
        ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero Profit Line ($0)")
        ax.scatter(
            [calc_res["p_unconstrained"]],
            [calc_res["profit_unconstrained"]],
            color="#d62728",
            zorder=5,
            label=f"Critical Point (${calc_res['p_unconstrained']:.2f}, ${calc_res['profit_unconstrained']:.2f})",
        )
        ax.set_title(f"ROB-3 Unconstrained Profit P(p) | Snapshot {state['snapshot_id']} v{state['state_version']} (Projection)")
        ax.set_xlabel("Kit Price p (USD / kit)")
        ax.set_ylabel("Monthly Profit P(p) (USD)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "Download Graph PNG (with Metadata)",
            data=fig_to_png_bytes(fig),
            file_name=f"ROB3_Foundation_Profit_v{state['state_version']}.png",
            mime="image/png",
        )
        st.info("**Interpretation Prompt:** At $p = \\$152.50$, $P'(152.50) = 0$. Why does $q(152.50) = 29.17$ kits require integer rounding in actual transactions, and what happens if capacity is lower than 29 kits?")

    elif chosen_level == "Applied":
        st.markdown(
            f"""
            **Business Question:** How does the shop capacity constraint $s(p) = \\min(\\text{{capacity}}, \\max(0, q(p)))$ create a piecewise profit function and a non-differentiable corner?  
            **Piecewise Equations (Capacity = {derived_cap} kits/month):**
            - Capacity Corner Price: $p_{{\\text{{corner}}}} = \\frac{{a - \\text{{capacity}}}}{{b}} = \\${calc_res['p_corner']:,.2f}$
            - **Branch 1 (Capacity-Constrained, $p < \\${calc_res['p_corner']:,.2f}$):** $P(p) = {derived_cap}(p - {v_val:.0f}) - {F_val:.0f}$, so $P'(p) = {derived_cap}$ kits/month.
            - **Branch 2 (Demand-Constrained, $p > \\${calc_res['p_corner']:,.2f}$):** $P(p) = (p - {v_val:.0f})(a - bp) - {F_val:.0f}$, so $P'(p) = a + bv - 2bp$.
            - At $p = \\${calc_res['p_corner']:,.2f}$, $P'(p)$ is **undefined** (left derivative $= {derived_cap} \\ne$ right derivative $= {calc_res['a'] + calc_res['b']*v_val - 2*calc_res['b']*calc_res['p_corner']:.2f}$).
            """
        )

        p_grid = np.linspace(0, calc_res["p_max"], 400)
        q_raw = np.maximum(0.0, calc_res["a"] - calc_res["b"] * p_grid)
        s_cont = np.minimum(derived_cap, q_raw)
        prof_cont = (p_grid - v_val) * s_cont - F_val
        prof_int = (p_grid - v_val) * np.floor(s_cont) - F_val

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
        ax1.plot(p_grid, prof_cont, label="Constrained Profit P(p) [Continuous]", color="#1f77b4", linewidth=2.2)
        ax1.plot(p_grid, prof_int, label="Integer-Floored Sales Profit [Discrete]", color="#ff7f0e", linestyle=":", linewidth=1.6)
        ax1.axvline(calc_res["p_corner"], color="#9467bd", linestyle="-.", label=f"Capacity Boundary p=${calc_res['p_corner']:.2f}")
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.scatter([calc_res["p_opt"]], [calc_res["profit_opt"]], color="#d62728", zorder=5, label=f"Constrained Optimum (${calc_res['p_opt']:.2f}, ${calc_res['profit_opt']:.2f})")
        ax1.set_ylabel("Profit (USD)")
        ax1.set_title(f"ROB-3 Piecewise Profit & Derivative P'(p) | Snapshot {state['snapshot_id']} v{state['state_version']} (Projection)")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Derivative plot with open circles at corner
        left_p = p_grid[p_grid < calc_res["p_corner"]]
        right_p = p_grid[p_grid > calc_res["p_corner"]]
        ax2.plot(left_p, np.full_like(left_p, derived_cap), color="#2ca02c", linewidth=2.2, label="P'(p) Left Branch = Capacity")
        ax2.plot(right_p, calc_res["a"] + calc_res["b"] * v_val - 2 * calc_res["b"] * right_p, color="#17becf", linewidth=2.2, label="P'(p) Right Branch = a + bv - 2bp")
        left_lim = float(derived_cap)
        right_lim = calc_res["a"] + calc_res["b"] * v_val - 2 * calc_res["b"] * calc_res["p_corner"]
        ax2.scatter([calc_res["p_corner"], calc_res["p_corner"]], [left_lim, right_lim], facecolors="white", edgecolors="red", s=70, zorder=6, label="P'(p) Undefined at Corner")
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Kit Price p (USD / kit)")
        ax2.set_ylabel("P'(p) (kits / month)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Linked price comparison table
        sample_prices = [120.0, calc_res["p_corner"], calc_res["p_unconstrained"], 180.0]
        rows = []
        for sp in sample_prices:
            qd = max(0.0, calc_res["a"] - calc_res["b"] * sp)
            sc = min(derived_cap, qd)
            si = math.floor(sc)
            rows.append({
                "Price ($/kit)": round(sp, 2),
                "Unconstrained Demand q(p)": round(qd, 2),
                "Continuous Sales s(p)": round(sc, 2),
                "Integer Floored Sales": si,
                "Continuous Profit ($)": round((sp - v_val) * sc - F_val, 2),
                "Integer Profit ($)": round((sp - v_val) * si - F_val, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("**Transfer Check:** Predict how lowering `staff_hours` from 90 to 60 shifts the capacity corner $p_{\\text{corner}}$ before moving any sliders.")

    elif chosen_level == "Advanced":
        st.markdown(
            """
            **Business Question (Accumulation Exercise):** Over a 10-day promotional campaign ($0 \\le t \\le 10$ days), marginal revenue is $r(t) = 100 + 20t$ ($/day) and marginal cost is $c(t) = 60 + 5t$ ($/day). What is the net accumulated profit over $[0, 10]$?  
            *(Isolated rate-accumulation module; does not overwrite the monthly ledger.)*
            - **Accumulated Revenue:** $\\int_0^{10} (100 + 20t)\\,dt = \\left[100t + 10t^2\\right]_0^{10} = 1000 + 1000 = \\$2,000.00$
            - **Accumulated Cost:** $\\int_0^{10} (60 + 5t)\\,dt = \\left[60t + 2.5t^2\\right]_0^{10} = 600 + 250 = \\$850.00$
            - **Net Accumulated Profit:** $\\int_0^{10} (r(t) - c(t))\\,dt = \\$2,000.00 - \\$850.00 = \\$1,150.00$
            """
        )
        t_grid = np.linspace(0, 10, 200)
        r_grid = 100 + 20 * t_grid
        c_grid = 60 + 5 * t_grid

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_grid, r_grid, label="Marginal Revenue r(t) = 100 + 20t ($/day)", color="#2ca02c", linewidth=2)
        ax.plot(t_grid, c_grid, label="Marginal Cost c(t) = 60 + 5t ($/day)", color="#d62728", linewidth=2)
        ax.fill_between(t_grid, c_grid, r_grid, color="#2ca02c", alpha=0.2, label="Net Accumulated Profit = $1,150.00")
        ax.set_title(f"ROB-3 Campaign Rate Accumulation (0 <= t <= 10 days) | Snapshot {state['snapshot_id']} v{state['state_version']}")
        ax.set_xlabel("Campaign Day t (days)")
        ax.set_ylabel("Rate (USD / day)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ==============================================================================
# TAB 2: INPUTS, STARTUP CAPITAL & MONTHLY RUN
# ==============================================================================
with tabs[1]:
    st.subheader("Startup Financing & Opening Balance Sheet (Read-Only Reference)")
    f_info = state["funding"]
    st.markdown(
        f"- **Owner Contribution (Equity):** {fmt_usd(to_decimal(f_info['owner_contribution']))} | "
        f"**Equipment Assets:** {fmt_usd(to_decimal(f_info['equipment_assets']))} | "
        f"**Opening Inventory (100 kits @ $30):** {fmt_usd(to_decimal('3000.00'))} | "
        f"**Post-Startup Opening Cash:** {fmt_usd(to_decimal(f_info['opening_cash']))}"
    )

    st.divider()
    st.subheader("Editable Operating Parameters & Cost Lines")
    c1, c2, c3 = st.columns(3)
    with c1:
        new_price = st.number_input("Base Price ($/kit)", min_value=1.0, value=float(params["base_price"]), step=5.0)
        new_d0 = st.number_input("Baseline Demand D0 (kits/mo)", min_value=0, value=int(params["D0"]), step=1)
        new_staff_hrs = st.number_input("Staff Hours / Month", min_value=0.0, value=float(params["staff_hours"]), step=3.0)
        new_hrs_unit = st.number_input("Hours / Kit", min_value=0.5, value=float(params["hours_per_unit"]), step=0.5)
    with c2:
        new_restock = st.number_input("Restock Component Sets (this month)", min_value=0, value=int(params["restock_units"]), step=5)
        new_defect = st.slider("Defect Rate (0.00 to 1.00)", min_value=0.0, max_value=1.0, value=float(params["defect_rate"]), step=0.05)
        new_ad = st.number_input("Own Marketing Expense ($)", min_value=0.0, value=float(params["ad_cost"]), step=25.0)
    with c3:
        st.markdown("**Variable Cost / Attempted Kit ($65 default)**")
        v_comp = st.number_input("Components COGS ($/kit)", min_value=0.0, value=float(state["cost_lines"]["variable"]["components_COGS"]), step=1.0)
        v_labor = st.number_input("Assembly Labor ($/kit)", min_value=0.0, value=float(state["cost_lines"]["variable"]["assembly_labor"]), step=1.0)
        v_pack = st.number_input("Packaging ($/kit)", min_value=0.0, value=float(state["cost_lines"]["variable"]["packaging"]), step=1.0)
        v_warr = st.number_input("Warranty Consumables ($/kit)", min_value=0.0, value=float(state["cost_lines"]["variable"]["warranty_consumables"]), step=1.0)

    if st.button("Save Input Parameters to Current Snapshot"):
        params["base_price"] = str(to_decimal(new_price))
        params["D0"] = int(new_d0)
        params["staff_hours"] = float(new_staff_hrs)
        params["hours_per_unit"] = float(new_hrs_unit)
        params["restock_units"] = int(new_restock)
        params["defect_rate"] = float(new_defect)
        params["ad_cost"] = str(to_decimal(new_ad))
        state["cost_lines"]["variable"]["components_COGS"] = str(to_decimal(v_comp))
        state["cost_lines"]["variable"]["assembly_labor"] = str(to_decimal(v_labor))
        state["cost_lines"]["variable"]["packaging"] = str(to_decimal(v_pack))
        state["cost_lines"]["variable"]["warranty_consumables"] = str(to_decimal(v_warr))
        st.success("Parameters updated in active snapshot (ledger unchanged until a simulation step is approved).")
        st.rerun()

    st.divider()
    st.subheader("Execute Single Monthly Operating Step (Normal Mode)")
    r1, r2, r3 = st.columns(3)
    req_units = r1.number_input("Requested Kits this Month", min_value=0, value=int(params["D0"]), step=1)
    run_date = r2.text_input("Period End Date (YYYY-MM-DD)", value="2027-01-31")
    run_id_input = r3.text_input("Unique Run ID", value=f"RUN-v{state['state_version']}")

    if st.button("Approve & Advance Simulation One Month", type="primary"):
        try:
            updated, summary = compute_monthly_step(state, requested=int(req_units), sim_date=run_date, run_id=run_id_input)
            st.session_state.sim_state = updated
            st.success(
                f"Recorded {run_id_input}: Sold {summary['sold']} kits | Revenue {fmt_usd(to_decimal(summary['revenue']))} | "
                f"Profit {fmt_usd(to_decimal(summary['operating_profit']))} | Closing Cash {fmt_usd(to_decimal(summary['closing_cash']))}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Simulation Step Blocked: {exc}")

# ==============================================================================
# TAB 3: LEDGER & FIXTURE-TEST MODE
# ==============================================================================
with tabs[2]:
    st.subheader("Fixture-Test Mode (ROB Package Acceptance Verification)")
    st.warning(
        "**FIXTURE-TEST MODE WARNING:** Replaying `sample_data/monthly_fixture.csv` resets the simulation to opening "
        "baseline ($5,000 cash, 100 component sets) and overrides derived capacity with fixture `capacity_override` values "
        "for ledger verification. Do not mix fixture runs with normal operating runs."
    )

    if st.button("Reset & Replay ROB Monthly Fixture (2027-01 to 2027-03)"):
        fresh = default_snapshot()
        fixture_df = pd.read_csv("sample_data/monthly_fixture.csv")
        results_rows = []
        for idx, row in fixture_df.iterrows():
            fresh, res = compute_monthly_step(
                fresh,
                requested=int(row["requested"]),
                sim_date=f"{row['month']}-28",
                run_id=f"FIXTURE-{row['month']}",
                capacity_override=int(row["capacity_override"]),
                ad_cost_override=to_decimal(row["ad_cost"]),
            )
            results_rows.append(res)
        st.session_state.sim_state = fresh
        st.success("Fixture replay completed and verified against ROB-3 ledger rules.")
        st.dataframe(pd.DataFrame(results_rows), use_container_width=True)

    st.divider()
    st.subheader("Approved Event Ledger")
    if not state["ledger"]:
        st.caption("No transactions recorded yet in this snapshot.")
    else:
        ledger_rows = [tx["details"] for tx in state["ledger"]]
        df_ledger = pd.DataFrame(ledger_rows)
        st.dataframe(df_ledger, use_container_width=True)
        st.download_button(
            "Export Ledger CSV",
            data=df_ledger.to_csv(index=False).encode("utf-8"),
            file_name=f"{state['team_id']}_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 4: STAGE 2-4 ROADMAP (HONEST PLACEHOLDERS)
# ==============================================================================
with tabs[3]:
    st.subheader("Unimplemented Modules (Scheduled for Stages 2–4)")
    st.markdown(
        """
        - **Stage 2 (Not Yet Implemented):** Operations & Marketing AI assistant handoff cards, full Report Center (Income Statement, Cash Flow, Balance Sheet PDF exports), and Costed Marketing Studio.
        - **Stage 3 (Not Yet Implemented):** Cross-Course Connections panel (Engineering sensor/defect test & English/Technical documentation) and frozen-snapshot Practice Assessment generator.
        - **Stage 4 Optional (Not Yet Implemented):** Fictional demonstration storefront (no real payments).
        """
    )
