
from pathlib import Path
import numpy as np
import pandas as pd
import random

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)

N_SUPPLIERS = 100
MONTHS = pd.date_range("2025-01-01", "2026-06-01", freq="MS")

CATEGORIES = ["IT Services","Logistics","Facilities","Professional Services","Hardware","Telecom","Office Supplies","Staffing","Cloud/SaaS","Marketing"]
REGIONS = ["North","South","East","West","Central"]
CRITICALITIES = ["Low","Medium","High"]
ARCHETYPES = ["Reliable","Deteriorating","Improving","Low-Cost Risky","High-Cost High-Quality"]
ARCHETYPE_PROBS = [0.30,0.22,0.16,0.17,0.15]
ISSUE_DRIVERS = ["warehouse staffing shortage","material availability constraints","quality-control process gaps","logistics network disruption","invoice processing errors","capacity constraints","subcontractor performance issues","system integration delays","high employee turnover","forecasting and planning gaps"]

PREFIXES = ["Apex","Nova","Vertex","BluePeak","Orion","Nexus","Summit","Prime","Atlas","Brightline","Quantum","Evergreen","Crest","SilverOak","Redwood","Helix","Pioneer","Nimbus","Horizon","Sterling"]
SUFFIXES = ["Solutions","Technologies","Services","Industries","Systems","Enterprises","Partners","Group","Global","Works"]

def archetype_params(a):
    return {
        "Reliable": dict(sla=97, delivery=96, defect=1.8, invoice=98, resolution=2.2, trend=0.0),
        "Deteriorating": dict(sla=96, delivery=95, defect=2.2, invoice=97, resolution=2.5, trend=-0.55),
        "Improving": dict(sla=89, delivery=88, defect=5.5, invoice=92, resolution=5.2, trend=0.45),
        "Low-Cost Risky": dict(sla=90, delivery=89, defect=5.8, invoice=92, resolution=5.8, trend=-0.08),
        "High-Cost High-Quality": dict(sla=98, delivery=97.5, defect=1.2, invoice=99, resolution=1.7, trend=0.0),
    }[a]

def generate_suppliers():
    rows, used = [], set()
    for i in range(1, N_SUPPLIERS + 1):
        while True:
            name = f"{random.choice(PREFIXES)} {random.choice(SUFFIXES)}"
            if name not in used:
                used.add(name)
                break
        archetype = rng.choice(ARCHETYPES, p=ARCHETYPE_PROBS)
        rows.append({
            "supplier_id": f"SUP{i:03d}",
            "supplier_name": name,
            "category": random.choice(CATEGORIES),
            "region": random.choice(REGIONS),
            "criticality": random.choices(CRITICALITIES, weights=[0.25,0.50,0.25])[0],
            "preferred_supplier": "Yes" if (archetype in ["Reliable","High-Cost High-Quality"] and rng.random() < 0.75) else "No",
            "onboarding_date": (pd.Timestamp("2019-01-01") + pd.to_timedelta(int(rng.integers(0,2000)), unit="D")).date().isoformat(),
            "archetype": archetype,
            "primary_issue_driver": random.choice(ISSUE_DRIVERS),
        })
    return pd.DataFrame(rows)

def generate_performance(suppliers):
    rows = []
    for _, s in suppliers.iterrows():
        p = archetype_params(s.archetype)
        supplier_noise = rng.normal(0, 0.8)
        for t, m in enumerate(MONTHS):
            trend = p["trend"] * t
            seasonal = 0.8 * np.sin((t % 12) / 12 * 2 * np.pi)
            sla = np.clip(p["sla"] + trend + seasonal + supplier_noise + rng.normal(0,1.15), 70,100)
            delivery = np.clip(p["delivery"] + 0.9*trend + seasonal + supplier_noise + rng.normal(0,1.3), 65,100)
            defect = np.clip(p["defect"] - 0.22*trend + rng.normal(0,0.55), 0.1,15)
            invoice = np.clip(p["invoice"] + 0.35*trend + rng.normal(0,0.8), 75,100)
            resolution = np.clip(p["resolution"] - 0.12*trend + rng.normal(0,0.7), 0.5,20)
            risk = max(0,92-sla)/6 + max(0,92-delivery)/6 + max(0,defect-4)/2 + max(0,94-invoice)/6
            escalations = int(np.clip(rng.poisson(max(0.15,0.35+risk)),0,12))
            rows.append({
                "supplier_id": s.supplier_id,
                "month": m.date().isoformat(),
                "sla_compliance": round(float(sla),2),
                "on_time_delivery_rate": round(float(delivery),2),
                "defect_rate": round(float(defect),2),
                "invoice_accuracy": round(float(invoice),2),
                "avg_resolution_days": round(float(resolution),2),
                "escalation_count": escalations,
                "order_fulfillment_rate": round(float(np.clip(delivery-rng.normal(0.5,0.8),60,100)),2),
            })
    return pd.DataFrame(rows)

def generate_invoices(suppliers, performance):
    base = {"IT Services":180000,"Logistics":120000,"Facilities":90000,"Professional Services":220000,"Hardware":160000,"Telecom":140000,"Office Supplies":45000,"Staffing":130000,"Cloud/SaaS":200000,"Marketing":150000}
    rows, inv_id = [], 1
    for _, s in suppliers.iterrows():
        amount_base = base[s.category] * (1.35 if s.archetype=="High-Cost High-Quality" else 0.80 if s.archetype=="Low-Cost Risky" else 1)
        sp = performance[performance.supplier_id==s.supplier_id].set_index("month")
        for m in MONTHS:
            p = sp.loc[m.date().isoformat()]
            for _ in range(int(np.clip(rng.poisson(10),3,20))):
                amount = max(5000, rng.lognormal(np.log(amount_base),0.45))
                error = rng.random() < np.clip((100-p.invoice_accuracy)/100,0.005,0.25)
                rows.append({
                    "invoice_id": f"INV{inv_id:06d}",
                    "supplier_id": s.supplier_id,
                    "invoice_date": (m + pd.to_timedelta(int(rng.integers(0,27)),unit="D")).date().isoformat(),
                    "invoice_amount": round(float(amount),2),
                    "approved_amount": round(float(amount*(1-rng.uniform(0.005,0.03)) if error else amount),2),
                    "payment_delay_days": max(0,int(rng.normal(3+(100-p.invoice_accuracy)*0.25,4))),
                    "invoice_error_flag": int(error),
                })
                inv_id += 1
    return pd.DataFrame(rows)

def generate_incidents(suppliers, performance):
    types = ["Delivery Delay","Quality Defect","Invoice Discrepancy","SLA Breach","Capacity Issue","System/Integration Issue","Communication Breakdown"]
    resolutions = [
        "Supplier committed to weekly status updates and a corrective action plan.",
        "An improvement plan was agreed with fortnightly performance checkpoints.",
        "Additional staffing and management oversight were introduced.",
        "The issue was escalated to supplier leadership with a 30-day remediation target.",
        "Process controls were strengthened and performance will be monitored in the next review cycle.",
    ]
    rows, inc_id = [], 1
    for _, s in suppliers.iterrows():
        sp = performance[performance.supplier_id==s.supplier_id]
        for _, p in sp.iterrows():
            risk = max(0,93-p.sla_compliance)/7 + max(0,93-p.on_time_delivery_rate)/7 + max(0,p.defect_rate-3.5)/2 + p.escalation_count*0.22
            for _ in range(int(np.clip(rng.poisson(0.2+0.55*risk),0,5))):
                typ = random.choice(types)
                driver = s.primary_issue_driver
                description = f"{typ} was recorded during the period, with {driver} identified as a likely contributing factor."
                rows.append({
                    "incident_id": f"INC{inc_id:05d}",
                    "supplier_id": s.supplier_id,
                    "incident_date": (pd.Timestamp(p['month']) + pd.to_timedelta(int(rng.integers(0,27)),unit="D")).date().isoformat(),
                    "severity": random.choices(["Low","Medium","High","Critical"],weights=[0.35,0.40,0.20,0.05])[0],
                    "incident_type": typ,
                    "description": description,
                    "resolution": random.choice(resolutions),
                })
                inc_id += 1
    return pd.DataFrame(rows)

def generate_reviews(suppliers, performance):
    rows, rev_id = [], 1
    for _, s in suppliers.iterrows():
        sp = performance[performance.supplier_id==s.supplier_id].copy()
        sp["month"] = pd.to_datetime(sp["month"])
        selected = MONTHS[rng.random(len(MONTHS)) < 0.68]
        if len(selected) < 8:
            selected = MONTHS[::2]
        for m in selected:
            curr = sp[sp.month==m].iloc[0]
            hist = sp[sp.month<=m].tail(3)
            sla_delta = hist.sla_compliance.iloc[-1]-hist.sla_compliance.iloc[0] if len(hist)>1 else 0
            delivery_delta = hist.on_time_delivery_rate.iloc[-1]-hist.on_time_delivery_rate.iloc[0] if len(hist)>1 else 0
            defect_delta = hist.defect_rate.iloc[-1]-hist.defect_rate.iloc[0] if len(hist)>1 else 0
            direction = "deteriorated" if (sla_delta<-2 or delivery_delta<-2 or defect_delta>1) else "improved" if (sla_delta>2 or delivery_delta>2 or defect_delta<-1) else "remained broadly stable"
            summary = f"Supplier performance {direction} during the recent review period. SLA compliance is {curr.sla_compliance:.1f}%, on-time delivery is {curr.on_time_delivery_rate:.1f}%, and defect rate is {curr.defect_rate:.1f}%."
            issues = []
            if curr.sla_compliance < 92: issues.append("SLA compliance is below the preferred operating threshold")
            if curr.on_time_delivery_rate < 92: issues.append("on-time delivery performance is below expectations")
            if curr.defect_rate > 4: issues.append("quality defect levels are elevated")
            if curr.invoice_accuracy < 94: issues.append("invoice accuracy requires attention")
            if curr.escalation_count >= 3: issues.append("escalation volume has increased")
            key_issues = ("; ".join(issues) + f". Review discussion highlighted {s.primary_issue_driver} as a likely contributing factor.") if issues else f"No major performance breach was identified. The team continues to monitor {s.primary_issue_driver}."
            rows.append({
                "review_id": f"REV{rev_id:05d}",
                "supplier_id": s.supplier_id,
                "review_date": (m + pd.offsets.MonthEnd(0)).date().isoformat(),
                "review_type": "Monthly Performance Review",
                "performance_summary": summary,
                "key_issues": key_issues,
                "corrective_actions": "Supplier to submit a corrective action plan and provide regular progress updates." if issues else "Maintain current controls and continue monthly KPI monitoring.",
                "reviewer_notes": "Priority should remain on trend monitoring and early intervention." if direction!="improved" else "Recent improvements should be validated over the next two review cycles before lowering oversight.",
            })
            rev_id += 1
    return pd.DataFrame(rows)

def main():
    suppliers = generate_suppliers()
    performance = generate_performance(suppliers)
    invoices = generate_invoices(suppliers, performance)
    incidents = generate_incidents(suppliers, performance)
    reviews = generate_reviews(suppliers, performance)
    for name, df in {
        "suppliers.csv": suppliers,
        "monthly_performance.csv": performance,
        "invoices.csv": invoices,
        "incidents.csv": incidents,
        "supplier_reviews.csv": reviews,
    }.items():
        df.to_csv(OUT / name, index=False)
    print({name: len(df) for name, df in {
        "suppliers": suppliers, "monthly_performance": performance,
        "invoices": invoices, "incidents": incidents, "supplier_reviews": reviews
    }.items()})

if __name__ == "__main__":
    main()
