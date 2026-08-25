"""
run_all.py — One-click runner for Customer Segmentation Project
Generates dataset → Runs clustering → Opens dashboard
"""

import subprocess
import sys
import os
import webbrowser
from pathlib import Path

print("="*60)
print("   CUSTOMER SEGMENTATION — PROJECT RUNNER")
print("="*60)

def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])

# ── 1. Install dependencies if missing ───────────────────
print("\n📦 Checking dependencies…")
try:
    import numpy, pandas, sklearn, matplotlib, seaborn
    print("   ✅ All libraries already installed")
except ImportError:
    print("   ⬇️  Installing required libraries…")
    for pkg in ['numpy','pandas','scikit-learn','matplotlib','seaborn']:
        print(f"      Installing {pkg}…")
        install(pkg)
    print("   ✅ Libraries installed successfully")

# ── 2. Generate dataset ───────────────────────────────────
print("\n🗃️  Generating customer dataset…")
exec(open('generate_dataset.py').read())

# ── 3. Run segmentation ───────────────────────────────────
print("\n🤖 Running K-Means clustering analysis…")
exec(open('segmentation.py').read())

# ── 4. Open dashboard in browser ─────────────────────────
print("\n🌐 Opening interactive dashboard…")
dash = Path('dashboard.html').absolute().as_uri()
webbrowser.open(dash)
print(f"   Dashboard URL: {dash}")

print("\n" + "="*60)
print("✅ PROJECT COMPLETE!  All outputs:")
print("   📄 customer_data.csv             — Raw dataset")
print("   📄 customer_segments_output.csv  — Clustered results")
print("   📊 output_plots/01_elbow_silhouette.png")
print("   📊 output_plots/02_income_vs_spending_clusters.png")
print("   📊 output_plots/03_cluster_profile_heatmap.png")
print("   📊 output_plots/04_pca_projection.png")
print("   📊 output_plots/05_demographic_breakdown.png")
print("   📊 output_plots/06_boxplots.png")
print("   📊 output_plots/07_correlation_heatmap.png")
print("   🌐 dashboard.html                — Interactive Dashboard")
print("="*60)
