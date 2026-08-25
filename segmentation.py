"""
Customer Segmentation using K-Means Clustering
Main Analysis Module - Complete Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Color palette ───────────────────────────────────────────────────────────
CLUSTER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                  '#DDA0DD', '#98D8C8', '#F7DC6F']
DARK_BG   = '#1a1a2e'
CARD_BG   = '#16213e'
ACCENT    = '#e94560'
TEXT_COL  = '#eaeaea'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    '#444',
    'axes.labelcolor':   TEXT_COL,
    'xtick.color':       TEXT_COL,
    'ytick.color':       TEXT_COL,
    'text.color':        TEXT_COL,
    'grid.color':        '#2a2a4a',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
})

os.makedirs('output_plots', exist_ok=True)

# ════════════════════════════════════════════════════════
# 1. LOAD & PREVIEW DATA
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
print("="*60)

try:
    df = pd.read_csv('customer_data.csv')
    print(f"\n✅ Dataset loaded: {df.shape[0]} customers × {df.shape[1]} features\n")
except FileNotFoundError:
    # Generate inline if CSV not found
    print("📦 Generating dataset on-the-fly…")
    exec(open('generate_dataset.py').read())
    df = pd.read_csv('customer_data.csv')

print(df.head())
print("\n📊 Basic Statistics:")
print(df.describe().round(2))

# ════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("STEP 2: DATA PREPROCESSING")
print("─"*60)

# Missing value check
print(f"\nMissing values:\n{df.isnull().sum()}")

# Encode Gender
df['Gender_Encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
print("\n✅ Gender encoded: Male→0, Female→1")

# Features for clustering (2D primary set)
FEATURES_2D  = ['Annual_Income_k', 'Spending_Score']
FEATURES_ALL = ['Age', 'Annual_Income_k', 'Spending_Score',
                 'Purchase_Frequency', 'Membership_Years']

scaler   = StandardScaler()
X_2d     = scaler.fit_transform(df[FEATURES_2D])
X_all    = scaler.fit_transform(df[FEATURES_ALL])

print(f"✅ Features scaled using StandardScaler")

# ════════════════════════════════════════════════════════
# 3. ELBOW METHOD  — find optimal K
# ════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("STEP 3: ELBOW METHOD — Finding Optimal K")
print("─"*60)

wcss        = []
silhouettes = []
K_range     = range(2, 12)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                max_iter=300, random_state=42)
    km.fit(X_2d)
    wcss.append(km.inertia_)
    sil = silhouette_score(X_2d, km.labels_)
    silhouettes.append(sil)
    print(f"  K={k:2d}  WCSS={km.inertia_:8.1f}  Silhouette={sil:.4f}")

optimal_k   = K_range[silhouettes.index(max(silhouettes))]
print(f"\n🎯 Best K by Silhouette Score = {optimal_k}")

# ─── Plot Elbow + Silhouette ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Finding Optimal Number of Clusters', fontsize=15,
             color=TEXT_COL, fontweight='bold', y=1.02)

# Elbow
ax1 = axes[0]
ax1.plot(K_range, wcss, 'o-', color=ACCENT, linewidth=2.5,
         markersize=8, markerfacecolor='white', markeredgecolor=ACCENT)
ax1.axvline(optimal_k, color='#FFEAA7', linestyle='--', alpha=0.8,
            label=f'Optimal K = {optimal_k}')
ax1.set_title('Elbow Method (WCSS)', color=TEXT_COL, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Within-Cluster Sum of Squares')
ax1.legend(facecolor='#2a2a4a', edgecolor='none', labelcolor=TEXT_COL)
ax1.grid(True)
ax1.fill_between(K_range, wcss, alpha=0.1, color=ACCENT)

# Silhouette
ax2 = axes[1]
bars = ax2.bar(K_range, silhouettes,
               color=[ACCENT if k == optimal_k else '#4ECDC4' for k in K_range],
               edgecolor='none', alpha=0.85)
ax2.axhline(max(silhouettes), color='#FFEAA7', linestyle='--', alpha=0.7)
ax2.set_title('Silhouette Score per K', color=TEXT_COL, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, axis='y')
ax2.set_xticks(list(K_range))

for bar, val in zip(bars, silhouettes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8,
             color=TEXT_COL)

plt.tight_layout()
plt.savefig('output_plots/01_elbow_silhouette.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/01_elbow_silhouette.png")

# ════════════════════════════════════════════════════════
# 4. TRAIN K-MEANS  (K = optimal_k)
# ════════════════════════════════════════════════════════
print("\n" + "─"*60)
print(f"STEP 4: TRAINING K-MEANS  (K={optimal_k})")
print("─"*60)

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=15,
                max_iter=300, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_2d)

print(f"\n✅ K-Means trained.  Inertia = {kmeans.inertia_:.2f}")
print(f"   Final Silhouette Score = {silhouette_score(X_2d, df['Cluster']):.4f}")

# Cluster sizes
print("\n📦 Cluster Sizes:")
print(df['Cluster'].value_counts().sort_index().to_string())

# ════════════════════════════════════════════════════════
# 5. CLUSTER LABELS & PROFILES
# ════════════════════════════════════════════════════════
# Compute cluster centroids (original scale)
centroids = df.groupby('Cluster')[FEATURES_ALL + ['Spending_Score',
             'Annual_Income_k']].mean()

# Assign human-readable labels based on Income & Spending
def label_cluster(row):
    inc = row['Annual_Income_k']
    spd = row['Spending_Score']
    if inc >= 90 and spd >= 65:
        return 'Premium / High Spenders'
    elif inc >= 70 and spd >= 50:
        return 'Loyal / Regular Spenders'
    elif inc >= 50 and spd < 45:
        return 'Potential / Savers'
    elif inc < 50 and spd >= 55:
        return 'Impulse / Young Spenders'
    elif inc < 40 and spd < 40:
        return 'Budget / Low Spenders'
    else:
        return 'Moderate / Balanced'

cluster_labels = {}
for cid, row in centroids.iterrows():
    cluster_labels[cid] = label_cluster(row)

df['Segment'] = df['Cluster'].map(cluster_labels)

print("\n🏷️  Cluster Segment Labels:")
for cid, lbl in cluster_labels.items():
    count = (df['Cluster'] == cid).sum()
    print(f"  Cluster {cid}: {lbl}  ({count} customers)")

# ════════════════════════════════════════════════════════
# 6. VISUALIZATION — Primary Scatter: Income vs Spending
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

for cid in sorted(df['Cluster'].unique()):
    mask  = df['Cluster'] == cid
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    label = f"Cluster {cid}: {cluster_labels[cid]}"
    ax.scatter(df.loc[mask, 'Annual_Income_k'],
               df.loc[mask, 'Spending_Score'],
               c=color, label=label, s=80, alpha=0.8,
               edgecolors='white', linewidths=0.4)

# Plot centroids
for cid, (inc, spd) in enumerate(
        zip(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])):
    # de-normalize (approx)
    inc_orig = df.groupby('Cluster')['Annual_Income_k'].mean()[cid]
    spd_orig = df.groupby('Cluster')['Spending_Score'].mean()[cid]
    ax.scatter(inc_orig, spd_orig, s=250, c='white',
               marker='*', zorder=5, edgecolors='black', linewidths=0.8)
    ax.annotate(f'C{cid}', (inc_orig, spd_orig),
                textcoords='offset points', xytext=(8, 5),
                fontsize=9, color='white', fontweight='bold')

ax.set_title('Customer Segmentation\nAnnual Income vs Spending Score',
             fontsize=15, fontweight='bold', color=TEXT_COL, pad=15)
ax.set_xlabel('Annual Income (₹ thousands)', fontsize=12)
ax.set_ylabel('Spending Score (1–100)', fontsize=12)
ax.legend(loc='upper left', facecolor='#2a2a4a', edgecolor='#444',
          labelcolor=TEXT_COL, fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Watermark
ax.text(0.98, 0.02, '★ Customer Segmentation System',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, color='#555', style='italic')

plt.tight_layout()
plt.savefig('output_plots/02_income_vs_spending_clusters.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("\n📈 Plot saved: output_plots/02_income_vs_spending_clusters.png")

# ════════════════════════════════════════════════════════
# 7. CLUSTER PROFILE HEATMAP
# ════════════════════════════════════════════════════════
profile_cols = ['Age', 'Annual_Income_k', 'Spending_Score',
                'Purchase_Frequency', 'Membership_Years']
profile = df.groupby('Cluster')[profile_cols].mean().round(1)
profile.index = [f"C{i}: {cluster_labels[i][:16]}" for i in profile.index]

profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

sns.heatmap(profile_norm.T, annot=profile.T, fmt='.1f',
            cmap='RdYlGn', ax=ax, linewidths=0.5,
            linecolor='#2a2a4a', annot_kws={'size': 10},
            cbar_kws={'label': 'Normalized Score'})

ax.set_title('Cluster Profile Heatmap\n(Normalized Feature Averages)',
             fontsize=14, fontweight='bold', color=TEXT_COL, pad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right',
                   fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('output_plots/03_cluster_profile_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/03_cluster_profile_heatmap.png")

# ════════════════════════════════════════════════════════
# 8. PCA — 3D-LIKE SCATTER (2D PCA of ALL features)
# ════════════════════════════════════════════════════════
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

for cid in sorted(df['Cluster'].unique()):
    mask  = df['Cluster'] == cid
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    ax.scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'],
               c=color, label=f'C{cid}: {cluster_labels[cid]}',
               s=70, alpha=0.8, edgecolors='white', linewidths=0.3)

var = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance)', fontsize=11)
ax.set_title(f'PCA Projection — All Features\n'
             f'({var[0]+var[1]:.1f}% total variance explained)',
             fontsize=14, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor='#2a2a4a', edgecolor='#444', labelcolor=TEXT_COL,
          fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/04_pca_projection.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/04_pca_projection.png")

# ════════════════════════════════════════════════════════
# 9. DEMOGRAPHIC BREAKDOWN PER CLUSTER
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Demographic Breakdown by Customer Segment',
             fontsize=14, fontweight='bold', color=TEXT_COL, y=1.02)

# (a) Age distribution violin
ax = axes[0]
data_per_cluster = [df.loc[df['Cluster']==c, 'Age'].values
                    for c in sorted(df['Cluster'].unique())]
parts = ax.violinplot(data_per_cluster, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
    pc.set_alpha(0.8)
parts['cmedians'].set_color('white')
parts['cbars'].set_color('#888')
parts['cmins'].set_color('#888')
parts['cmaxes'].set_color('#888')
ax.set_title('Age Distribution', fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Age')
ax.set_xticks(range(1, optimal_k + 1))
ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
ax.grid(True, axis='y', alpha=0.3)

# (b) Gender pie per cluster — aggregate
ax = axes[1]
gender_counts = df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0)
x = np.arange(optimal_k)
w = 0.35
ax.bar(x - w/2, gender_pct['Male'],  width=w, label='Male',
       color='#45B7D1', alpha=0.85)
ax.bar(x + w/2, gender_pct['Female'], width=w, label='Female',
       color='#FF6B9D', alpha=0.85)
ax.set_title('Gender Ratio per Cluster', fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Proportion')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
ax.legend(facecolor='#2a2a4a', labelcolor=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)

# (c) Cluster size donut
ax = axes[2]
sizes  = df['Cluster'].value_counts().sort_index()
colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in sizes.index]
wedges, texts, autotexts = ax.pie(
    sizes, labels=[f'C{i}' for i in sizes.index],
    colors=colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2)
)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(9)
ax.set_title('Cluster Size Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('output_plots/05_demographic_breakdown.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/05_demographic_breakdown.png")

# ════════════════════════════════════════════════════════
# 10. BOX PLOTS — Income & Spending
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('Feature Distribution by Cluster', fontsize=14,
             fontweight='bold', color=TEXT_COL)

for ax, feat, title in zip(axes,
        ['Annual_Income_k', 'Spending_Score'],
        ['Annual Income (₹k)', 'Spending Score']):
    data = [df.loc[df['Cluster']==c, feat].values
            for c in sorted(df['Cluster'].unique())]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        patch.set_alpha(0.8)
    for elem in ['whiskers', 'caps', 'fliers']:
        for item in bp[elem]:
            item.set_color('#888')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output_plots/06_boxplots.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/06_boxplots.png")

# ════════════════════════════════════════════════════════
# 11. CORRELATION HEATMAP
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
corr = df[FEATURES_ALL + ['Gender_Encoded']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size': 9}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=13,
             fontweight='bold', color=TEXT_COL)
plt.tight_layout()
plt.savefig('output_plots/07_correlation_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("📈 Plot saved: output_plots/07_correlation_heatmap.png")

# ════════════════════════════════════════════════════════
# 12. SAVE RESULTS
# ════════════════════════════════════════════════════════
df.to_csv('customer_segments_output.csv', index=False)

# ════════════════════════════════════════════════════════
# 13. FINAL REPORT
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   FINAL SEGMENTATION REPORT")
print("="*60)
report = df.groupby('Cluster').agg(
    Segment      = ('Segment', 'first'),
    Count        = ('Customer_ID', 'count'),
    Avg_Age      = ('Age', 'mean'),
    Avg_Income   = ('Annual_Income_k', 'mean'),
    Avg_Spending = ('Spending_Score', 'mean'),
    Avg_PurchFreq= ('Purchase_Frequency', 'mean'),
    Avg_MembYrs  = ('Membership_Years', 'mean'),
).round(1)
print(report.to_string())

print("\n💼 BUSINESS INSIGHTS:")
insights = {
    'Premium / High Spenders':    '→ Offer VIP loyalty programs, premium products',
    'Loyal / Regular Spenders':   '→ Provide reward points and seasonal discounts',
    'Potential / Savers':         '→ Send targeted upsell campaigns and EMI offers',
    'Impulse / Young Spenders':   '→ Social media marketing, flash sales',
    'Budget / Low Spenders':      '→ Basic promotions, budget product lines',
    'Moderate / Balanced':        '→ General newsletters and mid-tier offers',
}
for seg, tip in insights.items():
    count = (df['Segment'] == seg).sum()
    if count > 0:
        print(f"\n  [{seg}] — {count} customers\n  {tip}")

print("\n" + "="*60)
print("✅ SEGMENTATION COMPLETE!")
print(f"   Output CSV : customer_segments_output.csv")
print(f"   Plots saved: output_plots/ (7 charts)")
print("="*60)
