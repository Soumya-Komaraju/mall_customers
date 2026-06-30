import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config("Customer Segmentation", "🛍️", layout="wide")
st.title("🛍️ Customer Segmentation — K-Means")

# ── Upload / load data ────────────────────────────────────────────────────────
file = st.sidebar.file_uploader("Upload CSV", type="csv")
df   = pd.read_csv(file if file else "Mall_Customers.csv")

features = ["Age", "AnnualIncome", "SpendingScore"]
k        = st.sidebar.slider("Clusters (k)", 2, 10, 5)

# ── Cluster ───────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X       = scaler.fit_transform(df[features])
km      = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
df["Cluster"] = km.labels_
sil     = silhouette_score(X, km.labels_)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Customers",        len(df))
c2.metric("Clusters",         k)
c3.metric("Silhouette Score", f"{sil:.3f}")

COLORS = ["#4C6EF5","#F59E0B","#10B981","#EF4444","#8B5CF6",
          "#EC4899","#06B6D4","#84CC16","#F97316","#A78BFA"]

tab1, tab2, tab3, tab4 = st.tabs(["Overview","2D Scatter","3D Scatter","Elbow"])

# ── Overview ─────────────────────────────────────────────────────────────────
with tab1:
    st.dataframe(df.groupby("Cluster")[features].mean().round(1), use_container_width=True)
    fig, ax = plt.subplots()
    counts  = df["Cluster"].value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values,
           color=[COLORS[i] for i in counts.index])
    ax.set(xlabel="Cluster", ylabel="Count", title="Customers per Cluster")
    st.pyplot(fig); plt.close()

# ── 2D Scatter ────────────────────────────────────────────────────────────────
with tab2:
    x_ax = st.selectbox("X", features, index=1)
    y_ax = st.selectbox("Y", features, index=2)
    fig, ax = plt.subplots()
    for c in range(k):
        m = df["Cluster"] == c
        ax.scatter(df.loc[m, x_ax], df.loc[m, y_ax],
                   color=COLORS[c], label=f"Cluster {c}", alpha=0.7, s=50)
    centers = scaler.inverse_transform(km.cluster_centers_)
    ax.scatter(centers[:, features.index(x_ax)], centers[:, features.index(y_ax)],
               marker="X", s=200, c="white", edgecolors="black", zorder=5, label="Centers")
    ax.set(xlabel=x_ax, ylabel=y_ax); ax.legend()
    st.pyplot(fig); plt.close()

# ── 3D Scatter ────────────────────────────────────────────────────────────────
with tab3:
    fig = plt.figure(); ax3 = fig.add_subplot(111, projection="3d")
    for c in range(k):
        m = df["Cluster"] == c
        ax3.scatter(df.loc[m,"Age"], df.loc[m,"AnnualIncome"], df.loc[m,"SpendingScore"],
                    color=COLORS[c], label=f"Cluster {c}", s=40, alpha=0.7)
    ax3.set(xlabel="Age", ylabel="AnnualIncome", zlabel="SpendingScore")
    ax3.legend(fontsize=7)
    st.pyplot(fig); plt.close()

# ── Elbow ─────────────────────────────────────────────────────────────────────
with tab4:
    ks, wcss, sils = zip(*[
        (i, KMeans(n_clusters=i, random_state=42, n_init="auto").fit(X).inertia_,
         silhouette_score(X, KMeans(n_clusters=i, random_state=42, n_init="auto").fit_predict(X)))
        for i in range(2, 11)
    ])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ks, wcss, "o-"); a1.set(title="Elbow (WCSS)", xlabel="k")
    a2.plot(ks, sils, "o-", color="green"); a2.set(title="Silhouette", xlabel="k")
    st.pyplot(fig); plt.close()

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Predict Cluster")
cols = st.columns(3)
vals = [cols[i].number_input(features[i], value=int(df[features[i]].mean())) for i in range(3)]
pred = int(km.predict(scaler.transform([vals]))[0])
st.success(f"Predicted **Cluster {pred}** · colour {COLORS[pred]}")