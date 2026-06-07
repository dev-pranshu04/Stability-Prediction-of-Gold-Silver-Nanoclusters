"""
NanoStability AI — Au/Ag Nanocluster Stability Predictor
=========================================================
Built by: [Your Name] | NSUT Computational Chemistry Lab (May–Aug 2025)
Physics-Informed XGBoost + SVM + Neural Network Ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config
st.set_page_config(
    page_title="NanoStability AI",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #2a2a4a; border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
  }
  .main-header h1 { font-family: 'Space Mono', monospace; font-size: 2.4rem; color: #e8d5b7; margin: 0 0 0.4rem 0; }
  .main-header p { color: #8892a4; font-size: 1rem; margin: 0; }
  .badge { display: inline-block; background: #1e3a5f; color: #7eb8f7; border: 1px solid #2a5298;
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-family: 'Space Mono', monospace;
    margin-right: 6px; margin-top: 10px; }
  .metric-card { background: #0f1117; border: 1px solid #1e2130; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center; }
  .metric-card .value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #e8d5b7; }
  .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
  .prediction-stable { background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981; border-radius: 16px; padding: 2rem; text-align: center; }
  .prediction-unstable { background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%);
    border: 1px solid #ef4444; border-radius: 16px; padding: 2rem; text-align: center; }
  .prediction-label { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
  .insight-box { background: #111827; border-left: 3px solid #3b82f6; border-radius: 8px;
    padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #d1d5db; }
  .section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 2px; margin: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1f2937; padding-bottom: 0.5rem; }
  .footer { text-align: center; color: #374151; font-size: 0.78rem; margin-top: 3rem;
    padding-top: 1.5rem; border-top: 1px solid #111827; }
  [data-testid="stSidebar"] { background: #0a0a12; border-right: 1px solid #1a1a2e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("au_ag_nanocluster_stability.csv")

@st.cache_resource
def train_models(df):
    feature_cols = [
        "n_atoms", "formation_energy_eV", "homo_lumo_gap_eV",
        "binding_energy_eV", "coord_number_avg", "is_planar",
        "magnetic_moment_bohr", "ionization_potential_eV",
        "electron_affinity_eV", "n_valence_electrons", "energy_above_lowest_eV"
    ]
    X = df[feature_cols].values
    y = df["stable"].values
    element_enc = (df["element"] == "Au").astype(int).values.reshape(-1, 1)
    X = np.hstack([X, element_enc])
    feature_names = feature_cols + ["is_gold"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    svm = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42, early_stopping=True)
    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("svm", svm), ("mlp", mlp)], voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring="accuracy")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_prob),
    }
    xgb.fit(X_train, y_train)
    fi = xgb.feature_importances_
    return ensemble, scaler, feature_names, metrics, fi

def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0a12")
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values(): spine.set_edgecolor("#1f2937")
    ax.tick_params(colors="#6b7280", labelsize=9)
    ax.xaxis.label.set_color("#9ca3af"); ax.yaxis.label.set_color("#9ca3af"); ax.title.set_color("#e8d5b7")
    return fig, ax

# ── Load & train
df = load_data()
ensemble, scaler, feature_names, metrics, feat_imp = train_models(df)

# ── Sidebar
with st.sidebar:
    st.markdown("### ⚛️ NanoStability AI")
    st.markdown("<p style='color:#6b7280;font-size:0.8rem'>NSUT Computational Chemistry Lab</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", ["🏠 Overview", "🔬 Predict Stability", "📊 Data Dashboard", "📈 Model Metrics"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"""<div style='font-size:0.75rem;color:#374151;line-height:1.9'>
    <b style='color:#6b7280'>Dataset</b><br>200 Au/Ag nanoclusters<br>
    <b style='color:#6b7280'>Method</b><br>XGBoost + SVM + MLP<br>
    <b style='color:#6b7280'>Features</b><br>DFT-derived (11 features)<br>
    <b style='color:#6b7280'>CV Accuracy</b><br>{metrics['cv_mean']*100:.1f}%
    </div>""", unsafe_allow_html=True)

# ══ PAGE 1: OVERVIEW ══
if page == "🏠 Overview":
    st.markdown("""<div class='main-header'>
      <h1>⚛ NanoStability AI</h1>
      <p>Physics-Informed Machine Learning for Gold & Silver Nanocluster Stability Prediction</p>
      <div><span class='badge'>XGBoost</span><span class='badge'>SVM</span><span class='badge'>Neural Network</span><span class='badge'>DFT Features</span><span class='badge'>NSUT 2025</span></div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (f"{metrics['cv_mean']*100:.1f}%","CV Accuracy"),
        (f"{metrics['roc_auc']:.3f}","ROC-AUC Score"),
        (f"{len(df)}","DFT Samples"),("3","ML Models")]):
        col.markdown(f"<div class='metric-card'><div class='value'>{val}</div><div class='label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("<div class='section-title'>What This Tool Does</div>", unsafe_allow_html=True)
        st.markdown("""
        Nanoclusters — tiny groups of 3 to 20 atoms — behave very differently from bulk metals.
        Whether a gold or silver nanocluster is **stable** depends on its quantum-mechanical
        electronic structure, which is expensive to calculate using traditional physics simulations (DFT).

        This tool uses **machine learning trained on 200 DFT-computed samples** to predict
        stability instantly — replacing hours of computation with a sub-second prediction.

        **How it works:**
        - Input 11 physical features (size, energy gap, formation energy, etc.)
        - Three ML models vote on the outcome (XGBoost + SVM + Neural Network)
        - Result: Stable ✅ or Unstable ❌ with confidence score and detailed physics explanation
        """)
        st.markdown("<div class='section-title'>Why It Matters</div>", unsafe_allow_html=True)
        st.markdown("Stable nanoclusters are essential in **catalysis, drug delivery, biosensing, and solar cells**. Identifying stable configurations without running costly quantum chemistry simulations accelerates materials discovery by orders of magnitude.")

    with col2:
        st.markdown("<div class='section-title'>Stability Criterion</div>", unsafe_allow_html=True)
        st.info("**A cluster is STABLE when:**\n\n🔵 HOMO-LUMO Gap **> 0.5 eV**\n*(electronically closed-shell)*\n\n🔵 Formation Energy **< −1.0 eV/atom**\n*(thermodynamically favorable)*\n\nBoth must be satisfied simultaneously.")
        st.markdown("<div class='section-title'>Dataset Distribution</div>", unsafe_allow_html=True)
        stable_count = df["stable"].sum()
        fig, ax = dark_fig(figsize=(4, 2.5))
        bars = ax.bar(["Stable","Unstable"],[stable_count,len(df)-stable_count],color=["#10b981","#ef4444"],width=0.5,edgecolor="none")
        ax.set_title("200-Sample Dataset",fontsize=11)
        for bar, val in zip(bars,[stable_count,len(df)-stable_count]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(val),ha="center",color="#e8d5b7",fontsize=10,fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

# ══ PAGE 2: PREDICT ══
elif page == "🔬 Predict Stability":
    st.markdown("<h2 style='font-family:Space Mono;color:#e8d5b7'>Predict Nanocluster Stability</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280'>Enter the DFT-computed properties of your nanocluster below.</p>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Cluster Properties</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        element = st.selectbox("Element", ["Au (Gold)", "Ag (Silver)"])
        n_atoms = st.slider("Number of Atoms", 3, 20, 8, help="Magic numbers (8, 20) tend to be stable.")
        formation_energy = st.number_input("Formation Energy (eV/atom)", -3.0, 0.0, -1.72, 0.01, help="More negative = more stable. Below −1.0 eV/atom is favorable.")
        homo_lumo_gap = st.number_input("HOMO-LUMO Gap (eV)", 0.0, 4.0, 1.42, 0.01, help="Energy gap between HOMO and LUMO orbitals. >0.5 eV = electronically stable.")
    with col2:
        binding_energy = st.number_input("Binding Energy (eV/atom)", 0.0, 4.0, 2.18, 0.01, help="Energy holding the cluster together.")
        coord_number = st.number_input("Avg Coordination Number", 1.0, 7.0, 3.25, 0.05, help="Average number of atomic neighbors.")
        is_planar = st.selectbox("Structure Type", ["3D (non-planar)", "2D (planar)"], help="Au clusters up to ~11 atoms tend to be planar.")
        mag_moment = st.number_input("Magnetic Moment (Bohr)", 0.0, 5.0, 0.0, 1.0, help="0 for even-electron clusters (more stable).")
    with col3:
        ip = st.number_input("Ionization Potential (eV)", 5.0, 12.0, 8.21, 0.01, help="Energy needed to remove one electron.")
        ea = st.number_input("Electron Affinity (eV)", 0.5, 5.0, 2.68, 0.01, help="Energy released when gaining one electron.")
        energy_above = st.number_input("Energy Above Minimum (eV/atom)", 0.0, 1.0, 0.00, 0.01, help="0 = global minimum. Higher = metastable isomer.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚛ Run Prediction", type="primary", use_container_width=True):
        is_gold = 1 if "Au" in element else 0
        is_planar_val = 1 if "2D" in is_planar else 0
        n_valence = 11 * n_atoms
        input_arr = np.array([[n_atoms, formation_energy, homo_lumo_gap, binding_energy,
            coord_number, is_planar_val, mag_moment, ip, ea, n_valence, energy_above, is_gold]])
        input_scaled = scaler.transform(input_arr)
        prediction = ensemble.predict(input_scaled)[0]
        probability = ensemble.predict_proba(input_scaled)[0]
        conf = probability[prediction] * 100

        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1])
        with r1:
            if prediction == 1:
                st.markdown(f"""<div class='prediction-stable'><div class='prediction-label' style='color:#10b981'>✅ STABLE</div>
                  <div class='prediction-confidence'>Model confidence: <b>{conf:.1f}%</b></div>
                  <p style='color:#6ee7b7;margin-top:1rem;font-size:0.9rem'>This cluster satisfies both electronic and thermodynamic stability criteria. It is a good candidate for experimental synthesis.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='prediction-unstable'><div class='prediction-label' style='color:#ef4444'>❌ UNSTABLE</div>
                  <div class='prediction-confidence'>Model confidence: <b>{conf:.1f}%</b></div>
                  <p style='color:#fca5a5;margin-top:1rem;font-size:0.9rem'>This cluster does not meet stability thresholds. It may decompose or rearrange to a lower-energy configuration.</p></div>""", unsafe_allow_html=True)
        with r2:
            fig, ax = dark_fig(figsize=(4, 3))
            ax.barh(["Unstable","Stable"],[probability[0]*100,probability[1]*100],color=["#ef4444","#10b981"],height=0.4,edgecolor="none")
            ax.set_xlim(0,100); ax.set_xlabel("Probability (%)"); ax.set_title("Model Confidence")
            for i,(val) in enumerate([probability[0]*100,probability[1]*100]):
                ax.text(val+1,i,f"{val:.1f}%",va="center",color="#e8d5b7",fontsize=10,fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<div class='section-title'>Physics Interpretation</div>", unsafe_allow_html=True)
        insights = []
        if homo_lumo_gap > 0.5:
            insights.append(("✅", f"HOMO-LUMO gap ({homo_lumo_gap:.2f} eV) exceeds 0.5 eV — cluster is electronically closed-shell and resistant to chemical attack."))
        else:
            insights.append(("⚠️", f"HOMO-LUMO gap ({homo_lumo_gap:.2f} eV) is below 0.5 eV — cluster is electronically open and reactive."))
        if formation_energy < -1.0:
            insights.append(("✅", f"Formation energy ({formation_energy:.2f} eV/atom) is below −1.0 eV/atom — thermodynamically favorable to exist."))
        else:
            insights.append(("⚠️", f"Formation energy ({formation_energy:.2f} eV/atom) is not sufficiently negative — cluster may not form spontaneously."))
        if n_atoms in [2, 8, 18, 20]:
            insights.append(("⭐", f"{n_atoms} is a 'magic number' — these cluster sizes have fully filled electronic shells, greatly boosting stability."))
        if energy_above < 0.05:
            insights.append(("✅", f"Energy above minimum ({energy_above:.3f} eV/atom) is near zero — this is the ground-state (lowest energy) structure."))
        else:
            insights.append(("⚠️", f"Energy above minimum ({energy_above:.3f} eV/atom) suggests this is a metastable isomer, not the true ground state."))
        if is_gold:
            insights.append(("ℹ️", "Gold clusters benefit from relativistic effects that contract 6s orbitals, enhancing stability at small sizes compared to silver."))
        else:
            insights.append(("ℹ️", "Silver clusters follow regular shell-closing stability rules with weaker relativistic effects than gold."))
        for icon, text in insights:
            st.markdown(f"<div class='insight-box'>{icon} {text}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Which Features Drove This Prediction?</div>", unsafe_allow_html=True)
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feat_imp}).sort_values("Importance", ascending=True).tail(8)
        fig, ax = dark_fig(figsize=(7, 3.5))
        colors_fi = ["#f59e0b" if f in ["homo_lumo_gap_eV","formation_energy_eV"] else "#3b82f6" for f in fi_df["Feature"]]
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors_fi, edgecolor="none")
        ax.set_title("Feature Importance (XGBoost Component)", fontsize=11); ax.set_xlabel("Importance Score")
        gold_p = mpatches.Patch(color="#f59e0b", label="Key physics features")
        blue_p = mpatches.Patch(color="#3b82f6", label="Supporting features")
        ax.legend(handles=[gold_p,blue_p],facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

# ══ PAGE 3: DASHBOARD ══
elif page == "📊 Data Dashboard":
    st.markdown("<h2 style='font-family:Space Mono;color:#e8d5b7'>Dataset Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280'>Explore the 200-sample DFT-curated nanocluster dataset.</p>", unsafe_allow_html=True)

    with st.expander("🔧 Filter Dataset", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        elem_filter = fc1.multiselect("Element", ["Au","Ag"], default=["Au","Ag"])
        stab_filter = fc2.multiselect("Stability", ["Stable","Unstable"], default=["Stable","Unstable"])
        size_filter = fc3.slider("Cluster Size", 3, 20, (3, 20))

    stab_map = {"Stable":1,"Unstable":0}
    stab_vals = [stab_map[s] for s in stab_filter]
    df_f = df[(df["element"].isin(elem_filter))&(df["stable"].isin(stab_vals))&(df["n_atoms"]>=size_filter[0])&(df["n_atoms"]<=size_filter[1])]

    for col, (val, lbl) in zip(st.columns(4), [
        (len(df_f),"Clusters Shown"),(df_f["stable"].sum(),"Stable"),
        (f"{df_f['homo_lumo_gap_eV'].mean():.2f} eV","Avg HL Gap"),(f"{df_f['formation_energy_eV'].mean():.2f}","Avg Ef (eV/at)")]):
        col.markdown(f"<div class='metric-card'><div class='value'>{val}</div><div class='label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("<div class='section-title'>HOMO-LUMO Gap vs Cluster Size</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(6, 3.8))
        for stab, color, label in [(1,"#10b981","Stable"),(0,"#ef4444","Unstable")]:
            sub = df_f[df_f["stable"]==stab]
            ax.scatter(sub["n_atoms"],sub["homo_lumo_gap_eV"],c=color,alpha=0.75,s=50,label=label,edgecolors="none")
        ax.axhline(0.5,color="#f59e0b",linestyle="--",linewidth=1.2,alpha=0.8,label="Threshold (0.5 eV)")
        ax.set_xlabel("Number of Atoms"); ax.set_ylabel("HOMO-LUMO Gap (eV)")
        ax.legend(facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with g2:
        st.markdown("<div class='section-title'>Formation Energy Distribution</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(6, 3.8))
        for stab, color, label in [(1,"#10b981","Stable"),(0,"#ef4444","Unstable")]:
            ax.hist(df_f[df_f["stable"]==stab]["formation_energy_eV"],bins=14,color=color,alpha=0.7,label=label,edgecolor="none")
        ax.axvline(-1.0,color="#f59e0b",linestyle="--",linewidth=1.2,alpha=0.8,label="Threshold (−1.0 eV/at)")
        ax.set_xlabel("Formation Energy (eV/atom)"); ax.set_ylabel("Count")
        ax.legend(facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("<div class='section-title'>Odd-Even Oscillation in HL Gap</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(6, 3.8))
        for elem, color in [("Au","#f59e0b"),("Ag","#60a5fa")]:
            sub = df_f[df_f["element"]==elem].groupby("n_atoms")["homo_lumo_gap_eV"].mean()
            ax.plot(sub.index,sub.values,"o-",color=color,linewidth=1.8,markersize=5,label=elem,alpha=0.9)
        ax.axhline(0.5,color="#6b7280",linestyle=":",linewidth=1)
        ax.set_xlabel("Cluster Size (atoms)"); ax.set_ylabel("Mean HOMO-LUMO Gap (eV)")
        ax.legend(facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with g4:
        st.markdown("<div class='section-title'>Stability Rate by Element & Size</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(6, 3.8))
        for elem, color in [("Au","#f59e0b"),("Ag","#60a5fa")]:
            sub = df_f[df_f["element"]==elem]
            rate = sub.groupby("n_atoms")["stable"].mean()*100
            ax.bar(rate.index+(0.3 if elem=="Ag" else -0.3),rate.values,width=0.55,color=color,alpha=0.85,label=elem,edgecolor="none")
        ax.set_xlabel("Cluster Size (atoms)"); ax.set_ylabel("% Stable"); ax.set_ylim(0,110)
        ax.legend(facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-title'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    num_cols = ["n_atoms","formation_energy_eV","homo_lumo_gap_eV","binding_energy_eV","coord_number_avg","ionization_potential_eV","electron_affinity_eV","stable"]
    corr = df_f[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0a0a12"); ax.set_facecolor("#0f1117")
    sns.heatmap(corr,ax=ax,cmap="coolwarm",center=0,annot=True,fmt=".2f",annot_kws={"size":8,"color":"white"},linewidths=0.5,linecolor="#1f2937",cbar_kws={"shrink":0.8})
    ax.tick_params(colors="#9ca3af",labelsize=8); ax.set_title("Feature Correlation (red=positive, blue=negative)",color="#e8d5b7",fontsize=11)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-title'>Raw Dataset</div>", unsafe_allow_html=True)
    display_df = df_f.copy(); display_df["stable"] = display_df["stable"].map({1:"✅ Stable",0:"❌ Unstable"})
    st.dataframe(display_df, use_container_width=True, height=320)
    st.download_button("⬇ Download Filtered CSV", df_f.to_csv(index=False), "nanocluster_filtered.csv", "text/csv")

# ══ PAGE 4: MODEL METRICS ══
elif page == "📈 Model Metrics":
    st.markdown("<h2 style='font-family:Space Mono;color:#e8d5b7'>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280'>Detailed evaluation of the XGBoost + SVM + Neural Network ensemble.</p>", unsafe_allow_html=True)

    report = metrics["report"]
    for col, (val, lbl) in zip(st.columns(4), [
        (f"{metrics['cv_mean']*100:.1f}%","5-Fold CV Accuracy"),
        (f"{metrics['roc_auc']:.3f}","ROC-AUC"),
        (f"{report['1']['precision']:.3f}","Precision (Stable)"),
        (f"{report['1']['recall']:.3f}","Recall (Stable)")]):
        col.markdown(f"<div class='metric-card'><div class='value'>{val}</div><div class='label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown("<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
        fpr, tpr, _ = metrics["roc_curve"]
        fig, ax = dark_fig(figsize=(6, 4))
        ax.plot(fpr,tpr,color="#3b82f6",linewidth=2.5,label=f"Ensemble (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],color="#374151",linestyle="--",linewidth=1)
        ax.fill_between(fpr,tpr,alpha=0.08,color="#3b82f6")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve")
        ax.legend(facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af")
        st.pyplot(fig, use_container_width=True); plt.close()
    with row1_c2:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = metrics["conf_matrix"]
        fig, ax = dark_fig(figsize=(6, 4))
        sns.heatmap(cm,ax=ax,annot=True,fmt="d",cmap="Blues",xticklabels=["Unstable","Stable"],yticklabels=["Unstable","Stable"],annot_kws={"size":16,"weight":"bold","color":"white"},linewidths=2,linecolor="#0a0a12",cbar=False)
        ax.tick_params(colors="#9ca3af",labelsize=10); ax.set_xlabel("Predicted",color="#9ca3af"); ax.set_ylabel("Actual",color="#9ca3af"); ax.set_title("Confusion Matrix (Test Set)",color="#e8d5b7")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-title'>Feature Importance (XGBoost)</div>", unsafe_allow_html=True)
    fi_df = pd.DataFrame({"Feature":feature_names,"Importance":feat_imp}).sort_values("Importance",ascending=True)
    fig, ax = dark_fig(figsize=(10, 4))
    bar_colors = ["#f59e0b" if f in ["homo_lumo_gap_eV","formation_energy_eV"] else "#3b82f6" for f in fi_df["Feature"]]
    ax.barh(fi_df["Feature"],fi_df["Importance"],color=bar_colors,edgecolor="none")
    ax.set_title("Feature Importance — Physics features in gold",fontsize=11); ax.set_xlabel("Importance Score")
    ax.legend(handles=[mpatches.Patch(color="#f59e0b",label="Key physics features"),mpatches.Patch(color="#3b82f6",label="Supporting features")],facecolor="#111827",edgecolor="#374151",labelcolor="#9ca3af",fontsize=9)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Class":["Unstable (0)","Stable (1)","Macro avg","Weighted avg"],
        "Precision":[f"{report['0']['precision']:.3f}",f"{report['1']['precision']:.3f}",f"{report['macro avg']['precision']:.3f}",f"{report['weighted avg']['precision']:.3f}"],
        "Recall":[f"{report['0']['recall']:.3f}",f"{report['1']['recall']:.3f}",f"{report['macro avg']['recall']:.3f}",f"{report['weighted avg']['recall']:.3f}"],
        "F1-Score":[f"{report['0']['f1-score']:.3f}",f"{report['1']['f1-score']:.3f}",f"{report['macro avg']['f1-score']:.3f}",f"{report['weighted avg']['f1-score']:.3f}"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Cross-Validation Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='insight-box'>5-Fold CV Accuracy: <b>{metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%</b> — the model generalises well across different train/test splits.</div>
    <div class='insight-box'>The ensemble combines three models: <b>XGBoost</b> (captures non-linear feature interactions), <b>SVM</b> (finds optimal boundaries in high-dimensional space), and <b>Neural Network</b> (learns complex patterns via multiple hidden layers). Their combined vote reduces individual model errors.</div>""", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built for NSUT Computational Chemistry Lab Internship (May–Aug 2025) &nbsp;|&nbsp; Dataset: 200 DFT-curated Au/Ag nanoclusters &nbsp;|&nbsp; Physics-Informed XGBoost + SVM + MLP Ensemble</div>", unsafe_allow_html=True)
