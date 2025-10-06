import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="DSS Calculator (SAW, AHP, WP, TOPSIS)", layout="wide")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Unpad_logo.png", width=100)
    st.title("🎓 DSS Calculator")
    st.caption("by **Ayumi Fathiyaraisha** — Teknik Informatika FMIPA UNPAD")
    st.divider()
    st.write("Pilih metode untuk perhitungan:")
    method = st.selectbox("Metode:", ["SAW", "AHP", "WP", "TOPSIS"])
    st.divider()
    st.write("✨ Sistem Pendukung Keputusan untuk latihan perbandingan berbagai metode.")

st.markdown(
    "<h2 style='text-align:center; color:#3949AB;'>🧮 Decision Support System Dashboard</h2>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

default_data = pd.DataFrame({
    "Alternatif": ["A1", "A2", "A3"],
    "C1": [70, 80, 90],
    "C2": [85, 75, 95],
    "C3": [60, 65, 55]
})
st.markdown("### Input Data Alternatif & Kriteria")
data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
alternatives = data["Alternatif"].tolist()
criteria = [c for c in data.columns if c != "Alternatif"]
values = data[criteria].to_numpy(dtype=float)

if method in ["SAW", "WP", "TOPSIS"]:
    st.markdown("### Bobot & Jenis Kriteria")
    cols = st.columns(len(criteria))
    weights = []
    for i, c in enumerate(criteria):
        with cols[i]:
            w = st.number_input(f"Bobot {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
            weights.append(w)
    weights = np.array(weights)
    weights = weights / weights.sum()

    cols2 = st.columns(len(criteria))
    types = []
    for i, c in enumerate(criteria):
        with cols2[i]:
            t = st.selectbox(f"Tipe {c}", ["Benefit", "Cost"], key=f"type_{c}")
            types.append(t)

with st.spinner("⏳ Sedang menghitung ranking..."):
    time.sleep(1)

if method == "SAW":
    norm = np.zeros_like(values)
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            norm[:, j] = values[:, j] / values[:, j].max()
        else:
            norm[:, j] = values[:, j].min() / values[:, j]
    scores = norm.dot(weights)
    rank = scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")

elif method == "AHP":
    n = len(criteria)
    pairwise = np.ones((n, n))
    st.markdown("### Matriks Perbandingan Kriteria (AHP)")
    for i in range(n):
        for j in range(i + 1, n):
            val = st.number_input(f"{criteria[i]} dibanding {criteria[j]}", value=1.0, min_value=0.1, step=0.1)
            pairwise[i, j] = val
            pairwise[j, i] = 1 / val
    col_sum = pairwise.sum(axis=0)
    A_norm = pairwise / col_sum
    w = A_norm.mean(axis=1)
    Aw = pairwise.dot(w)
    lamda_max = (Aw / w).mean()
    CI = (lamda_max - n) / (n - 1)
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='padding:20px; border-radius:15px; background-color:white; box-shadow:0 0 10px rgba(0,0,0,0.1)'>"
            f"<h4 style='color:#3949AB;'>Bobot Kriteria</h4><p>{[round(x,4) for x in w]}</p></div>",
            unsafe_allow_html=True
        )
    with c2:
        status = "✅ Konsisten" if CR <= 0.1 else "⚠️ Tidak Konsisten"
        st.markdown(
            f"<div style='padding:20px; border-radius:15px; background-color:white; box-shadow:0 0 10px rgba(0,0,0,0.1)'>"
            f"<h4 style='color:#3949AB;'>Konsistensi</h4><p>λmax = {lamda_max:.4f}<br>CI = {CI:.4f}<br>CR = {CR:.4f}<br>{status}</p></div>",
            unsafe_allow_html=True
        )
    norm_alt = values / values.max(axis=0)
    final_scores = norm_alt.dot(w)
    rank = final_scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor Akhir": final_scores.round(4), "Ranking": rank}).sort_values("Ranking")

elif method == "WP":
    norm = np.zeros_like(values)
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            norm[:, j] = values[:, j] / values[:, j].max()
        else:
            norm[:, j] = values[:, j].min() / values[:, j]
    S = np.prod(norm ** weights, axis=1)
    V = S / S.sum()
    rank = V.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": V.round(4), "Ranking": rank}).sort_values("Ranking")

elif method == "TOPSIS":
    norm = values / np.sqrt((values ** 2).sum(axis=0))
    V = norm * weights
    ideal_pos, ideal_neg = np.zeros(len(criteria)), np.zeros(len(criteria))
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            ideal_pos[j], ideal_neg[j] = V[:, j].max(), V[:, j].min()
        else:
            ideal_pos[j], ideal_neg[j] = V[:, j].min(), V[:, j].max()
    D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
    scores = D_neg / (D_pos + D_neg)
    rank = scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#3949AB;'>📊 Hasil Perankingan</h4>", unsafe_allow_html=True)
st.dataframe(result, use_container_width=True)

st.markdown(
    "<hr><center><p style='font-size:14px;'>© 2025 Ayumi Fathiyaraisha | DSS App (SAW, AHP, WP, TOPSIS)</p></center>",
    unsafe_allow_html=True
)
