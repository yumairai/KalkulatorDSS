import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="DSS Calculator (SAW, AHP, WP, TOPSIS)", layout="centered")

st.title("🧮 Decision Support System Calculator")
st.write("Pilih metode DSS dan masukkan data alternatif serta kriteria di bawah ini.")

method = st.selectbox("Pilih Metode:", ["SAW", "AHP", "WP", "TOPSIS"])

st.subheader("Input Data Alternatif dan Kriteria")
default_data = pd.DataFrame({
    "Alternatif": ["A1", "A2", "A3"],
    "C1": [70, 80, 90],
    "C2": [85, 75, 95],
    "C3": [60, 65, 55]
})
data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
alternatives = data["Alternatif"].tolist()
criteria = [c for c in data.columns if c != "Alternatif"]
values = data[criteria].to_numpy(dtype=float)

if method in ["SAW", "WP", "TOPSIS"]:
    st.subheader("Bobot dan Jenis Kriteria")
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

if method == "SAW":
    st.subheader("🔹 Metode SAW (Simple Additive Weighting)")
    norm = np.zeros_like(values)
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            norm[:, j] = values[:, j] / values[:, j].max()
        else:
            norm[:, j] = values[:, j].min() / values[:, j]
    scores = norm.dot(weights)
    rank = scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")
    st.dataframe(result, use_container_width=True)

elif method == "AHP":
    st.subheader("🔹 Metode AHP (Analytic Hierarchy Process)")
    n = len(criteria)
    st.write(f"Masukkan matriks perbandingan berpasangan ({n}×{n}) untuk kriteria:")
    pairwise = np.ones((n, n))
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
    st.write("**Bobot Kriteria:**", [round(x, 4) for x in w])
    st.write(f"**λmax:** {lamda_max:.4f}  |  **CI:** {CI:.4f}  |  **CR:** {CR:.4f}")
    if CR <= 0.1:
        st.success("✅ Matriks konsisten (CR ≤ 0.1)")
    else:
        st.warning("⚠️ Matriks tidak konsisten (CR > 0.1)")
    st.markdown("---")
    st.write("Gunakan bobot di atas untuk menghitung skor alternatif (perkalian bobot × nilai normalisasi).")
    norm_alt = values / values.max(axis=0)
    final_scores = norm_alt.dot(w)
    rank = final_scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor Akhir": final_scores.round(4), "Ranking": rank}).sort_values("Ranking")
    st.dataframe(result, use_container_width=True)

elif method == "WP":
    st.subheader("🔹 Metode WP (Weighted Product)")
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
    st.dataframe(result, use_container_width=True)

elif method == "TOPSIS":
    st.subheader("🔹 Metode TOPSIS")
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
    st.dataframe(result, use_container_width=True)
