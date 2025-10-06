import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="Aplikasi SPK: Kalkulator (SAW, AHP, WP, TOPSIS)", layout="wide")

# ===== Sidebar =====
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("logo-unpad1.png", width=180)
    st.markdown(
        "<p style='text-align:center; color:#3949AB; font-weight:bold;'>Decision Support System</p>"
        "<p style='text-align:center;'>by Ayumi Fathiyaraisha<br>Teknik Informatika FMIPA UNPAD</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    method = st.selectbox("Pilih Metode:", ["SAW", "AHP", "WP", "TOPSIS"])
    st.divider()
    st.caption("📘 Mode Edukasi — menampilkan langkah-langkah perhitungan secara detail.")

# ===== Header =====
st.markdown("<h2 style='text-align:center; color:#3949AB;'>🧮 DSS Edukasi — Langkah per Langkah</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ===== Input Data =====
default_data = pd.DataFrame({
    "Alternatif": ["A1", "A2", "A3"],
    "C1": [70, 80, 90],
    "C2": [85, 75, 95],
    "C3": [60, 65, 55]
})
st.subheader("Input Data Alternatif & Kriteria")
data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
alternatives = data["Alternatif"].tolist()
criteria = [c for c in data.columns if c != "Alternatif"]
values = data[criteria].to_numpy(dtype=float)

# ===== Input Bobot & Tipe =====
if method in ["SAW", "WP", "TOPSIS"]:
    st.subheader("Bobot & Jenis Kriteria")
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

# ===== Hasil Perhitungan =====
with st.spinner("⏳ Sedang memproses langkah demi langkah..."):
    time.sleep(1)

# === SAW ===
if method == "SAW":
    st.header("🌟 Metode SAW (Simple Additive Weighting)")
    norm = np.zeros_like(values)
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            norm[:, j] = values[:, j] / values[:, j].max()
        else:
            norm[:, j] = values[:, j].min() / values[:, j]

    with st.expander("📘 Langkah 1: Normalisasi Matriks"):
        st.write("Setiap nilai dibagi dengan nilai maksimum (benefit) atau minimum (cost) per kolom.")
        st.dataframe(pd.DataFrame(norm, columns=criteria, index=alternatives), use_container_width=True)

    scores = norm.dot(weights)
    with st.expander("📗 Langkah 2: Hitung Nilai Preferensi"):
        st.write("Nilai preferensi diperoleh dari penjumlahan hasil normalisasi × bobot.")
        st.dataframe(pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4)}), use_container_width=True)

    rank = scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")
    with st.expander("📙 Langkah 3: Hasil Akhir & Ranking", expanded=True):
        st.dataframe(result, use_container_width=True)

# === AHP ===
elif method == "AHP":
    st.header("🧠 Metode AHP (Analytic Hierarchy Process)")
    n = len(criteria)
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

    with st.expander("📘 Langkah 1: Matriks Perbandingan Berpasangan"):
        st.dataframe(pd.DataFrame(pairwise, columns=criteria, index=criteria), use_container_width=True)

    with st.expander("📗 Langkah 2: Normalisasi & Bobot Kriteria"):
        st.dataframe(pd.DataFrame(A_norm.round(4), columns=criteria, index=criteria), use_container_width=True)
        st.write("Bobot Kriteria:", [round(x, 4) for x in w])

    with st.expander("📙 Langkah 3: Uji Konsistensi", expanded=True):
        st.write(f"λmax = {lamda_max:.4f}, CI = {CI:.4f}, CR = {CR:.4f}")

        st.write(f"λmax = {lamda_max:.4f}, CI = {CI:.4f}, CR = {CR:.4f}")
        st.write("**Indikator Rasio Konsistensi (CR):**")

        cr_scaled = min(CR / 0.2, 1.0)
        st.progress(1.0 - cr_scaled)  # semakin kecil CR, bar makin penuh (hijau)
        if CR <= 0.1:
            st.success("✅ Matriks konsisten — nilai CR ≤ 0.1")
        else:
            st.warning("⚠️ Matriks tidak konsisten — CR > 0.1")



    # Pesan status
    if CR <= 0.1:
        st.success("✅ Matriks konsisten — nilai CR ≤ 0.1 menandakan perbandingan antar kriteria stabil.")
    else:
        st.warning("⚠️ Matriks tidak konsisten — CR > 0.1, disarankan revisi perbandingan antar kriteria.")


    norm_alt = values / values.max(axis=0)
    final_scores = norm_alt.dot(w)
    rank = final_scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor Akhir": final_scores.round(4), "Ranking": rank}).sort_values("Ranking")

    with st.expander("📒 Langkah 4: Hasil Akhir"):
        st.dataframe(result, use_container_width=True)

# === WP ===
elif method == "WP":
    st.header("⚙️ Metode WP (Weighted Product)")
    norm = np.zeros_like(values)
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            norm[:, j] = values[:, j] / values[:, j].max()
        else:
            norm[:, j] = values[:, j].min() / values[:, j]

    with st.expander("📘 Langkah 1: Normalisasi Matriks"):
        st.dataframe(pd.DataFrame(norm, columns=criteria, index=alternatives), use_container_width=True)

    S = np.prod(norm ** weights, axis=1)
    with st.expander("📗 Langkah 2: Hitung Nilai S"):
        st.dataframe(pd.DataFrame({"Alternatif": alternatives, "S": S.round(4)}), use_container_width=True)

    V = S / S.sum()
    rank = V.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": V.round(4), "Ranking": rank}).sort_values("Ranking")

    with st.expander("📙 Langkah 3: Normalisasi Nilai S ke V", expanded=True):
        st.dataframe(result, use_container_width=True)

# === TOPSIS ===
elif method == "TOPSIS":
    st.header("🏆 Metode TOPSIS")
    norm = values / np.sqrt((values ** 2).sum(axis=0))
    with st.expander("📘 Langkah 1: Normalisasi Matriks"):
        st.dataframe(pd.DataFrame(norm.round(4), columns=criteria, index=alternatives), use_container_width=True)

    V = norm * weights
    with st.expander("📗 Langkah 2: Matriks Terbobot"):
        st.dataframe(pd.DataFrame(V.round(4), columns=criteria, index=alternatives), use_container_width=True)

    ideal_pos, ideal_neg = np.zeros(len(criteria)), np.zeros(len(criteria))
    for j in range(len(criteria)):
        if types[j] == "Benefit":
            ideal_pos[j], ideal_neg[j] = V[:, j].max(), V[:, j].min()
        else:
            ideal_pos[j], ideal_neg[j] = V[:, j].min(), V[:, j].max()

    with st.expander("📙 Langkah 3: Solusi Ideal Positif & Negatif"):
        st.write("Ideal Positif (+):", np.round(ideal_pos, 4))
        st.write("Ideal Negatif (-):", np.round(ideal_neg, 4))

    D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
    with st.expander("📒 Langkah 4: Jarak ke Solusi Ideal"):
        st.dataframe(pd.DataFrame({"Alternatif": alternatives, "D+": D_pos.round(4), "D-": D_neg.round(4)}), use_container_width=True)

    scores = D_neg / (D_pos + D_neg)
    rank = scores.argsort()[::-1].argsort() + 1
    result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")

    with st.expander("📔 Langkah 5: Hasil Akhir & Ranking", expanded=True):
        st.dataframe(result, use_container_width=True)

st.markdown(
    "<hr><center><p style='font-size:14px;'>© 2025 Ayumi Fathiyaraisha | Aplikasi SPK: Kalkulator (SAW, AHP, WP, TOPSIS)</p></center>",
    unsafe_allow_html=True
)
