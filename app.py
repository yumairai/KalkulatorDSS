import streamlit as st
import numpy as np
import pandas as pd
import time

# ===== Konfigurasi Halaman =====
st.set_page_config(page_title="Aplikasi SPK: Kalkulator (SAW, AHP, WP, TOPSIS)", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "main"

def go_to(page_name):
    st.session_state.page = page_name

# ===== Sidebar =====
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .sidebar-link {
            text-align: center;
            font-size: 14px;
            background-color: #E8EAF6;
            border-radius: 10px;
            padding: 6px;
            margin-top: 10px;
            color: #1A237E !important;
            font-weight: 600;
            text-decoration: none;
            display: block;
        }
        .sidebar-link:hover {
            background-color: #C5CAE9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image("logo-unpad1.png", width=120)
    st.markdown(
        """
        <p style='text-align:center; color:#3949AB; font-weight:600; font-size:15px; margin-bottom:4px;'>
            Decision Support System
        </p>
        <p style='text-align:center; font-size:12px; margin-top:0;'>
            by <b>Ayumi Fathiyaraisha</b><br>Teknik Informatika FMIPA UNPAD
        </p>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Navigasi
    st.markdown(
        f"""
        <a href='?page=main' target='_self' class='sidebar-link'>🏠 Halaman Utama</a>
        <a href='?page=guide' target='_self' class='sidebar-link'>📘 User Guide</a>
        """,
        unsafe_allow_html=True
    )
    st.divider()
    st.caption("Pilih metode dan masukkan data untuk mulai perhitungan.")

# ===== Logika Halaman =====
if st.session_state.page == "guide" or "guide" in st.query_params:
    st.title("📘 Panduan Penggunaan – User Guide")
    st.markdown(
        """
        <div style='background-color:#F5F5F5; padding:20px; border-radius:15px;'>
        <h4>🧩 Langkah-langkah Menggunakan Aplikasi</h4>
        <ol>
        <li>Pilih metode DSS di sidebar (SAW, AHP, WP, atau TOPSIS).</li>
        <li>Masukkan <b>data alternatif dan kriteria</b> pada tabel utama.</li>
        <li>Tambahkan atau hapus kriteria sesuai kebutuhan menggunakan tombol di bawah tabel.</li>
        <li>Isi bobot dan tipe (Benefit/Cost) untuk setiap kriteria.</li>
        <li>Hasil perhitungan ditampilkan langkah demi langkah dengan tabel dan penjelasan.</li>
        </ol>

        <h4>📈 Keterangan Metode</h4>
        <ul>
        <li><b>SAW</b> – Menghitung nilai preferensi berdasarkan penjumlahan terbobot.</li>
        <li><b>AHP</b> – Menggunakan matriks perbandingan berpasangan dan uji konsistensi (CR).</li>
        <li><b>WP</b> – Menggunakan perkalian nilai kriteria berpangkat bobot.</li>
        <li><b>TOPSIS</b> – Menentukan jarak alternatif ke solusi ideal positif dan negatif.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("⬅️ Kembali ke Halaman Utama"):
        go_to("main")

# ===== Halaman Utama =====
else:
    st.markdown("<h2 style='text-align:center; color:#3949AB;'>🧮 Aplikasi SPK: Kalkulator — Langkah per Langkah</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ===== Data Awal =====
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame({
            "Alternatif": ["A1", "A2", "A3"],
            "C1": [70, 80, 90],
            "C2": [85, 75, 95],
            "C3": [60, 65, 55]
        })

    data = st.session_state.data.copy()
    st.subheader("Input Data Alternatif & Kriteria")
    data = st.data_editor(data, num_rows="dynamic", use_container_width=True)

    col_btn = st.columns([1, 2])
    with col_btn[0]:
        if st.button("➕ Tambah Kriteria"):
            new_col = f"C{len([c for c in data.columns if c.startswith('C')]) + 1}"
            data[new_col] = 0
            st.session_state.data = data
            st.rerun()

    criteria_cols = [c for c in data.columns if c != "Alternatif"]
    selected_col = st.selectbox("Pilih kolom untuk dihapus:", criteria_cols)

    if st.button("🗑️ Hapus Kolom Terpilih"):
        if selected_col:
            data.drop(columns=selected_col, inplace=True)
            st.session_state.data = data
            st.rerun()

    st.session_state.data = data
    alternatives = data["Alternatif"].tolist()
    criteria = [c for c in data.columns if c != "Alternatif"]
    values = data[criteria].to_numpy(dtype=float)

    # ===== Pilih Metode =====
    method = st.selectbox("Pilih Metode Perhitungan:", ["SAW", "AHP", "WP", "TOPSIS"])

    # ===== Input Bobot & Jenis =====
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

    with st.spinner("⏳ Sedang memproses langkah demi langkah..."):
        time.sleep(1)

    # ========== METODE SAW ==========
    if method == "SAW":
        st.header("🌟 Metode SAW (Simple Additive Weighting)")
        norm = np.zeros_like(values)
        for j in range(len(criteria)):
            if types[j] == "Benefit":
                norm[:, j] = values[:, j] / values[:, j].max()
            else:
                norm[:, j] = values[:, j].min() / values[:, j]
        with st.expander("📘 Langkah 1: Normalisasi Matriks"):
            st.dataframe(pd.DataFrame(norm, columns=criteria, index=alternatives), use_container_width=True)
        scores = norm.dot(weights)
        with st.expander("📗 Langkah 2: Nilai Preferensi"):
            st.dataframe(pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4)}), use_container_width=True)
        rank = scores.argsort()[::-1].argsort() + 1
        result = pd.DataFrame({"Alternatif": alternatives, "Skor": scores.round(4), "Ranking": rank}).sort_values("Ranking")
        with st.expander("📙 Langkah 3: Hasil Akhir & Ranking", expanded=True):
            st.dataframe(result, use_container_width=True)

    # ========== METODE AHP ==========
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

        with st.expander("📘 Langkah 1: Matriks Perbandingan"):
            st.dataframe(pd.DataFrame(pairwise, columns=criteria, index=criteria), use_container_width=True)
        with st.expander("📗 Langkah 2: Normalisasi & Bobot"):
            st.dataframe(pd.DataFrame(A_norm.round(4), columns=criteria, index=criteria), use_container_width=True)
            st.write("Bobot:", [round(x, 4) for x in w])
        with st.expander("📙 Langkah 3: Uji Konsistensi", expanded=True):
            st.write(f"λmax = {lamda_max:.4f}, CI = {CI:.4f}, CR = {CR:.4f}")
            st.progress(1.0 - min(CR / 0.2, 1.0))
            if CR <= 0.1:
                st.success("✅ Matriks konsisten")
            else:
                st.warning("⚠️ Matriks tidak konsisten")
        norm_alt = values / values.max(axis=0)
        final_scores = norm_alt.dot(w)
        rank = final_scores.argsort()[::-1].argsort() + 1
        result = pd.DataFrame({"Alternatif": alternatives, "Skor Akhir": final_scores.round(4), "Ranking": rank}).sort_values("Ranking")
        with st.expander("📒 Langkah 4: Hasil Akhir"):
            st.dataframe(result, use_container_width=True)

    # ========== METODE WP ==========
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
        V = S / S.sum()
        rank = V.argsort()[::-1].argsort() + 1
        result = pd.DataFrame({"Alternatif": alternatives, "Skor": V.round(4), "Ranking": rank}).sort_values("Ranking")
        with st.expander("📙 Langkah 2: Hasil Akhir & Ranking", expanded=True):
            st.dataframe(result, use_container_width=True)

    # ========== METODE TOPSIS ==========
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
        with st.expander("📙 Langkah 3: Solusi Ideal"):
            ideal_df = pd.DataFrame({
                "Kriteria": criteria,
                "Ideal Positif (+)": np.round(ideal_pos, 4),
                "Ideal Negatif (−)": np.round(ideal_neg, 4)
            })
            st.dataframe(ideal_df, use_container_width=True)
        D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
        D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
        scores = D_neg / (D_pos + D_neg)
        rank = scores.argsort()[::-1].argsort() + 1
        kategori = ["🟩 Sangat Baik" if s >= 0.75 else "🟦 Baik" if s >= 0.5 else "🟨 Cukup" if s >= 0.25 else "🟥 Kurang" for s in scores]
        result = pd.DataFrame({
            "Alternatif": alternatives,
            "Skor": scores.round(4),
            "Kategori": kategori,
            "Ranking": rank
        }).sort_values("Ranking")
        with st.expander("📔 Langkah 4: Hasil Akhir & Kategori", expanded=True):
            st.dataframe(result, use_container_width=True)

# ===== Footer =====
st.markdown(
    "<hr><center><p style='font-size:14px;'>© 2025 Ayumi Fathiyaraisha | Aplikasi SPK: Kalkulator (SAW, AHP, WP, TOPSIS)</p></center>",
    unsafe_allow_html=True
)
