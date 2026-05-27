import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

# =========================
# WEBOLDAL BEÁLLÍTÁSAI
# =========================
st.set_page_config(page_title="Rezsi Kalkulátor - TDK", layout="wide")
st.title("Lakossági Rezsi Kalkulátor Szimuláció")
st.write("A bemeneti adatok alapján a látogatók különböző tarifarendszereket alkothatnak.")

# =========================
# ADATOK BETÖLTÉSE
# =========================
YEARS = range(2015, 2026)
SHIFTS = [0, .1, .2, .3]

@st.cache_data
def load_data():
    piac = np.array(pd.read_excel("piac.xlsx"))
    euro = np.array(pd.read_excel("euro.xlsx"))
    eon_excel = np.array(pd.read_excel("eon.xlsx"))
    return piac, euro, eon_excel

piac, euro, eon_excel = load_data()

# ==========================================================
# ADATGENERÁLÁS (Gauss Keverékmodell)
# ==========================================================
@st.cache_data
def generate_synthetic_data():
    n_samples = 10000 
    sigma_components = 900
    components = [
        (3800,  0.26), (5600,  0.28), (7400,  0.21), (9200,  0.06),
        (11000, 0.06), (12800, 0.02), (14600, 0.06), (16400, 0.02)
    ]
    raw_data = []
    for mu, weight in components:
        count = int(n_samples * weight)
        raw_data.extend(np.random.normal(mu, sigma_components, count))
    
    res = np.array(raw_data)
    return np.clip(res, 2800, 20000)

consumer_loads_synthetic = generate_synthetic_data()

# ==========================================================
# SZÁMÍTÁSI MOTOR (Kiterjesztve az órás adatok visszaadásával)
# ==========================================================
def calculate_all_scenarios(tou_mag, red_count, white_count, tou_blocks, return_hourly=False, plot_year=2017, shift_ratio=0.3, skala=3000):
    results_list = []
    
    # Mentjük a plot_year-hez szükséges órás tömböket, ha azt is kéri a kód
    hourly_data = {}

    for year in YEARS:
        y_idx = list(YEARS).index(year) + 1
        price = piac[:, y_idx] * euro[:, y_idx] / 1000
        load_unit = eon_excel[:, y_idx] / 1000
        n_hours = len(price)
        n_days = n_hours // 24
        
        qlen = n_hours // 4
        base_prices = np.zeros(n_hours)
        for q in range(4):
            q_slice = slice(q*qlen, (q+1)*qlen)
            prev_q_slice = slice(max(0, (q-1)*qlen), q*qlen)
            base = np.nanmedian(price[prev_q_slice]) if q > 0 else np.nanmedian(price[q_slice])
            base_prices[q_slice] = base

        d_price = price[:n_days*24].reshape(n_days, 24)
        d_mean = np.nanmean(d_price, axis=1)
        dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
        is_winter, is_weekday, is_sunday = (dates.month >= 11) | (dates.month <= 3), dates.weekday < 5, dates.weekday == 6
        
        red_cand = np.where(is_winter & is_weekday)[0]
        red_days = red_cand[np.argsort(d_mean[red_cand])[-int(red_count):]]
        rem_idx = np.setdiff1d(np.where(~is_sunday)[0], red_days)
        white_days = rem_idx[np.argsort(d_mean[rem_idx])[-int(white_count):]]

        edf_tariff = base_prices.copy().reshape(n_days, 24)
        for d in red_days: edf_tariff[d, np.argsort(d_price[d])[-6:]] *= 3.5
        for d in white_days: edf_tariff[d, np.argsort(d_price[d])[-8:]] *= 1.5
        edf_tariff = edf_tariff.flatten()

        tou_tariff = base_prices.copy().reshape(n_days, 24)
        ranges = [(0,8), (8,16), (16,24)]
        for i, (s, e) in enumerate(ranges):
            if tou_blocks[i]: tou_tariff[:, s:e] *= tou_mag
            else: tou_tariff[:, s:e] /= tou_mag
        tou_tariff = tou_tariff.flatten()

        # Fogyasztáseltolási belső függvény az órás logikához
        def shift_consumption(load_vec, t_vec, s_rat):
            l_mat = load_vec[:n_days*24].reshape(n_days, 24).copy()
            t_mat = t_vec[:n_days*24].reshape(n_days, 24)
            expensive_mask = t_mat > 1.01 * np.nanpercentile(t_mat, 20, axis=1, keepdims=True)
            removed = np.sum(l_mat * expensive_mask * s_rat, axis=1)
            l_mat[expensive_mask] *= (1 - s_rat)
            l_mat += (~expensive_mask * (removed / np.where(np.sum(~expensive_mask, axis=1)>0, np.sum(~expensive_mask, axis=1), 1))[:, None])
            return l_mat.flatten()

        # Ha csak a heti/órás adatokat kérjük egy konkrét évre (gyorsításképp)
        if return_hourly and year == plot_year:
            eon_scaled = (eon_excel[:, y_idx] * skala / 1000)
            hourly_data['piac'] = price
            hourly_data['edf_tarifa'] = edf_tariff
            hourly_data['tou_tarifa'] = tou_tariff
            hourly_data['eon_orig'] = eon_scaled
            hourly_data['eon_shifted_edf'] = shift_consumption(eon_scaled, edf_tariff, shift_ratio)
            hourly_data['eon_shifted_tou'] = shift_consumption(eon_scaled, tou_tariff, shift_ratio)

        for s_ratio in SHIFTS:
            def get_cost(t_vec):
                l_mat = load_unit[:n_days*24].reshape(n_days, 24).copy()
                t_mat = t_vec[:n_days*24].reshape(n_days, 24)
                expensive_mask = t_mat > 1.01 * np.nanpercentile(t_mat, 20, axis=1, keepdims=True)
                removed = np.sum(l_mat * expensive_mask * s_ratio, axis=1)
                l_mat[expensive_mask] *= (1 - s_ratio)
                l_mat += (~expensive_mask * (removed / np.where(np.sum(~expensive_mask, axis=1)>0, np.sum(~expensive_mask, axis=1), 1))[:, None])
                return np.nansum(l_mat.flatten() * t_vec)

            results_list.append({
                'Év': year, 'Rugalmasság': f"{int(s_ratio*100)}%", 
                'EDF_UC': get_cost(edf_tariff), 'TOU_UC': get_cost(tou_tariff)
            })
            
    if return_hourly:
        return pd.DataFrame(results_list), hourly_data
    return pd.DataFrame(results_list)

# ==========================================================
# STREAMLIT OLDALSÁV (SIDEBAR) - GLOBÁLIS BEÁLLÍTÁSOK
# ==========================================================
st.sidebar.header("⚙️ Globális Tarifa Beállítások")

txt_tou_mag = st.sidebar.number_input("TOU szorzó (magasság)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
txt_red = st.sidebar.number_input("Piros napok száma (EDF)", min_value=0, max_value=150, value=22)
txt_white = st.sidebar.number_input("Fehér napok száma (EDF)", min_value=0, max_value=150, value=43)

st.sidebar.subheader("TOU Időblokkok aktív szorzója")
ch08 = st.sidebar.checkbox("0-8h", value=False)
ch816 = st.sidebar.checkbox("8-16h", value=False)
ch1624 = st.sidebar.checkbox("16-24h", value=True)
tou_blocks_status = [ch08, ch816, ch1624]

# Minden fülhöz szükséges alap futtatás (Éves szintű eredmények)
df_unit = calculate_all_scenarios(txt_tou_mag, txt_red, txt_white, tou_blocks_status)

# ==========================================================
# FÜLEK (TABS) LÉTREHOZÁSA
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Éves Boxplotok", 
    "⏱️ Órás Heti Profil (168 óra)", 
    "📈 Fogyasztási Eloszlás (KDE)", 
    "💾 Excel Export"
])

# ----------------------------------------------------------
# 1. FÜL: ÉVES BOXPLOT DIAGRAMOK
# ----------------------------------------------------------
with tab1:
    st.subheader("Szimulációs Eredmények (Boxplot diagramok)")
    
    final_data = []
    for _, row in df_unit.iterrows():
        edf_part = pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['EDF_UC'], 'Modell': 'EDF'})
        tou_part = pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['TOU_UC'], 'Modell': 'TOU'})
        final_data.extend([edf_part, tou_part])
    df_plot = pd.concat(final_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, mod, title, pal in zip([ax1, ax2], ['EDF', 'TOU'], ['CPP (EDF Tempo jellegű)', 'TOU Rendszer'], ['viridis', 'magma']):
        sns.boxplot(x='Év', y='Számla', hue='Rugalmasság', data=df_plot[df_plot['Modell'] == mod], 
                    palette=pal, showfliers=False, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel("Éves számla [Ft]")
        ax.legend(title="Rugalmasság", loc='upper left', fontsize='x-small', ncol=4)
        ax.grid(axis='y', alpha=0.15)
        ax.axhline(y=36*np.median(consumer_loads_synthetic), color='black', ls='--', alpha=0.3)
        ax.set_ylim(0, 1e6)

    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------------
# 2. FÜL: ÓRÁS HETI PROFIL (Kért funkció beépítése)
# ----------------------------------------------------------
with tab2:
    st.subheader("Órás szintű heti profil vizualizáció (168 óra)")
    st.write("Vizsgáld meg tetszőlegesen kiválasztott órában a tarifák és a fogyasztás áthelyeződésének dinamikáját.")
    
    # Lokális vezérlők ehhez a fülhöz
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_year = st.selectbox("Szimulált év kiválasztása", options=list(YEARS), index=2) # Alapértelmezett: 2017
    with col2:
        skala_input = st.number_input("Skála (Fogyasztás kWh/év):", min_value=1000, max_value=10000, value=3000, step=500)
    with col3:
        shift_input = st.slider("Shift ratio (Rugalmasság 0-1):", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    # DÁTUM ÉS ÓRA VÁLASZTÓ (A csúszka helyett)
    st.write("**Kezdő időpont kiválasztása:**")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        # A naptár minimum és maximum dátuma igazodik a kiválasztott évhez
        selected_date = st.date_input(
            "Válaszd ki a napot:",
            value=pd.to_datetime(f"{selected_year}-10-15").date(),
            min_value=pd.to_datetime(f"{selected_year}-01-01").date(),
            max_value=pd.to_datetime(f"{selected_year}-12-25").date() # dec 25 a max, hogy beleférjen még 176 óra az évbe
        )
    with d_col2:
        selected_hour = st.number_input("Kezdő óra (0-23):", min_value=0, max_value=23, value=12, step=1)

    # Kiszámoljuk, hogy a kiválasztott dátum és óra az év hányadik órája (0-8760 között)
    base_date = pd.to_datetime(f"{selected_year}-01-01")
    target_date = pd.to_datetime(f"{selected_date} {selected_hour}:00:00")
    
    # A timedelta segítségével megkapjuk az eltelt órák számát
    start_hour = int((target_date - base_date).total_seconds() // 3600)
    end_hour = start_hour + 168
    
# Láthatósági beállítások Streamlit checkboxokkal (Szétbontva a módosított fogyasztások)
    st.write("**Görbék ki-be kapcsolása:**")
    c_col1, c_col2, c_col3, c_col4, c_col5, c_col6 = st.columns(6)
    show_piac = c_col1.checkbox("Piaci ár", value=True)
    show_cpp = c_col2.checkbox("CPP (EDF) ár", value=True)
    show_tou = c_col3.checkbox("TOU ár", value=True)
    show_orig_load = c_col4.checkbox("Eredeti fogyasztás", value=True)
    show_shifted_edf = c_col5.checkbox("Módosított CPP (EDF)", value=True)  # Új külön gomb
    show_shifted_tou = c_col6.checkbox("Módosított TOU", value=True)      # Új külön gomb

    # Számítás elvégzése az órás adatok kinyerésével
    _, hourly = calculate_all_scenarios(
        txt_tou_mag, txt_red, txt_white, tou_blocks_status, 
        return_hourly=True, plot_year=selected_year, shift_ratio=shift_input, skala=skala_input
    )

    # Szeletek kivágása (1 hét = 168 óra)
    v_slice = hourly['piac'][start_hour:end_hour]
    cpp_slice = hourly['edf_tarifa'][start_hour:end_hour]
    tou_slice = hourly['tou_tarifa'][start_hour:end_hour]
    eon_orig_slice = hourly['eon_orig'][start_hour:end_hour]
    eon_shift_edf_slice = hourly['eon_shifted_edf'][start_hour:end_hour]
    eon_shift_tou_slice = hourly['eon_shifted_tou'][start_hour:end_hour]

    # Grafikon felépítése
    fig3, ax_p = plt.subplots(figsize=(14, 6))
    ax_l = ax_p.twinx()

    # Árak kirajzolása (Bal tengely)
    if show_piac:
        ax_p.plot(v_slice, label="Piaci ár", color="blue", alpha=0.3)
    if show_cpp:
        ax_p.plot(cpp_slice, label="CPP ár (EDF)", color="orange", linewidth=2)
    if show_tou:
        ax_p.plot(tou_slice, label="TOU ár", color="purple", linewidth=2, linestyle=":")
    ax_p.set_ylabel("Áramár [Ft/kWh]", color="blue")
    ax_p.tick_params(axis='y', labelcolor="blue")

    # Terhelések kirajzolása (Jobb tengely - most már egyesével kapcsolható)
    if show_orig_load:
        ax_l.plot(eon_orig_slice, label="Eredeti fogyasztás", color="green", linestyle="--", alpha=0.7)
    if show_shifted_edf:
        ax_l.plot(eon_shift_edf_slice, label="Módosított fogyasztás (CPP)", color="red", linewidth=1.5)
    if show_shifted_tou:
        ax_l.plot(eon_shift_tou_slice, label="Módosított fogyasztás (TOU)", color="crimson", linestyle="-.", linewidth=1.5)
    ax_l.set_ylabel("Fogyasztás [kWh]", color="green")
    ax_l.tick_params(axis='y', labelcolor="green")

    # Közös formázás
    ax_p.set_title(f"Tarifarendszerek és lakossági reakciók összehasonlítása ({selected_year}, {selected_date} {selected_hour}:00-tól, 1 hét)", fontweight='bold')
    ax_p.set_xlim(0, 168)
    ax_p.set_xticks(range(0, 169, 24))
    ax_p.set_xlabel("Eltelt órák a héten")
    ax_p.grid(True, alpha=0.2)
    
    # Össvont legend kezelés
    lines_p, labels_p = ax_p.get_legend_handles_labels()
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    ax_p.legend(lines_p + lines_l, labels_p + labels_l, loc="upper left")

    st.pyplot(fig3)

# ----------------------------------------------------------
# 3. FÜL: KDE DIAGRAM
# ----------------------------------------------------------
with tab3:
    st.subheader("Fogyasztási Eloszlás Rekonstrukciója (KDE)")
    
    fig2, ax_kde = plt.subplots(figsize=(10, 4))
    sns.histplot(consumer_loads_synthetic, bins=8, color='#88c488', edgecolor='black', stat="density", alpha=0.9, ax=ax_kde)
    sns.kdeplot(consumer_loads_synthetic, color='green', linewidth=3, bw_adjust=1.5, ax=ax_kde)
    ax_kde.set_xlim(2500, 18000)
    ax_kde.set_xlabel("Éves fogyasztás [kWh]")
    ax_kde.set_ylabel("Sűrűség (Részesedés)")
    
    plt.tight_layout()
    st.pyplot(fig2)

# ----------------------------------------------------------
# 4. FÜL: EXCEL GENERÁLÁS ÉS ADATOK
# ----------------------------------------------------------
with tab4:
    st.subheader("Adatok Exportálása TDK dolgozathoz")

    df_unit['Rugalmasság_num'] = df_unit['Rugalmasság'].str.replace('%', '').astype(int)
    stats_data = []
    for _, row in df_unit.iterrows():
        bills_edf = consumer_loads_synthetic * row['EDF_UC']
        bills_tou = consumer_loads_synthetic * row['TOU_UC']
        stats_data.append({
            'Rendszer': 'EDF', 'Év': row['Év'], 'Rugalmasság_érték': row['Rugalmasság_num'], 'Rugalmasság (%)': row['Rugalmasság'],
            'Alsó kvartilis (Q1) [Ft]': np.percentile(bills_edf, 25), 'Medián [Ft]': np.percentile(bills_edf, 50),
            'Felső kvartilis (Q3) [Ft]': np.percentile(bills_edf, 75), 'Átlag [Ft]': np.mean(bills_edf)
        })
        stats_data.append({
            'Rendszer': 'TOU', 'Év': row['Év'], 'Rugalmasság_érték': row['Rugalmasság_num'], 'Rugalmasság (%)': row['Rugalmasság'],
            'Alsó kvartilis (Q1) [Ft]': np.percentile(bills_tou, 25), 'Medián [Ft]': np.percentile(bills_tou, 50),
            'Felső kvartilis (Q3) [Ft]': np.percentile(bills_tou, 75), 'Átlag [Ft]': np.mean(bills_tou)
        })
    df_final_stats = pd.DataFrame(stats_data)

    median_list = []
    fix_median_val = 36 * np.median(consumer_loads_synthetic)
    for year in YEARS:
        year_row = {'Év': year, 'Fix_36Ft_Median': fix_median_val}
        year_data = df_unit[df_unit['Év'] == year]
        for _, row in year_data.iterrows():
            r_label = row['Rugalmasság']
            year_row[f'EDF_{r_label}'] = np.median(consumer_loads_synthetic * row['EDF_UC'])
            year_row[f'TOU_{r_label}'] = np.median(consumer_loads_synthetic * row['TOU_UC'])
        median_list.append(year_row)
    df_medians_summary = pd.DataFrame(median_list)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_final_stats[df_final_stats['Rendszer'] == 'EDF'].to_excel(writer, sheet_name='EDF_Reszletes', index=False)
        df_final_stats[df_final_stats['Rendszer'] == 'TOU'].to_excel(writer, sheet_name='TOU_Reszletes', index=False)
        df_medians_summary.to_excel(writer, sheet_name='Osszesitett_Medianok', index=False)

    st.write("### Összesitett Mediánok Előnézete:")
    st.dataframe(df_medians_summary.head(5))

    st.download_button(
        label="📊 Teljes Excel Statisztika Letöltése",
        data=buffer.getvalue(),
        file_name="Energia_Szamla_Statisztika.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
