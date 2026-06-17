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
st.title("Lakossági Rezsi Kalkulátor & Piaci Elemzés")
st.write("Dinamikus tarifarendszerek és lakossági fogyasztói válaszok szimulációja.")

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
# SZÁMÍTÁSI MOTOR
# ==========================================================
def calculate_all_scenarios(tou_mag, red_count, white_count, tou_blocks, return_hourly=False, plot_year=2017, shift_ratio=0.3, skala=3000):
    results_list = []
    hourly_data = {}

    for year in YEARS:
        y_idx = list(YEARS).index(year) + 1
        price = piac[:, y_idx].astype(float) * euro[:, y_idx].astype(float) / 1000
        price_eur = piac[:, y_idx].astype(float)
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

        def shift_consumption(load_vec, t_vec, s_rat):
            l_mat = load_vec[:n_days*24].reshape(n_days, 24).copy()
            t_mat = t_vec[:n_days*24].reshape(n_days, 24)
            expensive_mask = t_mat > 1.01 * np.nanpercentile(t_mat, 20, axis=1, keepdims=True)
            removed = np.sum(l_mat * expensive_mask * s_rat, axis=1)
            l_mat[expensive_mask] *= (1 - s_rat)
            l_mat += (~expensive_mask * (removed / np.where(np.sum(~expensive_mask, axis=1)>0, np.sum(~expensive_mask, axis=1), 1))[:, None])
            return l_mat.flatten()

        if return_hourly and year == plot_year:
            eon_scaled = (eon_excel[:, y_idx] * skala / 1000)
            hourly_data['piac'] = price
            hourly_data['piac_eur'] = price_eur
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
            results_list.append({'Év': year, 'Rugalmasság': f"{int(s_ratio*100)}%", 'EDF_UC': get_cost(edf_tariff), 'TOU_UC': get_cost(tou_tariff)})
            
    if return_hourly: return pd.DataFrame(results_list), hourly_data
    return pd.DataFrame(results_list)

@st.cache_data
def generate_synthetic_data():
    res = []
    components = [(3800, 0.26), (5600, 0.28), (7400, 0.21), (9200, 0.06), (11000, 0.06), (12800, 0.02), (14600, 0.06), (16400, 0.02)]
    for mu, weight in components: res.extend(np.random.normal(mu, 900, int(10000 * weight)))
    return np.clip(np.array(res), 2800, 20000)

consumer_loads_synthetic = generate_synthetic_data()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Globális Beállítások")
txt_tou_mag = st.sidebar.number_input("Időszakos árszabályozás szorzója", 1.0, 5.0, 1.5, 0.1)
txt_red = st.sidebar.number_input("Piros napok száma", 0, 150, 22)
txt_white = st.sidebar.number_input("Fehér napok száma", 0, 150, 43)
tou_blocks_status = [st.sidebar.checkbox("0-8h", False), st.sidebar.checkbox("8-16h", False), st.sidebar.checkbox("16-24h", True)]

df_unit = calculate_all_scenarios(txt_tou_mag, txt_red, txt_white, tou_blocks_status)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Éves Boxplotok", "Heti Profil", "Piaci Áringadozás", "Export & Háttér"])

# --- TAB 1 ---
with tab1:
    st.subheader("A dinamikus tarifarendszerek pénzügyi hatása a fogyasztókra")
    final_data = []
    for _, row in df_unit.iterrows():
        final_data.extend([pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['EDF_UC'], 'Modell': 'EDF'}),
                          pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['TOU_UC'], 'Modell': 'TOU'})])
    df_plot = pd.concat(final_data)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, mod, title, pal in zip([ax1, ax2], ['EDF', 'TOU'], ['Kritikus csúcs (EDF Tempo)', 'Időszakos árszabályozás'], ['viridis', 'magma']):
        sns.boxplot(x='Év', y='Számla', hue='Rugalmasság', data=df_plot[df_plot['Modell'] == mod], palette=pal, showfliers=False, ax=ax)
        ax.axhline(y=36*np.median(consumer_loads_synthetic), color='black', ls='--', alpha=0.3)
        ax.set_title(title, fontweight='bold')
    st.pyplot(fig1)

    st.info("""A grafikonon összehasonlíthatjuk a két vizsgált tarifarandszer hatását egy általunk kiválasztott fogyasztóra. Célunk egy olyan rendszer kialakítása, 
    amivel nem kényszerítjük a lakosságot fogyasztásuk módosítására, hanem lehetőséget adunk nekik pénzügyi megtakarítás elérésére. Ahhoz, hogy ezt a célunkat 
    szemelőtt tartsuk, a grafikonokon feltüntettük a fix árszabáshoz tartozó éves villanyszámlát, egy átlagosnak mondható (3000 kWh/év) fogyaszó esetére (ez a 
    szaggatott vonal). A grafikonok elkészítésekor a Háttér oldalon látható fogyaszói eloszlást feltételeztünk, a számításokat pedig egy 10000 fős populációra 
    végeztük el.
    A számításokat négy rugalmassági szinten értékeltük ki (0%, 10%, 20%, 30%), és ezeket közösen ábrázoltuk. A tarifarendszernek adtunk egy kezdeti állapotot, 
    amely tetszőlegesen módosítható. """)

# --- TAB 2 ---
with tab2:
    st.subheader("Órás bontású heti profil")
    
    # 1. SZINT: Bemeneti paraméterek (Minden, ami nehéz számítást igényel)
    col1, col2, col3 = st.columns(3)
    sel_year = col1.selectbox("Év", list(YEARS), index=2)
    skala_in = col2.number_input("Skála (kWh/év)", 1000, 10000, 3000, 500)
    shift_in = col3.slider("Rugalmasság", 0.0, 1.0, 0.3)
    
    d_col1, d_col2 = st.columns(2)
    sel_date = d_col1.date_input("Nap", pd.to_datetime(f"{sel_year}-10-15").date())
    sel_hour = d_col2.number_input("Óra", 0, 23, 12)
    
    start_h = int((pd.to_datetime(f"{sel_date} {sel_hour}:00") - pd.to_datetime(f"{sel_year}-01-01")).total_seconds() // 3600)
    sl = slice(start_h, start_h+168)
    
    # A nehéz matematikai számítást CSAK akkor futtatjuk újra, ha a fenti paraméterek változnak
    _, hourly = calculate_all_scenarios(txt_tou_mag, txt_red, txt_white, tou_blocks_status, True, sel_year, shift_in, skala_in)

    # 2. SZINT: Interaktív görbe-megjelenítés újrafutás-gátló fragmentben
    # Az ezen a dekorátoron belüli checkboxok kattintása NEM futtatja újra a fenti modellt!
    @st.fragment
    def render_interactive_plots(hourly_data, time_slice):
        c1, c2, c3, c4 = st.columns(4)
        s_p = c1.checkbox("Piaci ár mutatása", True)
        s_o = c2.checkbox("Eredeti fogyasztás mutatása", True)
        s_se = c3.checkbox("Módosított kritikus csúcs mutatása", True)
        s_st = c4.checkbox("Módosított időszakos szabályozás mutatása", True)
        
        # Szeletek előkészítése
        v_s = hourly_data['piac'][time_slice]
        cpp_s = hourly_data['edf_tarifa'][time_slice]
        tou_s = hourly_data['tou_tarifa'][time_slice]
        eon_s = hourly_data['eon_orig'][time_slice]
        eon_cpp_s = hourly_data['eon_shifted_edf'][time_slice]
        eon_tou_s = hourly_data['eon_shifted_tou'][time_slice]
        
        # Két grafikon egymás mellett
        g_col1, g_col2 = st.columns(2)
        
        # ----------------------------------------------------------
        # BAL OLDAL: CPP (EDF TEMPO) PLOT
        # ----------------------------------------------------------
        with g_col1:
            fig_cpp, ax_p1 = plt.subplots(figsize=(8, 4.5))
            ax_l1 = ax_p1.twinx()
            
            # Árak (Bal tengely)
            if s_p: 
                ax_p1.plot(v_s, label="Piaci ár", color="blue", alpha=0.25, linewidth=1)
            ax_p1.plot(cpp_s, label="Kritikus csúcs tarifa (EDF)", color="orange", linewidth=2.5)
            ax_p1.set_ylabel("Villamos energia ára [Ft/kWh]", color="orange")
            ax_p1.tick_params(axis='y', labelcolor="orange")
            
            # Fogyasztások (Jobb tengely)
            if s_o: 
                ax_l1.plot(eon_s, label="Eredeti fogyasztás", color="green", ls="--", alpha=0.25, linewidth=1)
            if s_se: 
                ax_l1.plot(eon_cpp_s, label="Módosított kritikus csúcs melletti fogy.", color="red", linewidth=2)
            ax_l1.set_ylabel("Fogyasztás [kWh]", color="red")
            ax_l1.tick_params(axis='y', labelcolor="red")
            
            # Elrendezés és Legend
            ax_p1.set_title("Kritikus csúcs rendszer profilja", fontweight='bold')
            ax_p1.set_xlim(0, 168)
            ax_p1.set_xticks(range(0, 169, 24))
            ax_p1.grid(True, alpha=0.15)
            
            h1, l1 = ax_p1.get_legend_handles_labels()
            h2, l2 = ax_l1.get_legend_handles_labels()
            ax_p1.legend(h1+h2, l1+l2, loc="upper left", fontsize='small')
            
            st.pyplot(fig_cpp)
            
        # ----------------------------------------------------------
        # JOBB OLDAL: TOU PLOT
        # ----------------------------------------------------------
        with g_col2:
            fig_tou, ax_p2 = plt.subplots(figsize=(8, 4.5))
            ax_l2 = ax_p2.twinx()
            
            # Árak (Bal tengely)
            if s_p: 
                ax_p2.plot(v_s, label="Piaci ár", color="blue", alpha=0.25, linewidth=1)
            ax_p2.plot(tou_s, label="Időszakos árszabályozás", color="purple", linewidth=2.5, ls="-")
            ax_p2.set_ylabel("Villamos energia ára [Ft/kWh]", color="purple")
            ax_p2.tick_params(axis='y', labelcolor="purple")
            
            # Fogyasztások (Jobb tengely)
            if s_o: 
                ax_l2.plot(eon_s, label="Eredeti fogyasztás", color="green", ls="--", alpha=0.25, linewidth=1)
            if s_st: 
                ax_l2.plot(eon_tou_s, label="Módosított időszakos árszabályozás melletti fogy.", color="crimson", linewidth=2)
            ax_l2.set_ylabel("Fogyasztás [kWh]", color="crimson")
            ax_l2.tick_params(axis='y', labelcolor="crimson")
            
            # Elrendezés és Legend
            ax_p2.set_title("Időszakos rendszer profilja", fontweight='bold')
            ax_p2.set_xlim(0, 168)
            ax_p2.set_xticks(range(0, 169, 24))
            ax_p2.grid(True, alpha=0.15)
            
            h1, l1 = ax_p2.get_legend_handles_labels()
            h2, l2 = ax_l2.get_legend_handles_labels()
            ax_p2.legend(h1+h2, l1+l2, loc="upper left", fontsize='small')
            
            st.pyplot(fig_tou)

    # Meghívjuk a renderelést
    render_interactive_plots(hourly, sl)

# --- TAB 3 (ÚJ: Piaci Áringadozás) ---
with tab3:
    st.subheader("Villamosenergia piaci áreloszlása óránként")
    sel_year_market = st.selectbox("Év kiválasztása az elemzéshez:", list(YEARS), index=len(YEARS)-1, key="market_year")
    
    # Adatok előkészítése a boxplot-hoz
    y_idx_m = list(YEARS).index(sel_year_market) + 1
    p_eur = piac[:, y_idx_m].astype(float)
    p_eur = p_eur[~np.isnan(p_eur)]
    hrs = np.arange(len(p_eur)) % 24
    df_market = pd.DataFrame({'Óra': hrs, 'Ár': p_eur})
    
    fig3, ax_m = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Óra', y='Ár', data=df_market, showfliers=False, ax=ax_m)
    ax_m.set_title(f"Piaci ár eloszlása óránként - {sel_year_market} [Eur/MWh]", fontsize=16, fontweight='bold')
    ax_m.set_xlabel("A nap órája (0-23)", fontsize=12)
    ax_m.set_ylabel("Ár [Eur/MWh]", fontsize=12)
    ax_m.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig3)

# --- TAB 4 ---
with tab4:
    st.subheader("Háttérstatisztika és Export")
    fig4, ax_kde = plt.subplots(figsize=(10, 4))
    sns.histplot(consumer_loads_synthetic, bins=8, color='#88c488', stat="density", ax=ax_kde)
    sns.kdeplot(consumer_loads_synthetic, color='green', linewidth=3, bw_adjust=1.5, ax=ax_kde)
    st.pyplot(fig4)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_unit.to_excel(writer, sheet_name='Eredmenyek', index=False)
    st.download_button("📊 Excel Letöltése", buffer.getvalue(), "statisztika.xlsx")

# ==========================================================
# LÁBLÉC (FOOTER)
# ==========================================================
st.markdown("---") # Elválasztó vonal

# 1. SOR: LOGÓK (Nagyobb méretben, középre igazítva)
col_logok1, col_logok2 = st.columns([1, 1])

with col_logok1:
    st.image("BME_logo_transparent_0.png", width=200) # Itt állítsd be a kívánt méretet
with col_logok2:
    st.image("bme_nti_logo_white-250x125.png", width=200)

# 2. SOR: COPYRIGHT ÉS INFORMÁCIÓK (Kisebb betűmérettel)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85em; color: #666; margin-top: 20px;">
    <strong>BME Nukleáris Technikai Intézet (NTI)</strong> | Dinamikus Lakossági Tarifarendszerek Kutatása<br>
    © 2026 Szauter Benedek. Minden jog fenntartva.<br>
    <em>Ez a dashboard a BME NTI keretein belül végzett TDK kutatás részeként készült.</em>
    </div>
    """, 
    unsafe_allow_html=True
)
