import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import TextBox, Button, CheckButtons
from scipy.stats import lognorm
from scipy.stats import gaussian_kde

# =========================
# ADATOK BETÖLTÉSE
# =========================
YEARS = range(2015, 2026)
SHIFTS = [0, .1, .2, .3]

# Itt töltsd be a saját fájljaidat!
piac = np.array(pd.read_excel("piac.xlsx"))
euro = np.array(pd.read_excel("euro.xlsx"))
eon_excel = np.array(pd.read_excel("eon.xlsx"))

# ==========================================================
# 1. ADATGENERÁLÁS A KÉP ALAPJÁN (Gauss Keverékmodell)
# ==========================================================
# A hisztogram oszlopainak helyére helyezünk el Gauss-görbéket a megfelelő arányban.

# Paraméterek a kép alapján történő "reverse engineering" finomhangolással:
n_samples = 10000  # Szimulált populáció mérete
sigma_components = 900 # A kicsi Gauss-komponensek szélessége

# Komponensek definiálása: (Átlag, Arány/Súly a populációban)
# Az arányokat szemre becsültük meg a hisztogram oszlopok magassága alapján.
components = [
    (3800,  0.26), # 1. oszlop (3000-4800): magas
    (5600,  0.28), # 2. oszlop (4800-6600): csúcs
    (7400,  0.21), # 3. oszlop (6600-8400): kicsit alacsonyabb
    (9200,  0.06), # 4. oszlop (8400-10200): hirtelen esés
    (11000, 0.06), # 5. oszlop (10200-12000): tartós alacsony szint
    (12800, 0.02), # 6. oszlop (12000-13800): mélypont
    (14600, 0.06), # 7. oszlop (13800-15600): második kisebb púp
    (16400, 0.02), # 8. oszlop (15600-17400): a farok vége
]

# Nyers adat generálása: össze fűzzük a megfelelő arányú mintákat
raw_data = []
for mu, weight in components:
    count = int(n_samples * weight)
    # Normál (Gauss) eloszlást mintázunk az adott oszlop közepére
    raw_data.extend(np.random.normal(mu, sigma_components, count))

consumer_loads_synthetic = np.array(raw_data)

# Levágjuk a fizikai képtelenségeket a kép alsó korlátja alapján
consumer_loads_synthetic = np.clip(consumer_loads_synthetic, 2800, 20000)

# =========================
# SZÁMÍTÁSI MOTOR
# =========================

def calculate_all_scenarios(tou_mag, red_count, white_count, tou_blocks):
    results_list = []
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
            
    return pd.DataFrame(results_list)

# =========================
# VIZUALIZÁCIÓ
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
plt.subplots_adjust(left=0.18, bottom=0.1, hspace=0.25, right=0.95, top=0.92)

def update(event):
    try:
        t_mag, r_cnt, w_cnt = float(txt_tou_mag.text), int(txt_red.text), int(txt_white.text)
    except: return

    df_unit = calculate_all_scenarios(t_mag, r_cnt, w_cnt, check_tou.get_status())
    
    final_data = []
    for _, row in df_unit.iterrows():
        # Itt szorozzuk fel a valósághű populációval
        edf_part = pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['EDF_UC'], 'Modell': 'EDF'})
        tou_part = pd.DataFrame({'Év': row['Év'], 'Rugalmasság': row['Rugalmasság'], 'Számla': consumer_loads_synthetic * row['TOU_UC'], 'Modell': 'TOU'})
        final_data.extend([edf_part, tou_part])
    
    df_plot = pd.concat(final_data)

    for ax, mod, title, pal in zip([ax1, ax2], ['EDF', 'TOU'], ['CPP (EDF Tempo)', 'TOU'], ['viridis', 'magma']):
        ax.clear()
        sns.boxplot(x='Év', y='Számla', hue='Rugalmasság', data=df_plot[df_plot['Modell'] == mod], 
                    palette=pal, showfliers=False, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel("Éves számla [Ft]")
        ax.legend(title="Rugalmasság", loc='upper left', fontsize='x-small', ncol=4)
        ax.grid(axis='y', alpha=0.15)
        # Bázisvonal a medián fogyasztóval (kb. 5500 kWh)
        ax.axhline(y=36*np.median(consumer_loads_synthetic), color='black', ls='--', alpha=0.3, label="Medián fix ár")
        ax.set_ylim(0, 1e6)

    plt.draw()

    # ==========================================================
    # STATISZTIKAI ADATOK EXPORTÁLÁSA EXCELBE
    # ==========================================================

    # 1. Statisztikai alapok kiszámítása
    stats_data = []

    # Feltételezzük, hogy a 'df_unit' a calculate_all_scenarios() eredménye
    # A rugalmasságot számmá alakítjuk a helyes sorrendbe rendezéshez
    df_unit['Rugalmasság_num'] = df_unit['Rugalmasság'].str.replace('%', '').astype(int)

    for _, row in df_unit.iterrows():
        # Számlák kiszámítása a teljes populációra az adott egységköltséggel
        bills_edf = consumer_loads_synthetic * row['EDF_UC']
        bills_tou = consumer_loads_synthetic * row['TOU_UC']

        # EDF adatok gyűjtése
        stats_data.append({
            'Rendszer': 'EDF',
            'Év': row['Év'],
            'Rugalmasság_érték': row['Rugalmasság_num'],
            'Rugalmasság (%)': row['Rugalmasság'],
            'Alsó kvartilis (Q1) [Ft]': np.percentile(bills_edf, 25),
            'Medián [Ft]': np.percentile(bills_edf, 50),
            'Felső kvartilis (Q3) [Ft]': np.percentile(bills_edf, 75),
            'Átlag [Ft]': np.mean(bills_edf)
        })

        # TOU adatok gyűjtése
        stats_data.append({
            'Rendszer': 'TOU',
            'Év': row['Év'],
            'Rugalmasság_érték': row['Rugalmasság_num'],
            'Rugalmasság (%)': row['Rugalmasság'],
            'Alsó kvartilis (Q1) [Ft]': np.percentile(bills_tou, 25),
            'Medián [Ft]': np.percentile(bills_tou, 50),
            'Felső kvartilis (Q3) [Ft]': np.percentile(bills_tou, 75),
            'Átlag [Ft]': np.mean(bills_tou)
        })

    df_final_stats = pd.DataFrame(stats_data)

    # 2. Mentés Excelbe külön munkalapokra
    with pd.ExcelWriter("Energia_Szamla_Statisztika.xlsx", engine='openpyxl') as writer:
    
        # --- EDF fül ---
        edf_export = df_final_stats[df_final_stats['Rendszer'] == 'EDF'].copy()
        # Rendezés: Év (növekvő), majd Rugalmasság (növekvő)
        edf_export = edf_export.sort_values(by=['Év', 'Rugalmasság_érték'])
        # Felesleges segédoszlopok eldobása
        edf_export = edf_export.drop(columns=['Rendszer', 'Rugalmasság_érték'])
        edf_export.to_excel(writer, sheet_name='EDF_Tempo', index=False)
    
        # --- TOU fül ---
        tou_export = df_final_stats[df_final_stats['Rendszer'] == 'TOU'].copy()
        tou_export = tou_export.sort_values(by=['Év', 'Rugalmasság_érték'])
        tou_export = tou_export.drop(columns=['Rendszer', 'Rugalmasság_érték'])
        tou_export.to_excel(writer, sheet_name='TOU_Rendszer', index=False)

        # ==========================================================
    # KIEGÉSZÍTÉS: CSAK A MEDIÁNOK ÖSSZESÍTÉSE (SZÉLES FORMÁTUM)
    # ==========================================================
    
    median_list = []
    # A fix ár mediánja (mivel a 36 Ft állandó, ez a fogyasztói eloszlás mediánjától függ)
    fix_median_val = 36 * np.median(consumer_loads_synthetic)

    for year in YEARS:
        # Alap adatsor az adott évre
        year_row = {
            'Év': year, 
            'Fix_36Ft_Median': fix_median_val
        }
        
        # Kiválogatjuk az adott évhez tartozó eredményeket a df_unit-ból
        year_data = df_unit[df_unit['Év'] == year]
        
        for _, row in year_data.iterrows():
            r_label = row['Rugalmasság']
            # Kiszámoljuk a populáció mediánját az adott egységköltséggel
            year_row[f'EDF_{r_label}'] = np.median(consumer_loads_synthetic * row['EDF_UC'])
            year_row[f'TOU_{r_label}'] = np.median(consumer_loads_synthetic * row['TOU_UC'])
            
        median_list.append(year_row)

    df_medians_summary = pd.DataFrame(median_list)

    # =========================
    # MENTÉS AZ EXCEL FÁJLBA
    # =========================
    with pd.ExcelWriter("Energia_Szamla_Statisztika.xlsx", engine='openpyxl') as writer:
        # Az eredeti részletes adatok (ha kellenek még)
        df_final_stats[df_final_stats['Rendszer'] == 'EDF'].to_excel(writer, sheet_name='EDF_Reszletes', index=False)
        df_final_stats[df_final_stats['Rendszer'] == 'TOU'].to_excel(writer, sheet_name='TOU_Reszletes', index=False)
        
        # AZ ÚJ ÖSSZEFOGLALÓ TÁBLA
        # Ez az, amit közvetlenül be tudsz másolni a TDK dolgozatodba táblázatként
        df_medians_summary.to_excel(writer, sheet_name='Osszesitett_Medianok', index=False)

    print("A mediánok kigyűjtése sikeresen megtörtént az 'Osszesitett_Medianok' munkalapra!")


    

# --- GUI ---
ax_tou_mag = plt.axes([0.05, 0.75, 0.06, 0.03])
txt_tou_mag = TextBox(ax_tou_mag, 'TOU x ', initial="1.5")
ax_red = plt.axes([0.05, 0.65, 0.06, 0.03])
txt_red = TextBox(ax_red, 'Piros n. ', initial="22")
ax_white = plt.axes([0.05, 0.55, 0.06, 0.03])
txt_white = TextBox(ax_white, 'Fehér n. ', initial="43")
ax_check = plt.axes([0.02, 0.3, 0.1, 0.15])
check_tou = CheckButtons(ax_check, ['0-8h', '8-16h', '16-24h'], [False, False, True])
ax_btn = plt.axes([0.04, 0.15, 0.08, 0.05])
btn_update = Button(ax_btn, 'FRISSÍTÉS', color='lime')
btn_update.on_clicked(update)

update(None)
plt.show()

plt.clf()

# ==========================================================
# 2. KDE ÉS VIZUALIZÁCIÓ (A KDE összesíti a komponenseket)
# ==========================================================
plt.figure(figsize=(10, 6))

# A háttér: a szintetikus adatok hisztogramja, ami másolja a képet
# A 'bins=8' és a fix tartomány biztosítja, hogy ugyanazok az oszlopok jelenjenek meg.
sns.histplot(consumer_loads_synthetic, bins=8, color='#88c488', edgecolor='black', 
             stat="density", alpha=0.9, label="Szintetikus adatok (10k fő)")

# A KDE görbe: a bandwidth (bw_adjust) paraméterrel szabályozzuk, 
# mennyire simuljanak össze a kicsi Gauss komponensek.
# Ha 'bw_adjust=1', akkor a komponensek kicsit még elkülönülnek.
# Ha 'bw_adjust=1.5' (mint itt), szinte tökéletesen összeolvadnak egy sima görbévé.
sns.kdeplot(consumer_loads_synthetic, color='green', linewidth=3, bw_adjust=1.5, label="KDE görbe (Összesített Gauss-görbék)")

# Formázás a beküldött kép stílusában
plt.title("Lakossági fogyasztás eloszlásának rekonstrukciója KDE módszerrel", fontsize=14)
plt.xlabel("Overall consumption [kWh]")
plt.ylabel("Density [technically Density, labeled as Share]")
plt.xlim(2500, 18000)
plt.grid(axis='y', alpha=0.1)
plt.legend()
plt.tight_layout()

