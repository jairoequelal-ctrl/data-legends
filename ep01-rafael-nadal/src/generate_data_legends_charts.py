import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

out_dir = "/working_dir/c_2a8c77acb81307f8/artifacts/file_generation/ttl=63d/output"
os.makedirs(out_dir, exist_ok=True)

# Remove old intermediate/superseded PNGs from output so output stays clean
for old_f in [
    "01_portada_nadal_outlier_verse.png",
    "02_paradoja_arcilla_saque_vs_resto.png",
    "03_machine_learning_pca_kmeans.png",
    "04_anomalia_zscore_roland_garros.png",
    "05_longevidad_reinvencion_30_anos.png",
    "02_paradoja_arcilla_minimalista.png",
    "03_machine_learning_pca_minimalista.png",
    "04_anomalia_zscore_minimalista.png",
    "05_longevidad_reinvencion_minimalista.png",
]:
    p = os.path.join(out_dir, old_f)
    if os.path.exists(p):
        os.remove(p)

# 1. SAVE SLIDE 01 (PORTADA DATA LEGENDS BY JAIRO QUELAL)
img_path = "/working_dir/c_2a8c77acb81307f8/user_uploaded_files/ttl=63d/tool_generated/watermarked_img_1723958563932257270.jpg"
im = Image.open(img_path).convert("RGB")
target_w, target_h = 1440, 1920
im_resized = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
im_dark = ImageEnhance.Brightness(im_resized).enhance(0.94)

out_s1 = os.path.join(out_dir, "01_data_legends_portada_nadal.png")
im_dark.save(out_s1, "PNG")
print("Saved:", out_s1, os.path.exists(out_s1))

# 2. PREPARE DATA & ML MODELS FOR SLIDES 02 - 05
seasons_data = [
    (2003, 17, 1, 2, 11, 6, 2, 1, 0, 0, 49),
    (2004, 18, 14, 10, 14, 3, 0, 0, 1, 0, 51),
    (2005, 19, 28, 6, 50, 2, 1, 2, 11, 1, 2),
    (2006, 20, 25, 10, 26, 0, 8, 2, 5, 1, 2),
    (2007, 21, 31, 12, 31, 1, 8, 2, 6, 1, 2),
    (2008, 22, 46, 10, 24, 1, 12, 0, 8, 2, 1),
    (2009, 23, 42, 12, 24, 2, 0, 0, 5, 1, 2),
    (2010, 24, 40, 9, 22, 0, 9, 1, 7, 3, 1),
    (2011, 25, 33, 11, 28, 2, 8, 2, 3, 1, 2),
    (2012, 26, 17, 3, 23, 1, 2, 2, 4, 1, 4),
    (2013, 27, 36, 4, 39, 2, 0, 1, 10, 2, 1),
    (2014, 28, 20, 6, 25, 3, 3, 2, 4, 1, 3),
    (2015, 29, 30, 12, 26, 6, 5, 2, 3, 0, 5),
    (2016, 30, 18, 10, 21, 4, 0, 0, 2, 0, 9),
    (2017, 31, 41, 10, 24, 1, 3, 1, 6, 2, 1),
    (2018, 32, 14, 2, 26, 1, 5, 1, 5, 1, 2),
    (2019, 33, 32, 3, 21, 3, 5, 1, 4, 2, 1),
    (2020, 34, 18, 6, 9, 1, 0, 0, 2, 1, 2),
    (2021, 35, 5, 2, 19, 3, 0, 0, 2, 0, 6),
    (2022, 36, 24, 6, 10, 2, 5, 0, 4, 2, 2),
    (2023, 37, 1, 3, 0, 0, 0, 0, 0, 0, 670),
    (2024, 38, 2, 2, 10, 6, 0, 0, 0, 0, 153),
]

df = pd.DataFrame(seasons_data, columns=[
    "Year", "Age", "Hard_W", "Hard_L", "Clay_W", "Clay_L",
    "Grass_W", "Grass_L", "Titles", "GS_Titles", "Rank"
])
df["Total_W"] = df["Hard_W"] + df["Clay_W"] + df["Grass_W"]
df["Total_L"] = df["Hard_L"] + df["Clay_L"] + df["Grass_L"]
df["Total_M"] = df["Total_W"] + df["Total_L"]
df["Win_Pct"] = (df["Total_W"] / df["Total_M"]) * 100
df["Clay_Pct"] = np.where((df["Clay_W"] + df["Clay_L"]) > 0,
                          (df["Clay_W"] / (df["Clay_W"] + df["Clay_L"])) * 100, 0)
df["NonClay_W"] = df["Hard_W"] + df["Grass_W"]
df["NonClay_L"] = df["Hard_L"] + df["Grass_L"]
df["NonClay_Pct"] = np.where((df["NonClay_W"] + df["NonClay_L"]) > 0,
                             (df["NonClay_W"] / (df["NonClay_W"] + df["NonClay_L"])) * 100, 0)

df_prime = df[(df["Year"] >= 2004) & (df["Year"] <= 2022)].copy()
features = ["Age", "Total_M", "Clay_Pct", "NonClay_Pct", "Titles", "GS_Titles"]

X = df_prime[features].values.astype(float)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
explained_variance = (S ** 2) / (len(X_scaled) - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
coords = X_scaled @ Vt.T[:, :2]
if np.corrcoef(coords[:, 0], df_prime["Titles"])[0, 1] < 0:
    coords[:, 0] *= -1
if np.corrcoef(coords[:, 1], df_prime["Age"])[0, 1] < 0:
    coords[:, 1] *= -1
df_prime["PC1"] = coords[:, 0]
df_prime["PC2"] = coords[:, 1]

np.random.seed(42)
best_inertia = np.inf
best_labels = None
for _ in range(50):
    idx = np.random.choice(len(X_scaled), 3, replace=False)
    centroids = X_scaled[idx].copy()
    for _ in range(100):
        dists = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X_scaled[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(3)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    inertia = sum(np.linalg.norm(X_scaled[labels == k] - centroids[k]) ** 2 for k in range(3))
    if inertia < best_inertia:
        best_inertia = inertia
        best_labels = labels.copy()
df_prime["Cluster"] = best_labels

tournaments = [
    ("Roland Garros (GS)", 112, 4, "Clay", 14),
    ("Monte-Carlo (M1000)", 73, 6, "Clay", 11),
    ("Roma (M1000)", 70, 9, "Clay", 10),
    ("US Open (GS)", 67, 12, "Hard", 4),
    ("Indian Wells (M1000)", 59, 11, "Hard", 3),
    ("Madrid (M1000)", 57, 12, "Clay/Hard", 5),
    ("Canadá (M1000)", 38, 8, "Hard", 5),
    ("Wimbledon (GS)", 58, 12, "Grass", 2),
    ("Australian Open (GS)", 77, 16, "Hard", 2),
    ("Miami (M1000)", 40, 12, "Hard", 0),
    ("París-Bercy (M1000)", 22, 7, "Hard (i)", 0),
    ("Shanghái (M1000)", 29, 13, "Hard", 1),
    ("Cincinnati (M1000)", 22, 12, "Hard", 1),
    ("ATP Finals", 21, 18, "Hard (i)", 0),
]
df_tourn = pd.DataFrame(tournaments, columns=["Tournament", "W", "L", "Surface", "Titles"])
df_tourn["Win_Pct"] = (df_tourn["W"] / (df_tourn["W"] + df_tourn["L"])) * 100
mean_tourn = df_tourn["Win_Pct"].mean()
std_tourn = df_tourn["Win_Pct"].std()
df_tourn["Z_Score"] = (df_tourn["Win_Pct"] - mean_tourn) / std_tourn

BG_MAIN     = "#0B0E14"
BG_PANEL    = "#121721"
BORDER_COL  = "#1F2937"
TEXT_WHITE  = "#F1F5F9"
TEXT_SILVER = "#94A3B8"
TEXT_MUTED  = "#64748B"

ACCENT_CLAY  = "#E05A47"
ACCENT_STEEL = "#5BC0BE"
ACCENT_GOLD  = "#D4B46A"
ACCENT_SLATE = "#3B4758"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = BORDER_COL
plt.rcParams["axes.linewidth"] = 1.0

def draw_dl_header(ax_h, slide_str, category_spaced, main_title, subtitle):
    ax_h.set_facecolor(BG_MAIN)
    ax_h.set_xlim(0, 100)
    ax_h.set_ylim(0, 100)
    ax_h.axis("off")

    ax_h.text(
        0, 84, f"D A T A   L E G E N D S   •   {category_spaced}",
        fontsize=9.5, fontweight="normal", color=TEXT_SILVER
    )
    ax_h.text(
        100, 84, f"BY JAIRO QUELAL   |   {slide_str}",
        fontsize=9.0, fontweight="normal", color=ACCENT_GOLD, ha="right"
    )

    ax_h.plot([0, 100], [69, 69], color=BORDER_COL, linewidth=0.9)
    ax_h.plot([0, 12], [69, 69], color=ACCENT_CLAY, linewidth=2.0)

    ax_h.text(0, 39, main_title, fontsize=19.5, fontweight="normal", color=TEXT_WHITE)
    ax_h.text(0, 11, subtitle, fontsize=10.8, color=TEXT_SILVER)

def draw_dl_footer(ax_f, takeaway_label, takeaway_text, source_text, is_last=False):
    ax_f.set_facecolor(BG_MAIN)
    ax_f.set_xlim(0, 100)
    ax_f.set_ylim(0, 100)
    ax_f.axis("off")

    card = plt.Rectangle((0, 26), 100, 68, facecolor=BG_PANEL, edgecolor=BORDER_COL, linewidth=1.0, zorder=2)
    left_bar = plt.Rectangle((0, 26), 1.0, 68, facecolor=ACCENT_GOLD, edgecolor="none", zorder=3)
    ax_f.add_patch(card)
    ax_f.add_patch(left_bar)

    ax_f.text(3.2, 76, takeaway_label, fontsize=9.2, fontweight="bold", color=ACCENT_GOLD, va="center", zorder=4)
    ax_f.text(3.2, 48, takeaway_text, fontsize=9.8, color=TEXT_WHITE, linespacing=1.42, va="center", zorder=4)

    ax_f.text(0, 8, source_text, fontsize=8.6, color=TEXT_MUTED)
    right_note = "D A T A   L E G E N D S   •   B Y   J A I R O   Q U E L A L" if not is_last else "B Y   J A I R O   Q U E L A L   •   ¿ P R Ó X I M O   A N Á L I S I S ?"
    ax_f.text(100, 8, right_note, fontsize=8.4, fontweight="normal", color=TEXT_SILVER, ha="right")

# SLIDE 2
fig2 = plt.figure(figsize=(9.0, 12.0), dpi=160, facecolor=BG_MAIN)
ax2_h = fig2.add_axes([0.06, 0.81, 0.88, 0.15])
draw_dl_header(
    ax2_h, "02 / 05", "S U R F A C E   P A R A D O X",
    "EL SECRETO NUNCA FUE SU SAQUE",
    "En hierba retuvo más juegos de saque (89% vs 84%), pero en arcilla quebró el 43% al resto."
)

ax2_main = fig2.add_axes([0.23, 0.23, 0.71, 0.53])
ax2_main.set_facecolor(BG_PANEL)
ax2_main.spines["top"].set_visible(False)
ax2_main.spines["right"].set_visible(False)

metrics_labels = [
    "Juegos Ganados\nal Resto (Return)",
    "Puntos Ganados vs\n1er Saque Rival",
    "Break Points\nConvertidos",
    "Puntos Ganados\ncon 1er Saque",
    "Juegos Ganados\nal Saque (Hold)",
    "% Victorias\nTotales (Win Rate)"
]
clay_vals  = [43, 40, 48, 70, 84, 90.5]
hard_vals  = [28, 32, 43, 73, 86, 77.4]
grass_vals = [24, 28, 41, 75, 89, 79.2]

y = np.arange(len(metrics_labels))
height = 0.24

r1 = ax2_main.barh(y + height, clay_vals, height, label="Arcilla (484–51)", color=ACCENT_CLAY, edgecolor="none")
r2 = ax2_main.barh(y, hard_vals, height, label="Pista Dura (518–151)", color=ACCENT_STEEL, alpha=0.9, edgecolor="none")
r3 = ax2_main.barh(y - height, grass_vals, height, label="Hierba (76–20)", color=ACCENT_GOLD, alpha=0.9, edgecolor="none")

ax2_main.set_yticks(y)
ax2_main.set_yticklabels(metrics_labels, fontsize=10.2, color=TEXT_WHITE)
ax2_main.set_xlim(0, 108)
ax2_main.set_xlabel("Porcentaje Oficial ATP (%)", fontsize=10.2, color=TEXT_SILVER)
ax2_main.tick_params(axis="x", colors=TEXT_SILVER, labelsize=9.8)
ax2_main.grid(axis="x", linestyle="--", alpha=0.14, color=TEXT_SILVER)

for rects, col in [(r1, ACCENT_CLAY), (r2, ACCENT_STEEL), (r3, ACCENT_GOLD)]:
    for r in rects:
        w = r.get_width()
        txt = f"{w:.1f}%" if w in [90.5, 77.4, 79.2] else f"{w:.0f}%"
        ax2_main.text(w + 1.4, r.get_y() + r.get_height()/2, txt,
                      va="center", ha="left", fontsize=9.6, fontweight="bold", color=col)

ax2_main.annotate(
    "DIFERENCIAL CLAVE\n43% Juegos al Resto en Arcilla\nvs. 24% en Hierba (+19 pts %)\n4,914 Break Points generados",
    xy=(43, 0.24), xytext=(53, 0.72),
    arrowprops=dict(facecolor=ACCENT_CLAY, edgecolor=ACCENT_CLAY, shrink=0.06, width=1.5, headwidth=5.5),
    fontsize=9.3, color=TEXT_WHITE,
    bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_MAIN, edgecolor=ACCENT_CLAY, lw=1.2)
)
ax2_main.legend(loc="lower right", facecolor=BG_MAIN, edgecolor=BORDER_COL, labelcolor=TEXT_WHITE, fontsize=9.5)

ax2_f = fig2.add_axes([0.06, 0.03, 0.88, 0.15])
draw_dl_footer(
    ax2_f,
    "I N S I G H T   D E   N E G O C I O   •   F E A T U R E   I M P O R T A N C E",
    "Su servicio fue estadísticamente más efectivo en hierba (89%) que en arcilla (84%). Su dominio del 90.5%\nen tierra batida nace al resto: quebró el 43% de los juegos rivales y convirtió el 48% de sus break points.",
    "Fuente: ATP Tour Official Player Stats"
)

out_s2 = os.path.join(out_dir, "02_data_legends_paradoja_arcilla.png")
plt.savefig(out_s2, dpi=160, facecolor=BG_MAIN)
plt.close(fig2)

# SLIDE 3
fig3 = plt.figure(figsize=(9.0, 12.0), dpi=160, facecolor=BG_MAIN)
ax3_h = fig3.add_axes([0.06, 0.81, 0.88, 0.15])
draw_dl_header(
    ax3_h, "03 / 05", "P C A   &   K - M E A N S   M O D E L",
    "LAS 3 ETAPAS TÁCTICAS DE SU CARRERA",
    "Reducción de dimensionalidad (81.8% varianza) sobre sus 19 temporadas completas."
)

ax3_main = fig3.add_axes([0.11, 0.23, 0.83, 0.53])
ax3_main.set_facecolor(BG_PANEL)
ax3_main.spines["top"].set_visible(False)
ax3_main.spines["right"].set_visible(False)

cluster_stats = df_prime.groupby("Cluster")[["Total_M", "Age", "Win_Pct"]].mean()
prime_c = cluster_stats["Total_M"].idxmax()
remaining = [c for c in cluster_stats.index if c != prime_c]
if cluster_stats.loc[remaining[0], "Win_Pct"] > cluster_stats.loc[remaining[1], "Win_Pct"]:
    reinv_c, trans_c = remaining[0], remaining[1]
else:
    reinv_c, trans_c = remaining[1], remaining[0]

cluster_meta = {
    prime_c: ("Cluster 1: Prime Físico & Volumen (2005–2013 | ~83 part/año)", ACCENT_CLAY, "o"),
    reinv_c: ("Cluster 2: Reinvención +30 Años (Alta Eficiencia | ~50 part/año)", ACCENT_STEEL, "s"),
    trans_c: ("Cluster 3: Transición / Impacto de Lesiones (2004, 2015–16, 2021)", ACCENT_GOLD, "^"),
}

for c_id, (c_label, c_color, c_marker) in cluster_meta.items():
    sub = df_prime[df_prime["Cluster"] == c_id]
    ax3_main.scatter(
        sub["PC1"], sub["PC2"],
        s=sub["Titles"] * 50 + 120,
        c=c_color, marker=c_marker, alpha=0.9,
        edgecolors=TEXT_WHITE, linewidth=1.2, label=c_label, zorder=4
    )

custom_offsets = {
    2013: (0, 12),
    2010: (14, -16),
    2006: (-13, 9),
    2007: (13, 9),
    2011: (-11, 9),
    2009: (11, 9),
}
for _, row in df_prime.iterrows():
    yr = int(row["Year"])
    if yr in [2005, 2008, 2010, 2013, 2017, 2019, 2022]:
        yr_str = f"{yr} ({int(row['GS_Titles'])}GS)"
        fontw = "bold"
        fcol = TEXT_WHITE
    else:
        yr_str = f"'{str(yr)[2:]}"
        fontw = "normal"
        fcol = TEXT_SILVER
    offset_xy = custom_offsets.get(yr, (0, 10))
    ax3_main.annotate(
        yr_str,
        (row["PC1"], row["PC2"]),
        xytext=offset_xy, textcoords="offset points",
        ha="center", fontsize=9.4, fontweight=fontw, color=fcol, zorder=5
    )

ax3_main.axhline(0, color=BORDER_COL, linestyle="--", lw=1.0)
ax3_main.axvline(0, color=BORDER_COL, linestyle="--", lw=1.0)
ax3_main.grid(True, linestyle=":", alpha=0.16, color=TEXT_SILVER)
ax3_main.set_xlabel(f"Componente Principal 1: Volumen + Títulos Ganados ({explained_variance_ratio[0]*100:.1f}% varianza)",
                    fontsize=10.2, color=TEXT_SILVER)
ax3_main.set_ylabel(f"Componente Principal 2: Edad + Eficiencia Selectiva ({explained_variance_ratio[1]*100:.1f}% varianza)",
                    fontsize=10.2, color=TEXT_SILVER)
ax3_main.tick_params(colors=TEXT_SILVER, labelsize=9.8)
ax3_main.set_xlim(df_prime["PC1"].min() - 0.9, df_prime["PC1"].max() + 1.1)
ax3_main.set_ylim(df_prime["PC2"].min() - 0.85, df_prime["PC2"].max() + 1.0)
ax3_main.legend(loc="lower left", facecolor=BG_MAIN, edgecolor=BORDER_COL, labelcolor=TEXT_WHITE, fontsize=9.2)

ax3_f = fig3.add_axes([0.06, 0.03, 0.88, 0.15])
draw_dl_footer(
    ax3_f,
    "C L U S T E R I N G   N O   S U P E R V I S A D O   •   E V O L U C I Ó N",
    "El modelo separa su juventud de alto volumen (~83 partidos/año) del Cluster 2 (+30 años), donde recortó\nun 39% sus partidos pero mantuvo un 85.4% de efectividad y sumó 8 títulos de Grand Slam.",
    "Fuente: Modelo PCA + K-Means (2004–2022)"
)

out_s3 = os.path.join(out_dir, "03_data_legends_pca_kmeans.png")
plt.savefig(out_s3, dpi=160, facecolor=BG_MAIN)
plt.close(fig3)

# SLIDE 4
fig4 = plt.figure(figsize=(9.0, 12.0), dpi=160, facecolor=BG_MAIN)
ax4_h = fig4.add_axes([0.06, 0.81, 0.88, 0.15])
draw_dl_header(
    ax4_h, "04 / 05", "O U T L I E R   D E T E C T I O N",
    "PARÍS: UNA ANOMALÍA A +1.49 SIGMA",
    "Efectividad comparada en los 14 grandes torneos del calendario (media élite = 79.9%)."
)

ax4_main = fig4.add_axes([0.24, 0.23, 0.70, 0.53])
ax4_main.set_facecolor(BG_PANEL)
ax4_main.spines["top"].set_visible(False)
ax4_main.spines["right"].set_visible(False)

df_t_sorted = df_tourn.sort_values("Z_Score", ascending=True)
colors_t = [
    ACCENT_CLAY if z > 1.0 else (ACCENT_STEEL if z > 0 else ACCENT_SLATE)
    for z in df_t_sorted["Z_Score"]
]
bars_z = ax4_main.barh(df_t_sorted["Tournament"], df_t_sorted["Z_Score"], color=colors_t, edgecolor="none", height=0.66)

ax4_main.axvline(0, color=TEXT_WHITE, linestyle="-", linewidth=1.1, alpha=0.65)
ax4_main.axvline(1.0, color=ACCENT_GOLD, linestyle="--", linewidth=1.1, alpha=0.7)
ax4_main.axvline(-1.0, color=TEXT_MUTED, linestyle="--", linewidth=1.0, alpha=0.45)

for bar, (_, row) in zip(bars_z, df_t_sorted.iterrows()):
    z = row["Z_Score"]
    pct = row["Win_Pct"]
    offset = 0.08 if z >= 0 else -0.08
    ha_align = "left" if z >= 0 else "right"
    col_txt = ACCENT_GOLD if row["Tournament"] == "Roland Garros (GS)" else TEXT_WHITE
    ax4_main.text(z + offset, bar.get_y() + bar.get_height()/2,
                  f"{pct:.1f}% ({int(row['W'])}–{int(row['L'])})",
                  va="center", ha=ha_align, fontsize=9.3, fontweight="bold", color=col_txt)

ax4_main.set_xlim(-3.55, 2.55)
ax4_main.set_xlabel("Z-Score (Desviaciones Estándar respecto a su media en Grandes Torneos = 79.9%)",
                    fontsize=10.0, color=TEXT_SILVER)
ax4_main.tick_params(axis="y", colors=TEXT_WHITE, labelsize=10.0)
ax4_main.tick_params(axis="x", colors=TEXT_SILVER, labelsize=9.8)
ax4_main.grid(axis="x", linestyle="--", alpha=0.14, color=TEXT_SILVER)

ax4_f = fig4.add_axes([0.06, 0.03, 0.88, 0.15])
draw_dl_footer(
    ax4_f,
    "A N O M A L Í A   E S T A D Í S T I C A   •   R O L A N D   G A R R O S",
    "112 victorias y 4 derrotas (96.6%), 14–0 en finales, 90 triunfos en sets corridos y 4 torneos sin ceder\nun solo set. Incluso frente a su propia media de élite (79.9%), París se ubica a +1.49σ.",
    "Fuente: Roland-Garros Official Records"
)

out_s4 = os.path.join(out_dir, "04_data_legends_zscore_paris.png")
plt.savefig(out_s4, dpi=160, facecolor=BG_MAIN)
plt.close(fig4)

# SLIDE 5
fig5 = plt.figure(figsize=(9.0, 12.0), dpi=160, facecolor=BG_MAIN)
ax5_h = fig5.add_axes([0.06, 0.81, 0.88, 0.15])
draw_dl_header(
    ax5_h, "05 / 05", "T I M E   S E R I E S   &   W O R K L O A D",
    "912 SEMANAS EN EL TOP 10 SIN QUEMARSE",
    "Reducción estratégica de volumen (-39%) tras los 30 años y récord de saque en 2019 (90%)."
)

ax5_main = fig5.add_axes([0.11, 0.23, 0.78, 0.53])
ax5_twin = ax5_main.twinx()

ax5_twin.set_facecolor(BG_PANEL)
ax5_main.set_facecolor("none")
ax5_main.set_zorder(ax5_twin.get_zorder() + 1)
ax5_main.spines["top"].set_visible(False)
ax5_twin.spines["top"].set_visible(False)

df_traj = df[(df["Year"] >= 2004) & (df["Year"] <= 2022)].copy()

ax5_twin.bar(df_traj["Year"], df_traj["Total_M"], color="#1E2736", alpha=0.7, width=0.62, label="Partidos Jugados / Año")
ax5_twin.set_ylabel("Volumen de Partidos Jugados", fontsize=10.2, color=TEXT_SILVER)
ax5_twin.set_ylim(0, 115)
ax5_twin.tick_params(axis="y", colors=TEXT_SILVER, labelsize=9.8)

ax5_main.plot(df_traj["Year"], df_traj["Clay_Pct"], color=ACCENT_CLAY, marker="o", linewidth=2.6, markersize=6.0, label="% Victorias en Arcilla")
ax5_main.plot(df_traj["Year"], df_traj["Win_Pct"], color=ACCENT_STEEL, marker="s", linewidth=2.3, markersize=5.5, label="% Victorias Global Temporada")
ax5_main.plot(df_traj["Year"], df_traj["NonClay_Pct"], color=ACCENT_GOLD, marker="^", linewidth=1.9, linestyle="--", markersize=5.0, label="% Victorias Dura + Hierba")

ax5_main.axvline(2016.5, color=ACCENT_GOLD, linestyle="-.", linewidth=1.6)
ax5_main.text(
    2016.65, 45.5,
    "REINVENCIÓN +30 AÑOS (2017–2022)\n• 8 Títulos de Grand Slam ganados\n• 2019: Récord personal 90% juegos\n  ganados al saque y 76% 1er saque\n• Menor carga, máxima eficiencia",
    fontsize=9.2, color=TEXT_WHITE, zorder=10,
    bbox=dict(boxstyle="round,pad=0.45", facecolor=BG_MAIN, edgecolor=ACCENT_GOLD, lw=1.2)
)

ax5_main.set_ylim(40, 105)
ax5_main.set_xticks(df_traj["Year"][::2])
ax5_main.set_xticklabels([str(y) for y in df_traj["Year"][::2]], fontsize=9.8, color=TEXT_WHITE)
ax5_main.set_ylabel("Porcentaje de Victorias (%)", fontsize=10.2, color=TEXT_WHITE)
ax5_main.tick_params(axis="y", colors=TEXT_WHITE, labelsize=9.8)
ax5_main.grid(True, linestyle="--", alpha=0.14, color=TEXT_SILVER)

lines1, labels1 = ax5_main.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5_main.legend(lines1 + lines2, labels1 + labels2, loc="lower left", facecolor=BG_MAIN, edgecolor=BORDER_COL, labelcolor=TEXT_WHITE, fontsize=9.0)

ax5_f = fig5.add_axes([0.06, 0.03, 0.88, 0.15])
draw_dl_footer(
    ax5_f,
    "L E C C I Ó N   E J E C U T I V A   •   O P T I M I Z A C I Ó N   D E   M A R G E N",
    "Cuando el volumen bruto ya no escala, optimizar el margen por partido te mantiene en la cima:\n18 temporadas consecutivas ganando al menos 2 títulos ATP (2005–2022).",
    "Fuente: ATP Tour Rankings y Player Stats (2004–2022)",
    is_last=True
)

out_s5 = os.path.join(out_dir, "05_data_legends_longevidad.png")
plt.savefig(out_s5, dpi=160, facecolor=BG_MAIN)
plt.close(fig5)

print("Files in output now:", sorted(os.listdir(out_dir)))
