import os
import math
import wave
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out_dir = "/working_dir/c_2a8c77acb81307f8/artifacts/file_generation/ttl=63d/output"
inter_dir = "/working_dir/c_2a8c77acb81307f8/artifacts/file_generation/ttl=63d/intermediate"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)

W, H = 1080, 1920
FPS = 30

FONT_DIR = "/usr/share/fonts/opentype/inter"
def get_font(name, size):
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)

f_eyebrow   = get_font("Inter-SemiBold.otf", 19)
f_title     = get_font("InterDisplay-Light.otf", 48)
f_title_b   = get_font("InterDisplay-SemiBold.otf", 48)
f_sub       = get_font("Inter-Regular.otf", 24)
f_sec_title = get_font("Inter-SemiBold.otf", 23)
f_label     = get_font("Inter-Medium.otf", 24)
f_label_sm  = get_font("Inter-Regular.otf", 20)
f_val       = get_font("InterDisplay-Bold.otf", 28)
f_kpi_big   = get_font("InterDisplay-Bold.otf", 62)
f_kpi_med   = get_font("InterDisplay-Bold.otf", 42)
f_card_hdr  = get_font("Inter-Bold.otf", 21)
f_card_body = get_font("Inter-Regular.otf", 21)
f_footer    = get_font("Inter-Medium.otf", 17)

BG_MAIN      = (11, 14, 20)     # #0B0E14
BG_PANEL     = (18, 23, 33)     # #121721
BG_PANEL_ALT = (24, 31, 44)     # #181F2C
BORDER_COL   = (38, 49, 66)     # #263142
TEXT_WHITE   = (241, 245, 249)  # #F1F5F9
TEXT_SILVER  = (148, 163, 184)  # #94A3B8
TEXT_MUTED   = (100, 116, 139)  # #64748B

ACCENT_CLAY  = (224, 90, 71)    # #E05A47
ACCENT_STEEL = (91, 192, 190)   # #5BC0BE
ACCENT_GOLD  = (212, 180, 106)  # #D4B46A
ACCENT_SLATE = (59, 71, 88)     # #3B4758

def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

def ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2

def blend_col(c1, c2, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return tuple(int(c1[i] * (1 - alpha) + c2[i] * alpha) for i in range(3))

def draw_standard_chrome(draw, step_tag, cat_tag, title_line1, title_line2, subtitle_text, prog_ratio=1.0):
    draw.rectangle([0, 0, W, 6], fill=BG_PANEL)
    draw.rectangle([0, 0, int(W * prog_ratio), 6], fill=ACCENT_CLAY)

    draw.text((64, 54), f"D A T A   L E G E N D S   •   {cat_tag}", font=f_eyebrow, fill=TEXT_SILVER)
    draw.text((W - 64, 54), f"BY JAIRO QUELAL  |  {step_tag}", font=f_eyebrow, fill=ACCENT_GOLD, anchor="ra")

    draw.line([(64, 94), (W - 64, 94)], fill=BORDER_COL, width=2)
    draw.line([(64, 94), (190, 94)], fill=ACCENT_CLAY, width=4)

    draw.text((64, 116), title_line1, font=f_title_b, fill=TEXT_WHITE)
    if title_line2:
        draw.text((64, 174), title_line2, font=f_title, fill=ACCENT_GOLD)
        sub_y = 242
    else:
        sub_y = 184
    draw.text((64, sub_y), subtitle_text, font=f_sub, fill=TEXT_SILVER)

    draw.text((64, H - 48), "Fuente: ATP Tour Official Stats & Roland-Garros", font=f_footer, fill=TEXT_MUTED)
    draw.text((W - 64, H - 48), "DATA LEGENDS  •  BY JAIRO QUELAL", font=f_footer, fill=TEXT_SILVER, anchor="ra")

def draw_bottom_insight_card(draw, label_txt, line1, line2, alpha=1.0):
    if alpha <= 0.01:
        return
    y_offset = int((1.0 - ease_out_cubic(alpha)) * 45)
    y0 = 1610 + y_offset
    y1 = 1825 + y_offset
    bg = blend_col(BG_MAIN, BG_PANEL, alpha)
    bdr = blend_col(BG_MAIN, BORDER_COL, alpha)
    gold = blend_col(BG_MAIN, ACCENT_GOLD, alpha)
    white = blend_col(BG_MAIN, TEXT_WHITE, alpha)

    draw.rounded_rectangle([64, y0, W - 64, y1], radius=12, fill=bg, outline=bdr, width=2)
    draw.rectangle([64, y0, 74, y1], fill=gold)
    draw.text((96, y0 + 28), label_txt, font=f_card_hdr, fill=gold)
    draw.text((96, y0 + 78), line1, font=f_card_body, fill=white)
    draw.text((96, y0 + 114), line2, font=f_card_body, fill=white)

def open_ffmpeg_writer(out_mp4_path):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "veryfast",
        out_mp4_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================================================================
# CLIP 1: INTRO PORTADA CINEMÁTICA (6 SECONDS = 180 FRAMES)
# =========================================================================
print("Rendering Clip 1 (Intro Cover)...")
cover_src = os.path.join(out_dir, "01_data_legends_portada_nadal.png")
base_cover = Image.open(cover_src).convert("RGB")
cover_1080 = base_cover.resize((1080, 1440), Image.Resampling.LANCZOS)

clip1_path = os.path.join(out_dir, "clip_01_intro_nadal.mp4")
proc1 = open_ffmpeg_writer(clip1_path)
N1 = 6 * FPS

for f in range(N1):
    t = f / float(N1 - 1)
    frame = Image.new("RGB", (W, H), BG_MAIN)

    zoom = 1.0 + 0.045 * ease_out_cubic(t)
    zw, zh = int(1080 * zoom), int(1440 * zoom)
    z_img = cover_1080.resize((zw, zh), Image.Resampling.BILINEAR)
    x_crop = (zw - 1080) // 2
    y_crop = (zh - 1440) // 2
    cropped = z_img.crop((x_crop, y_crop, x_crop + 1080, y_crop + 1440))
    frame.paste(cropped, (0, 20))

    draw = ImageDraw.Draw(frame)

    hud_alpha = ease_out_cubic((t - 0.08) / 0.35)
    if hud_alpha > 0:
        panel_y = 1380 + int((1 - hud_alpha) * 40)
        draw.rounded_rectangle([56, panel_y, W - 56, 1835], radius=16, fill=BG_PANEL, outline=BORDER_COL, width=2)
        draw.rectangle([56, panel_y, 260, panel_y + 5], fill=ACCENT_CLAY)

        draw.text((92, panel_y + 28), "E P I S O D I O   # 0 1   •   E L   M I T O   E Q U I V O C A D O", font=f_eyebrow, fill=ACCENT_GOLD)

        cnt_prog = ease_out_cubic((t - 0.15) / 0.55)
        matches_val = int(1308 * cnt_prog)
        clay_val = 90.5 * cnt_prog
        rg_val = 96.6 * cnt_prog

        draw.text((220, panel_y + 92), f"{matches_val:,}", font=f_kpi_med, fill=TEXT_WHITE, anchor="ma")
        draw.text((220, panel_y + 150), "PARTIDOS ATP", font=f_footer, fill=TEXT_SILVER, anchor="ma")

        draw.line([(380, panel_y + 88), (380, panel_y + 175)], fill=BORDER_COL, width=2)

        draw.text((540, panel_y + 92), f"{clay_val:.1f}%", font=f_kpi_med, fill=ACCENT_CLAY, anchor="ma")
        draw.text((540, panel_y + 150), "VICTORIAS ARCILLA", font=f_footer, fill=TEXT_SILVER, anchor="ma")

        draw.line([(700, panel_y + 88), (700, panel_y + 175)], fill=BORDER_COL, width=2)

        draw.text((860, panel_y + 92), f"{rg_val:.1f}%", font=f_kpi_med, fill=ACCENT_GOLD, anchor="ma")
        draw.text((860, panel_y + 150), "ROLAND GARROS", font=f_footer, fill=TEXT_SILVER, anchor="ma")

        hook_alpha = ease_out_cubic((t - 0.35) / 0.40)
        if hook_alpha > 0:
            h_col = blend_col(BG_PANEL, TEXT_WHITE, hook_alpha)
            s_col = blend_col(BG_PANEL, ACCENT_STEEL, hook_alpha)
            draw.line([(92, panel_y + 205), (W - 92, panel_y + 205)], fill=BORDER_COL, width=1)
            draw.text((92, panel_y + 230), "«Todo el mundo cree que dominó por su fuerza física.", font=f_card_body, fill=h_col)
            draw.text((92, panel_y + 268), "Los datos dicen que mirábamos el lado equivocado de la cancha.»", font=f_card_hdr, fill=s_col)

    draw.text((64, H - 45), "D A T A   L E G E N D S", font=f_footer, fill=TEXT_MUTED)
    draw.text((W - 64, H - 45), "B Y   J A I R O   Q U E L A L", font=f_footer, fill=ACCENT_GOLD, anchor="ra")

    proc1.stdin.write(frame.tobytes())

proc1.stdin.close()
proc1.wait()

# =========================================================================
# CLIP 2: LA PARADOJA DE LA ARCILLA (12 SECONDS = 360 FRAMES)
# =========================================================================
print("Rendering Clip 2 (Clay Paradox)...")
clip2_path = os.path.join(out_dir, "clip_02_paradoja_arcilla.mp4")
proc2 = open_ffmpeg_writer(clip2_path)
N2 = 12 * FPS

for f in range(N2):
    t = f / float(N2 - 1)
    sec = t * 12.0
    frame = Image.new("RGB", (W, H), BG_MAIN)
    draw = ImageDraw.Draw(frame)

    draw_standard_chrome(
        draw, "02 / 05", "S U R F A C E   P A R A D O X",
        "EL SECRETO NUNCA FUE", "SU PROPIO SAQUE",
        "¿Dónde ganaba realmente Nadal? Comparativa en 1,308 partidos.",
        prog_ratio=t
    )

    p1_y0, p1_y1 = 315, 735
    draw.rounded_rectangle([64, p1_y0, W - 64, p1_y1], radius=12, fill=BG_PANEL, outline=BORDER_COL, width=2)
    draw.text((96, p1_y0 + 26), "1. JUEGOS GANADOS CON SU PROPIO SAQUE (HOLD %)", font=f_sec_title, fill=TEXT_WHITE)
    draw.text((96, p1_y0 + 62), "Sorpresa: La arcilla fue su superficie más baja reteniendo el servicio", font=f_label_sm, fill=TEXT_SILVER)

    hold_items = [
        ("Hierba", 89.0, ACCENT_GOLD, p1_y0 + 125),
        ("Pista Dura", 86.0, ACCENT_STEEL, p1_y0 + 210),
        ("Arcilla", 84.0, ACCENT_CLAY, p1_y0 + 295),
    ]
    for idx, (surf_name, target_pct, col, by) in enumerate(hold_items):
        bar_prog = ease_out_cubic((sec - (0.4 + idx * 0.45)) / 1.8)
        cur_pct = target_pct * bar_prog
        draw.text((96, by + 8), surf_name, font=f_label, fill=TEXT_WHITE)
        bar_x0 = 270
        max_bar_w = 560
        cur_w = int(max_bar_w * (cur_pct / 100.0))
        draw.rounded_rectangle([bar_x0, by, bar_x0 + max_bar_w, by + 44], radius=6, fill=BG_MAIN)
        if cur_w > 8:
            draw.rounded_rectangle([bar_x0, by, bar_x0 + cur_w, by + 44], radius=6, fill=col)
        if bar_prog > 0.1:
            draw.text((bar_x0 + cur_w + 18, by + 7), f"{cur_pct:.0f}%", font=f_val, fill=col)

    p2_y0, p2_y1 = 770, 1310
    p2_alpha = ease_out_cubic((sec - 3.2) / 0.8)
    if p2_alpha > 0:
        draw.rounded_rectangle([64, p2_y0, W - 64, p2_y1], radius=12, fill=BG_PANEL, outline=BORDER_COL, width=2)
        draw.text((96, p2_y0 + 26), "2. LA VERDADERA ANOMALÍA: JUEGOS QUEBRADOS AL RIVAL", font=f_sec_title, fill=ACCENT_CLAY)
        draw.text((96, p2_y0 + 62), "De cada 10 juegos de saque del oponente, ¿cuántos quebraba Nadal?", font=f_label_sm, fill=TEXT_SILVER)

        return_rows = [
            ("Hierba (24%)", 2.4, ACCENT_GOLD, p2_y0 + 145),
            ("Pista Dura (28%)", 2.8, ACCENT_STEEL, p2_y0 + 265),
            ("Arcilla (43%)", 4.3, ACCENT_CLAY, p2_y0 + 385),
        ]
        for r_idx, (r_label, broken_balls, r_col, ry) in enumerate(return_rows):
            row_prog = ease_out_cubic((sec - (3.8 + r_idx * 0.7)) / 2.2)
            cur_broken = broken_balls * row_prog
            draw.text((96, ry - 14), r_label, font=f_label, fill=TEXT_WHITE if r_idx == 2 else TEXT_SILVER)

            start_cx = 390
            spacing_cx = 56
            radius = 20
            for b_i in range(10):
                cx = start_cx + b_i * spacing_cx
                cy = ry
                fill_amt = max(0.0, min(1.0, cur_broken - b_i))
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=BG_MAIN, outline=BORDER_COL, width=2)
                if fill_amt > 0.02:
                    c_fill = blend_col(BG_MAIN, r_col, fill_amt)
                    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=c_fill, outline=TEXT_WHITE if fill_amt > 0.8 else r_col, width=2)
                    draw.arc([cx - 12, cy - 12, cx + 12, cy + 12], start=30, end=210, fill=BG_MAIN, width=2)

            if row_prog > 0.2:
                pct_val = cur_broken * 10.0
                draw.text((96, ry + 20), f"Quebró {cur_broken:.1f} de 10 ({pct_val:.0f}%)", font=f_label_sm, fill=r_col)

    p3_y0, p3_y1 = 1345, 1570
    p3_alpha = ease_out_cubic((sec - 7.0) / 1.2)
    if p3_alpha > 0:
        draw.rounded_rectangle([64, p3_y0, W - 64, p3_y1], radius=12, fill=BG_PANEL_ALT, outline=ACCENT_CLAY, width=2)
        bp_cnt = int(4914 * p3_alpha)
        conv_cnt = 48.0 * p3_alpha
        draw.text((310, p3_y0 + 40), f"{bp_cnt:,}", font=f_kpi_big, fill=ACCENT_CLAY, anchor="ma")
        draw.text((310, p3_y0 + 122), "OPORTUNIDADES DE QUIEBRE", font=f_label_sm, fill=TEXT_WHITE, anchor="ma")
        draw.text((310, p3_y0 + 152), "generadas en toda su carrera", font=f_footer, fill=TEXT_SILVER, anchor="ma")

        draw.line([(560, p3_y0 + 35), (560, p3_y1 - 35)], fill=BORDER_COL, width=2)

        draw.text((780, p3_y0 + 40), f"{conv_cnt:.0f}%", font=f_kpi_big, fill=ACCENT_GOLD, anchor="ma")
        draw.text((780, p3_y0 + 122), "CONVERSIÓN EN ARCILLA", font=f_label_sm, fill=TEXT_WHITE, anchor="ma")
        draw.text((780, p3_y0 + 152), "(vs. 43% en dura y 41% en hierba)", font=f_footer, fill=TEXT_SILVER, anchor="ma")

    card_alpha = ease_out_cubic((sec - 8.2) / 1.0)
    draw_bottom_insight_card(
        draw,
        "I N S I G H T   C L A V E   •   F E A T U R E   I M P O R T A N C E",
        "Su 90.5% de victorias en arcilla nació de asfixiar el servicio del rival:",
        "quebró el 43% de todos los turnos de saque que enfrentó en tierra batida.",
        alpha=card_alpha
    )

    proc2.stdin.write(frame.tobytes())

proc2.stdin.close()
proc2.wait()

# =========================================================================
# CLIP 3: ANOMALÍA EN ROLAND GARROS (+1.49 SIGMA) (11 SECONDS = 330 FRAMES)
# =========================================================================
print("Rendering Clip 3 (Roland Garros Z-Score Outlier)...")
clip3_path = os.path.join(out_dir, "clip_03_anomalia_paris_zscore.mp4")
proc3 = open_ffmpeg_writer(clip3_path)
N3 = 11 * FPS

tournaments = [
    ("Roland Garros (GS)", 112, 4, 96.55, 1.49),
    ("Monte-Carlo (M1000)", 73, 6, 92.41, 1.12),
    ("Roma (M1000)", 70, 9, 88.61, 0.78),
    ("US Open (GS)", 67, 12, 84.81, 0.44),
    ("Indian Wells (M1000)", 59, 11, 84.29, 0.39),
    ("Wimbledon (GS)", 58, 12, 82.86, 0.27),
    ("Australian Open (GS)", 77, 16, 82.80, 0.26),
    ("Madrid (M1000)", 57, 12, 82.61, 0.24),
    ("Canadá (M1000)", 38, 8, 82.61, 0.24),
    ("Miami (M1000)", 40, 12, 76.92, -0.27),
    ("París-Bercy (M1000)", 22, 7, 75.86, -0.36),
    ("Shanghái (M1000)", 29, 13, 69.05, -0.97),
    ("Cincinnati (M1000)", 22, 12, 64.71, -1.36),
    ("ATP Finals", 21, 18, 53.85, -2.33),
]

for f in range(N3):
    t = f / float(N3 - 1)
    sec = t * 11.0
    frame = Image.new("RGB", (W, H), BG_MAIN)
    draw = ImageDraw.Draw(frame)

    draw_standard_chrome(
        draw, "03 / 05", "O U T L I E R   Z - S C O R E",
        "PARÍS: UNA ANOMALÍA", "A +1.49 DESVIACIONES ESTÁNDAR",
        "Efectividad en los 14 grandes torneos frente a su media élite (79.9%).",
        prog_ratio=t
    )

    chart_y0, chart_y1 = 315, 1335
    draw.rounded_rectangle([64, chart_y0, W - 64, chart_y1], radius=12, fill=BG_PANEL, outline=BORDER_COL, width=2)

    zero_x = 615
    scale_x = 115.0

    for s_val, s_lbl in [(-2.0, "-2.0σ"), (-1.0, "-1.0σ"), (0.0, "Media (79.9%)"), (1.0, "+1.0σ")]:
        gx = int(zero_x + s_val * scale_x)
        col_g = ACCENT_GOLD if s_val == 1.0 else (TEXT_WHITE if s_val == 0.0 else BORDER_COL)
        draw.line([(gx, chart_y0 + 30), (gx, chart_y1 - 55)], fill=col_g, width=2 if s_val in [0.0, 1.0] else 1)
        draw.text((gx, chart_y1 - 42), s_lbl, font=f_footer, fill=col_g if s_val in [0.0, 1.0] else TEXT_MUTED, anchor="ma")

    row_h = 66
    start_y = chart_y0 + 40
    for idx, (t_name, w_cnt, l_cnt, win_pct, z_val) in enumerate(tournaments):
        rev_idx = (len(tournaments) - 1) - idx
        delay = 0.3 + rev_idx * 0.18
        bar_prog = ease_out_cubic((sec - delay) / 1.4)
        cur_z = z_val * bar_prog

        ry = start_y + idx * row_h
        is_rg = (idx == 0)
        lbl_col = ACCENT_GOLD if is_rg else TEXT_WHITE
        draw.text((335, ry + 11), t_name, font=f_label_sm if not is_rg else f_label, fill=lbl_col, anchor="ra")

        bx = int(zero_x + cur_z * scale_x)
        b_col = ACCENT_CLAY if z_val > 1.0 else (ACCENT_STEEL if z_val > 0 else ACCENT_SLATE)
        if bar_prog > 0.02:
            x_left = min(zero_x, bx)
            x_right = max(zero_x, bx)
            draw.rounded_rectangle([x_left, ry + 5, max(x_left + 4, x_right), ry + 43], radius=5, fill=b_col)
            if is_rg and bar_prog > 0.8:
                draw.rounded_rectangle([x_left, ry + 5, x_right, ry + 43], radius=5, outline=ACCENT_GOLD, width=2)

            if bar_prog > 0.35:
                txt_str = f"{win_pct:.1f}% ({w_cnt}–{l_cnt})"
                if z_val >= 0:
                    draw.text((x_right + 12, ry + 11), txt_str, font=f_label_sm, fill=ACCENT_GOLD if is_rg else TEXT_WHITE)
                else:
                    # Place negative bar labels to the right of zero_x so they never collide with tournament names!
                    draw.text((zero_x + 12, ry + 11), txt_str, font=f_label_sm, fill=TEXT_SILVER)

    rg_alpha = ease_out_cubic((sec - 5.0) / 1.0)
    if rg_alpha > 0:
        draw.rounded_rectangle([64, 1365, W - 64, 1575], radius=12, fill=BG_PANEL_ALT, outline=ACCENT_GOLD, width=2)
        draw.text((540, 1394), "RÉCORD HISTÓRICO EN ROLAND GARROS (+1.49σ)", font=f_sec_title, fill=ACCENT_GOLD, anchor="ma")

        rg_kpis = [
            ("112–4", "RÉCORD PARTIDOS", 200),
            ("14–0", "EN FINALES", 435),
            ("90", "EN 3 SETS", 655),
            ("4", "TORNEOS INVICTO", 875),
        ]
        for k_val, k_lbl, kx in rg_kpis:
            draw.text((kx, 1448), k_val, font=f_kpi_med, fill=TEXT_WHITE, anchor="ma")
            draw.text((kx, 1514), k_lbl, font=f_footer, fill=TEXT_SILVER, anchor="ma")

    card_alpha = ease_out_cubic((sec - 6.5) / 1.0)
    draw_bottom_insight_card(
        draw,
        "A N O M A L Í A   E S T A D Í S T I C A   •   Z - S C O R E",
        "Incluso comparándolo contra sí mismo en los 14 grandes torneos del mundo,",
        "su 96.6% en París rompe la curva situándose a +1.49 desviaciones estándar.",
        alpha=card_alpha
    )

    proc3.stdin.write(frame.tobytes())

proc3.stdin.close()
proc3.wait()

# =========================================================================
# CLIP 4: REINVENCIÓN A LOS 30 AÑOS (PCA + K-MEANS & LONGEVIDAD) (11s = 330 FRAMES)
# =========================================================================
print("Rendering Clip 4 (PCA + K-Means & +30 Years Reinvention)...")
clip4_path = os.path.join(out_dir, "clip_04_reinvencion_30_anos.mp4")
proc4 = open_ffmpeg_writer(clip4_path)
N4 = 11 * FPS

years_list = list(range(2004, 2023))
matches_yr = [31, 88, 71, 85, 93, 78, 78, 82, 48, 82, 59, 81, 53, 78, 49, 65, 34, 29, 47]
clay_pct_yr= [82.4, 96.2, 100.0, 96.9, 96.0, 92.3, 100.0, 93.3, 95.8, 95.1, 89.3, 81.3, 84.0, 96.0, 96.3, 87.5, 90.0, 86.4, 83.3]
win_pct_yr = [67.7, 88.6, 83.1, 82.4, 88.2, 84.6, 91.0, 84.1, 87.5, 91.5, 81.4, 75.3, 73.6, 87.2, 91.8, 89.2, 79.4, 82.8, 83.0]

for f in range(N4):
    t = f / float(N4 - 1)
    sec = t * 11.0
    frame = Image.new("RGB", (W, H), BG_MAIN)
    draw = ImageDraw.Draw(frame)

    draw_standard_chrome(
        draw, "04 / 05", "P C A   &   W O R K L O A D",
        "EL ALGORITMO DE", "REINVENTARSE A LOS 30 AÑOS",
        "Cómo redujo un 39% su volumen de partidos para ganar 8 Grand Slams más.",
        prog_ratio=t
    )

    c1_y0, c1_y1 = 315, 715
    draw.rounded_rectangle([64, c1_y0, 520, c1_y1], radius=12, fill=BG_PANEL, outline=ACCENT_CLAY, width=2)
    draw.text((92, c1_y0 + 24), "CLUSTER 1: PRIME FÍSICO", font=f_card_hdr, fill=ACCENT_CLAY)
    draw.text((92, c1_y0 + 56), "Temporadas 2005 – 2013 (19–27 años)", font=f_footer, fill=TEXT_SILVER)

    p_c1 = ease_out_cubic((sec - 0.3) / 1.5)
    draw.text((92, c1_y0 + 108), f"{int(83 * p_c1)} part/año", font=f_kpi_med, fill=TEXT_WHITE)
    draw.text((92, c1_y0 + 165), "Volumen físico máximo en cancha", font=f_label_sm, fill=TEXT_SILVER)
    draw.text((92, c1_y0 + 225), f"{int(13 * p_c1)} Grand Slams", font=f_kpi_med, fill=ACCENT_CLAY)
    draw.text((92, c1_y0 + 282), "87.1% efectividad media", font=f_label_sm, fill=TEXT_WHITE)

    p_c2 = ease_out_cubic((sec - 1.5) / 1.5)
    if p_c2 > 0:
        draw.rounded_rectangle([555, c1_y0, W - 64, c1_y1], radius=12, fill=BG_PANEL, outline=ACCENT_STEEL, width=2)
        draw.text((582, c1_y0 + 24), "CLUSTER 2: REINVENCIÓN +30", font=f_card_hdr, fill=ACCENT_STEEL)
        draw.text((582, c1_y0 + 56), "Temporadas 2017 – 2022 (31–36 años)", font=f_footer, fill=TEXT_SILVER)

        draw.text((582, c1_y0 + 108), f"{int(50 * p_c2)} part/año (-39%)", font=f_kpi_med, fill=ACCENT_GOLD)
        draw.text((582, c1_y0 + 165), "Dosificación selectiva del calendario", font=f_label_sm, fill=TEXT_SILVER)
        draw.text((582, c1_y0 + 225), f"+{int(8 * p_c2)} Grand Slams", font=f_kpi_med, fill=ACCENT_STEEL)
        draw.text((582, c1_y0 + 282), "Récord personal 90% saque en 2019", font=f_label_sm, fill=TEXT_WHITE)

    t_y0, t_y1 = 750, 1570
    draw.rounded_rectangle([64, t_y0, W - 64, t_y1], radius=12, fill=BG_PANEL, outline=BORDER_COL, width=2)
    draw.text((96, t_y0 + 26), "EVOLUCIÓN TEMPORAL (2004–2022): VOLUMEN VS. EFECTIVIDAD", font=f_sec_title, fill=TEXT_WHITE)
    draw.text((96, t_y0 + 62), "Barras oscuras = Partidos/año  |  Terracota = % Arcilla  |  Azul Acero = % Global", font=f_footer, fill=TEXT_SILVER)

    plot_x0, plot_x1 = 135, W - 105
    plot_y0, plot_y1 = t_y0 + 130, t_y1 - 80
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    for pct_g in [50, 75, 100]:
        gy = int(plot_y1 - ((pct_g - 40) / 65.0) * plot_h)
        draw.line([(plot_x0, gy), (plot_x1, gy)], fill=BORDER_COL, width=1)
        draw.text((plot_x0 - 15, gy - 10), f"{pct_g}%", font=f_footer, fill=TEXT_SILVER, anchor="ra")

    tl_prog = ease_in_out_cubic((sec - 2.2) / 5.4)
    max_idx_float = tl_prog * (len(years_list) - 1)

    pts_clay = []
    pts_win = []
    thresh_x = None
    for i, yr in enumerate(years_list):
        if i > max_idx_float + 0.5:
            break
        item_alpha = max(0.0, min(1.0, max_idx_float - i + 1.0))
        px = int(plot_x0 + (i / float(len(years_list) - 1)) * plot_w)

        m_val = matches_yr[i] * item_alpha
        bar_h = int((m_val / 110.0) * plot_h)
        bx0, bx1 = px - 14, px + 14
        draw.rectangle([bx0, plot_y1 - bar_h, bx1, plot_y1], fill=(32, 43, 60))

        if yr % 3 == 1 or yr == 2022:
            draw.text((px, plot_y1 + 16), str(yr), font=f_footer, fill=TEXT_SILVER, anchor="ma")

        cy_clay = int(plot_y1 - ((clay_pct_yr[i] - 40) / 65.0) * plot_h)
        cy_win  = int(plot_y1 - ((win_pct_yr[i] - 40) / 65.0) * plot_h)
        pts_clay.append((px, cy_clay))
        pts_win.append((px, cy_win))

        if yr == 2016 and item_alpha > 0.8:
            thresh_x = int(px + (plot_w / float(len(years_list) - 1)) * 0.5)

    if len(pts_clay) >= 2:
        draw.line(pts_clay, fill=ACCENT_CLAY, width=5)
        draw.line(pts_win, fill=ACCENT_STEEL, width=5)
    for px, py in pts_clay:
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=ACCENT_CLAY, outline=TEXT_WHITE, width=2)
    for px, py in pts_win:
        draw.rectangle([px - 5, py - 5, px + 5, py + 5], fill=ACCENT_STEEL, outline=TEXT_WHITE, width=2)

    # Draw 2016.5 threshold line and callout box AFTER bars/lines so it stays crisp and unoccluded!
    if thresh_x is not None:
        draw.line([(thresh_x, plot_y0 - 10), (thresh_x, plot_y1)], fill=ACCENT_GOLD, width=3)
        box_x0, box_x1 = 540, plot_x1 - 5
        box_y0, box_y1 = plot_y1 - 155, plot_y1 - 18
        draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=8, fill=BG_MAIN, outline=ACCENT_GOLD, width=2)
        draw.text((box_x0 + 20, box_y0 + 18), "PUNTO DE INFLEXIÓN (+30 AÑOS)", font=f_footer, fill=ACCENT_GOLD)
        draw.text((box_x0 + 20, box_y0 + 52), "• 912 semanas seguidas Top 10", font=f_label_sm, fill=TEXT_WHITE)
        draw.text((box_x0 + 20, box_y0 + 88), "• 18 años ganando ≥2 títulos ATP", font=f_label_sm, fill=ACCENT_STEEL)

    card_alpha = ease_out_cubic((sec - 7.8) / 1.0)
    draw_bottom_insight_card(
        draw,
        "L E C C I Ó N   D E   E S T R A T E G I A   •   B Y   J A I R O   Q U E L A L",
        "Cuando el volumen bruto ya no escala, optimizar el margen por partido",
        "mantiene a una leyenda en la cima. ¿A quién analizamos en el Episodio #02?",
        alpha=card_alpha
    )

    proc4.stdin.write(frame.tobytes())

proc4.stdin.close()
proc4.wait()

# Re-merge Master MP4
wav_path = os.path.join(inter_dir, "data_legends_soundtrack.wav")
concat_list_path = os.path.join(inter_dir, "clips_list.txt")
master_mp4_path = os.path.join(out_dir, "video_master_data_legends_ep01_nadal.mp4")
subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0", "-i", concat_list_path,
    "-i", wav_path,
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "128k",
    "-shortest",
    master_mp4_path
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Updated all 4 clips and Master MP4 cleanly!")
