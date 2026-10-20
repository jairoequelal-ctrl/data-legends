# DATA LEGENDS • Sports Analytics & Machine Learning
**Serie semanal de Ciencia de Datos aplicada a las mayores leyendas del deporte**  
*Autor: **Jairo Quelal** ([@jairoequelal-ctrl](https://github.com/jairoequelal-ctrl))*

---

## Sobre el Proyecto

**DATA LEGENDS** es un repositorio central donde cada semana audito desde cero la carrera de una leyenda del deporte utilizando **Python**, **estadística avanzada**, **reducción de dimensionalidad (PCA vía SVD)**, **clustering no supervisado (K-Means)** y **detección de anomalías (Z-Score)**.

Cada episodio cuenta con su propia carpeta que incluye los datasets limpios en formato `.csv`, el código fuente reproducible en Python y las visualizaciones editoriales en alta resolución.

---

## Índice de Episodios

| Episodio | Deportista / Leyenda | Modelos Aplicados | Hallazgo Principal | Carpeta del Proyecto |
| :--- | :--- | :--- | :--- | :--- |
| **#01** | **Rafael Nadal** *(Tenis ATP)* | `Feature Importance`, `PCA (SVD)`, `K-Means (k=3)`, `Z-Score` | Su 90.5% en arcilla nació al resto (**43% de quiebres**, 4,914 *break points*); **+1.49σ** en Roland Garros (`112–4`); **-39% de carga** tras los 30 años para ganar 8 Grand Slams más. | [`./ep01-rafael-nadal`](./ep01-rafael-nadal/) |
| **#02** | *Próxima semana (En votación)* | *Por definir* | *Próximamente* | — |

---

## Estructura del Repositorio (`data-legends`)

```text
data-legends/
├── ep01-rafael-nadal/
│   ├── data/
│   │   ├── nadal_seasons_2003_2024.csv          # Estadísticas anuales (2003–2024) por superficie y ranking
│   │   ├── nadal_big_tournaments_zscore.csv     # Efectividad y Z-Score en los 14 grandes torneos ATP
│   │   └── nadal_surface_metrics.csv            # Métricas oficiales de saque (Hold %) y resto (Break %)
│   ├── src/
│   │   ├── generate_data_legends_charts.py      # Pipeline PCA (SVD), K-Means, Z-Score y gráficas
│   │   └── generate_video_animations.py         # Motor de renderizado de animaciones de datos en MP4
│   ├── assets/
│   │   └── slides/                              # Las 5 láminas (1440x1920) y el Carrusel PDF
│   └── README.md                                # Documentación técnica detallada del Episodio #01
├── requirements.txt
└── README.md
```

---

## Instalación y Uso Rápido

```bash
git clone https://github.com/jairoequelal-ctrl/data-legends.git
cd data-legends
pip install -r requirements.txt
python3 ep01-rafael-nadal/src/generate_data_legends_charts.py
```

---

## Fuentes Oficiales Verificadas

* **ATP Tour Official Player Stats:** [Rafael Nadal Player Stats](https://www.atptour.com/en/players/rafael-nadal/n409/player-stats)
* **Roland-Garros Official Records:** [Nadal by the Numbers — 14 Titles](https://www.rolandgarros.com/en-us/article/nadal-by-the-numbers-rg-14-titles)
* **Historical Singles Performance Timelines:** [Rafael Nadal Career Statistics](https://en.wikipedia.org/wiki/Rafael_Nadal_career_statistics)
