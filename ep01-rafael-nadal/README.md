# DATA LEGENDS • Episodio #01: Rafael Nadal
**El Algoritmo de una Anomalía Estadística**  
*Autor: **Jairo Quelal** | Data Science & Sports Analytics*

---

## Descripción del Proyecto

**DATA LEGENDS** es una serie semanal de Ciencia de Datos y *Sports Analytics* donde auditamos la carrera de las mayores leyendas del deporte utilizando estadística avanzada, reducción de dimensionalidad (**PCA**), aprendizaje no supervisado (**K-Means**) y detección de anomalías (**Z-Score**) en Python.

En este **Episodio #01**, procesamos los **1,308 partidos individuales de Rafael Nadal en el circuito ATP** (1,080 victorias y 228 derrotas entre 2003 y 2024; 15,259 juegos de saque y 15,102 juegos al resto) para responder tres preguntas clave:
1. **¿Qué variable explica realmente su 90.5% de efectividad en tierra batida?** (*Feature Importance: Service Hold % vs. Return Break %*)
2. **¿Cómo evolucionó tácticamente a lo largo de 23 temporadas para sobrevivir 912 semanas consecutivas en el Top 10?** (*PCA via SVD + K-Means Clustering*)
3. **¿Qué tan grande es su anomalía estadística en Roland Garros comparándolo contra su propia media de élite?** (*Z-Score Outlier Detection*)

---

## Principales Hallazgos Técnicos

| Modelo / Técnica | Hallazgo Estadístico Clave | Métrica Oficial |
| :--- | :--- | :--- |
| **1. Paradoja de Superficie (Saque vs. Resto)** | La arcilla fue estadísticamente su **peor** superficie reteniendo el saque (`84%` vs `86%` en dura y `89%` en hierba). Su dominio nació al resto: quebró casi la mitad de todos los turnos de saque rivales. | **43% Juegos Ganados al Resto** en arcilla (+19 pts % vs. hierba) y **48% de conversión** sobre **4,914 Break Points**. |
| **2. PCA (SVD) + K-Means (`k=3`)** | Los 2 primeros Componentes Principales capturan el **81.8% de la varianza** (`PC1 = 55.8%` volumen/títulos; `PC2 = 26.0%` edad/eficiencia). El modelo separa su juventud de alto volumen de su reinvención tras cumplir 30 años. | **-39% reducción de partidos/año** en el *Cluster 2* (`83 → 50 part/año`), manteniendo **85.4% de victorias** y sumando **+8 Grand Slams**. |
| **3. Detección de Anomalías (`Z-Score`)** | Evaluando sus 14 torneos más exigentes (4 Grand Slams + 9 Masters 1000 + ATP Finals), su media de élite es `79.9%` (`σ = 11.15%`). Roland Garros rompe la campana estadística. | **+1.49σ en Roland Garros** (`96.6%` \| `112–4` \| `14–0` en finales \| `90` victorias en 3 sets corridos). |
| **4. Series Temporales y Longevidad** | Cuando el volumen físico dejó de escalar después de 2016, optimizó su margen operativo por partido, logrando el mejor saque de su vida a los 33 años. | **90% juegos ganados al saque en 2019** (récord personal) y **912 semanas seguidas en el Top 10** (2005–2023). |

---

## Estructura del Repositorio

```text
data-legends-ep01-rafael-nadal/
├── data/
│   ├── nadal_seasons_2003_2024.csv          # Estadísticas año por año (2003-2024) por superficie y ranking
│   ├── nadal_big_tournaments_zscore.csv     # Rendimiento y Z-Score en los 14 grandes torneos ATP
│   └── nadal_surface_metrics.csv            # Métricas oficiales de saque (Hold %) y resto (Break %)
├── src/
│   ├── generate_data_legends_charts.py      # Pipeline PCA (SVD), K-Means, Z-Score y generación de láminas
│   └── generate_video_animations.py         # Motor de renderizado de animaciones de datos en video (MP4)
├── assets/
│   └── slides/                              # Las 5 láminas en alta resolución (1440x1920) y el Carrusel PDF
├── requirements.txt
└── README.md
```

---

## Cómo Reproducir el Análisis y las Visualizaciones

1. Clona este repositorio e instala las dependencias:
```bash
git clone https://github.com/jairoequelal-ctrl/data-legends.git
cd data-legends/ep01-rafael-nadal
pip install -r requirements.txt
```

2. Ejecuta el pipeline analítico y de visualización:
```bash
python3 src/generate_data_legends_charts.py
```

---

## Fuentes de Datos Verificadas

* **ATP Tour Official Player Profile & Stats (Rafael Nadal):** Registros oficiales de victorias/derrotas (`1,080–228`), títulos (`92`), semanas en el Top 10 (`912`) y métricas de servicio/resto (`15,259` juegos de saque y `15,102` de resto).
* **Roland-Garros Official Records:** Desglose histórico en París (`112–4`, `14–0` en finales, `90` victorias en sets corridos y `4` títulos sin ceder sets: 2008, 2010, 2017 y 2020).
* **Historical Singles Performance Timelines (2003–2024):** Matrices anuales por torneo de Grand Slam y Masters 1000.

---

*Proyecto creado por **Jairo Quelal** para la serie semanal **DATA LEGENDS**.*
