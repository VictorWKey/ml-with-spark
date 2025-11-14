# 🎬 IMDB Rating Prediction - Comparación: Data Leakage vs Modelo Limpio

Este proyecto demuestra la diferencia entre un modelo con **data leakage** (R²=0.86) y un modelo **limpio** (R²=0.24).

---

## 🚀 Inicio Rápido

Los archivos del proyecto están en la carpeta `ml-prediction/`. **IMPORTANTE:** Debes ejecutar los modelos desde dentro de esa carpeta:

```bash
# Cambiar al directorio del proyecto
cd ml-prediction

# Luego ejecutar los comandos de Spark como se indica abajo
```

**Nota:** Los archivos Scala buscan los datos en `../IMDB-Movies-Extensive-Dataset-Analysis/data1/`, por lo que es fundamental ejecutarlos desde la carpeta `ml-prediction/`.

---

## 📊 Preparación de Datos

Antes de ejecutar los modelos, necesitas descargar el dataset. Este proyecto utiliza el **IMDB Movies Extensive Dataset**:

### Opción 1: Clonar el repositorio completo
```bash
# Desde la raíz del proyecto
git clone https://github.com/sahildit/IMDB-Movies-Extensive-Dataset-Analysis.git IMDB-Movies-Extensive-Dataset-Analysis
```

### Opción 2: Descargar manualmente
1. Ve a: https://github.com/datasciencedojo/datasets
2. Navega a la carpeta `raw/IMDb Movies Extensive Dataset`
3. Descarga los archivos:
   - `IMDb movies.csv`
   - `IMDb ratings.csv`
4. Crea la estructura de carpetas: `IMDB-Movies-Extensive-Dataset-Analysis/data1/`
5. Coloca los archivos CSV en esa carpeta

### Verificación
Después de descargar, verifica que tengas esta estructura:
```
[raíz del proyecto]/
├── IMDB-Movies-Extensive-Dataset-Analysis/
│   └── data1/
│       ├── IMDb movies.csv
│       └── IMDb ratings.csv
├── ml-prediction/
│   ├── IMDBPredictionModelWithDataLeakage.scala   # ❌ Modelo CON cheating
│   ├── IMDBPredictionModelClean.scala         # ✅ Modelo SIN cheating
│   └── resultados/                           # 📊 Outputs de ambos modelos
└── README.md (este archivo)
```

---

## ⚠️ MODELO CON CHEATING (Data Leakage)

**Archivo:** `IMDBPredictionModelWithDataLeakage.scala`

### ¿Cómo ejecutar?
```bash
# Iniciar Spark Shell
spark-shell \
  --driver-memory 10g \
  --executor-memory 10g \
  --conf spark.sql.shuffle.partitions=100

# Cargar y ejecutar modelo
:load IMDBPredictionModelWithDataLeakage.scala
IMDBPredictionModelWithDataLeakage.main(Array())
```

### Resultados (CON CHEATING)
| Modelo | R² | RMSE | MAE | Feature Dominante |
|--------|-----|------|-----|-------------------|
| Ridge | 0.81 | 0.539 | 0.399 | actors_encoded (54%) |
| Random Forest | 0.83 | 0.503 | 0.362 | actors_encoded (63%) |
| **GBT** | **0.86** | **0.463** | **0.336** | **actors_encoded (74%)** |
| Ensemble | 0.85 | 0.476 | 0.341 | actors_encoded (69%) |

### ❌ Problema Identificado: Target Encoding
```scala
// ESTO ES CHEATING - codifica usando mean(target)
val actorAvgRating = movieData.groupBy("actor").agg(avg("avg_vote"))
// actors_encoded ≈ avg_vote (correlación >0.90)
// El modelo simplemente "copia" el target en lugar de predecir
```

**¿Por qué es cheating?**
- Target Encoding usa `mean(avg_vote)` por categoría
- Crea correlación circular: `actors_encoded ⟷ avg_vote ≈ 0.90`
- El modelo aprende: `prediction = actors_encoded` (copia, no predicción)
- 74% de feature importance en una sola variable → señal de alarma

---

## ✅ MODELO LIMPIO (Sin Data Leakage)

**Archivo:** `IMDBPredictionModelClean.scala`

### ¿Cómo ejecutar?
```bash
# Iniciar Spark Shell (misma configuración)
spark-shell \
  --driver-memory 10g \
  --executor-memory 10g \
  --conf spark.sql.shuffle.partitions=100

# Cargar y ejecutar modelo limpio
:load IMDBPredictionModelClean.scala
IMDBPredictionModelClean.main(Array())
```

### Resultados (SIN CHEATING)
| Modelo | R² | RMSE | MAE | Features Balanceadas |
|--------|-----|------|-----|----------------------|
| Ridge | 0.20 | 0.727 | 0.588 | ✅ Distribuidas |
| Random Forest | 0.22 | 0.717 | 0.576 | ✅ Distribuidas |
| **GBT** | **0.24** | **0.706** | **0.568** | ✅ **Distribuidas** |
| Ensemble | 0.23 | 0.711 | 0.571 | ✅ Distribuidas |

### ✅ Solución Implementada: Frequency Encoding
```scala
// ESTO ES VÁLIDO - codifica usando frecuencia de aparición
val actorFrequency = movieData.groupBy("actor").count()
val totalMovies = movieData.count()
actors_freq = count / totalMovies  // No usa target
// Captura "popularidad" sin usar avg_vote
```

**¿Por qué es válido?**
- Frequency Encoding usa solo `count(appearances)`, no el target
- No hay correlación circular con `avg_vote`
- Features distribuidas (ninguna domina >30%)
- R²=0.24 es realista para este problema

---

## � Comparación Lado a Lado

| Aspecto | CON Cheating | SIN Cheating |
|---------|-------------|--------------|
| **R² (GBT)** | 0.86 | 0.24 |
| **Feature Encoding** | Target Encoding | Frequency Encoding |
| **actors_encoded correlación** | >0.90 | <0.30 |
| **Feature Importance** | actors_encoded: 74% | Distribuida: max 15% |
| **Validez** | ❌ Inválido | ✅ Válido |
| **Uso en producción** | ❌ No funciona | ✅ Sí funciona |

---

## 🎓 Lección Aprendida

**Target Encoding = Data Leakage sutil:**
- ✅ Funciona bien en **time series** (si filtras tiempo: solo pasado → futuro)
- ❌ NO funciona en **cross-sectional** (películas sin orden temporal)
- ⚠️ Síntoma: R² "demasiado bueno", feature importance desequilibrada

**Alternativas válidas a Target Encoding:**
1. **Frequency Encoding** - Cuenta apariciones (usado en Clean)
2. **Leave-One-Out Encoding** - Excluye fila actual del cálculo
3. **K-Fold Target Encoding** - Usa cross-validation para evitar leakage

---

## 📁 Archivos del Proyecto

```
ml-prediction/
├── IMDBPredictionModelWithDataLeakage.scala   # ❌ Modelo CON cheating (R²=0.86)
├── IMDBPredictionModelClean.scala         # ✅ Modelo SIN cheating (R²=0.24)
└── resultados/                           # 📊 Outputs de ambos modelos
    ├── with_data_leakage_*_predictions.txt      # Resultados modelo WithDataLeakage
    └── clean_*_predictions.txt            # Resultados modelo Clean
```

README.md (este archivo) está en la raíz del proyecto.

---

## 👨‍💻 Autor

**Victor W. Key**
- Dataset: IMDB Movies Extensive Dataset (85k películas)
- Framework: Apache Spark 3.3.1 + SparkML
- Lección: Data Leakage puede ser sutil pero devastador

---

**Conclusión:** Siempre verifica feature importance y correlaciones. Si algo parece "demasiado bueno para ser verdad", probablemente lo es.
