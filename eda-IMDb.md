# 🎬 Análisis Exploratorio de Datos (EDA) del Dataset IMDb

---

## 0. Carga de Datos y Preprocesamiento Inicial

- Archivos cargados: Películas (85855 filas), Ratings (85855 filas).

- **Datasets unidos**. Total de filas: **85855**, Total de columnas: **70**.

## 1. Estructura y Descripción de Datos

### Tipos de Datos y Clasificación de Variables:

| Columna                   | Tipo de Dato   |
|:--------------------------|:---------------|
| votes_2                   | int64          |
| votes_10                  | int64          |
| votes                     | int64          |
| votes_9                   | int64          |
| votes_8                   | int64          |
| votes_7                   | int64          |
| votes_6                   | int64          |
| total_votes               | int64          |
| votes_4                   | int64          |
| votes_5                   | int64          |
| duration                  | int64          |
| votes_1                   | int64          |
| year                      | int64          |
| votes_3                   | int64          |
| males_0age_avg_vote       | float64        |
| males_allages_votes       | float64        |
| males_allages_avg_vote    | float64        |
| allgenders_45age_votes    | float64        |
| males_0age_votes          | float64        |
| males_18age_avg_vote      | float64        |
| males_18age_votes         | float64        |
| males_30age_votes         | float64        |
| allgenders_45age_avg_vote | float64        |
| us_voters_votes           | float64        |
| us_voters_rating          | float64        |
| top1000_voters_votes      | float64        |
| top1000_voters_rating     | float64        |
| females_45age_votes       | float64        |
| females_45age_avg_vote    | float64        |
| females_30age_votes       | float64        |
| males_30age_avg_vote      | float64        |
| females_30age_avg_vote    | float64        |
| females_18age_avg_vote    | float64        |
| females_0age_votes        | float64        |
| females_0age_avg_vote     | float64        |
| females_allages_votes     | float64        |
| females_allages_avg_vote  | float64        |
| males_45age_votes         | float64        |
| males_45age_avg_vote      | float64        |
| females_18age_votes       | float64        |
| allgenders_30age_votes    | float64        |
| non_us_voters_votes       | float64        |
| allgenders_18age_votes    | float64        |
| date_published            | datetime64[ns] |
| avg_vote                  | float64        |
| allgenders_30age_avg_vote | float64        |
| usa_gross_income          | float64        |
| worlwide_gross_income     | float64        |
| metascore                 | float64        |
| reviews_from_users        | float64        |
| budget                    | float64        |
| weighted_average_vote     | float64        |
| mean_vote                 | float64        |
| median_vote               | float64        |
| non_us_voters_rating      | float64        |
| allgenders_0age_avg_vote  | float64        |
| allgenders_0age_votes     | float64        |
| allgenders_18age_avg_vote | float64        |
| reviews_from_critics      | float64        |
| title                     | object         |
| original_title            | object         |
| genre                     | object         |
| country                   | object         |
| writer                    | object         |
| director                  | object         |
| production_company        | object         |
| actors                    | object         |
| description               | object         |
| language                  | object         |
| imdb_title_id             | object         |

## 2. Calidad de los Datos

### Valores Duplicados

- Filas completas duplicadas: **0**

### Reporte de Valores Nulos (Missing Values)

|                          | Total Nulos   | Porcentaje (%)   |
|:-------------------------|:--------------|:-----------------|
| usa_gross_income         | 85855         | 100              |
| worlwide_gross_income    | 85855         | 100              |
| budget                   | 85855         | 100              |
| metascore                | 72550         | 84.5             |
| females_0age_votes       | 63738         | 74.24            |
| females_0age_avg_vote    | 63738         | 74.24            |
| males_0age_avg_vote      | 58444         | 68.07            |
| males_0age_votes         | 58444         | 68.07            |
| allgenders_0age_avg_vote | 52496         | 61.14            |
| allgenders_0age_votes    | 52496         | 61.14            |
| reviews_from_critics     | 11797         | 13.74            |
| reviews_from_users       | 7597          | 8.85             |
| females_18age_votes      | 6521          | 7.6              |
| females_18age_avg_vote   | 6521          | 7.6              |
| date_published           | 4563          | 5.31             |

### Outliers (Valores Mínimos y Máximos Extremos)

- **duration**: Mínimo=41, Máximo=808.
- **budget**: No se pudo calcular min/max (posibles NaN).
- **worlwide_gross_income**: No se pudo calcular min/max (posibles NaN).
- **votes**: Mínimo=99, Máximo=2,278,845.

Gráfico: `duration_boxplot.png`

## 3. Estadísticas Descriptivas

### Estadísticas Descriptivas de Variables Numéricas Clave:

|           | count   | mean    | std     | min   | 25%   | 50%   | 75%    | 95%     | 99%    | max         | Asimetría   | Kurtosis   |
|:----------|:--------|:--------|:--------|:------|:------|:------|:-------|:--------|:-------|:------------|:------------|:-----------|
| avg_vote  | 85855   | 5.9     | 1.23    | 1     | 5.2   | 6.1   | 6.8    | 7.6     | 8.2    | 9.9         | -0.76       | 0.6        |
| duration  | 85855   | 100.35  | 22.55   | 41    | 88    | 96    | 108    | 142     | 171    | 808         | 3.08        | 40.3       |
| votes     | 85855   | 9493.49 | 53574.4 | 99    | 205   | 484   | 1766.5 | 33416.2 | 205858 | 2.27884e+06 | 14.62       | 325.26     |
| budget    | 0       | nan     | nan     | nan   | nan   | nan   | nan    | nan     | nan    | nan         | nan         | nan        |
| metascore | 13305   | 55.9    | 17.78   | 1     | 43    | 57    | 69     | 84      | 93     | 100         | -0.16       | -0.43      |

## 4. Análisis de la Variable Objetivo: avg_vote

Gráfico: `avg_vote_distribution.png`

- **Media del Rating:** 5.90
- **Mediana del Rating:** 6.10
- **Asimetría (Skewness):** -0.76

  -> El rating está **sesgado a la izquierda** (sesgo negativo).

## 5. Correlación de Texto con Rating

- **Correlación entre Longitud de Descripción y Rating:** **-0.0180**

  -> La correlación es muy cercana a cero.

## 6. Distribuciones Importantes

Gráfico: `avg_vote_by_year.png`

## 7. Análisis de Duración vs Rating

Gráfico: `duration_vs_rating_boxplot.png`

### Rating Promedio por Categoría de Duración:

| duration_bin   | avg_vote   |
|:---------------|:-----------|
| >200min        | 7.006      |
| 150-200min     | 6.516      |
| 120-150min     | 6.486      |
| 100-120min     | 6.176      |
| <80min         | 5.722      |
| 80-100min      | 5.601      |
