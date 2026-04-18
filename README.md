# 🤖 MLOps — API de Plataformas de Streaming

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render">
</p>

---

## 📋 Descripción

Proyecto de **Machine Learning Operations (MLOps)** que integra un pipeline completo de datos — desde la ingesta y transformación (ETL) hasta el análisis exploratorio (EDA) y el despliegue de una API REST — sobre datasets de las principales plataformas de streaming: **Amazon Prime Video**, **Disney Plus**, **Hulu** y **Netflix**.

La API permite realizar consultas sobre el catálogo unificado de contenido y además incluye un **sistema de recomendación content-based** construido con TF-IDF y similitud coseno.

---

## 🏗️ Arquitectura del Proyecto

```
Machine_Learning_Ops_API_Deploy/
│
├── API/
│   ├── main.py               # Aplicación FastAPI con los endpoints
│   ├── final.parquet          # Dataset procesado (formato Parquet)
│   └── requirements.txt      # Dependencias de la API
│
├── Datasets_originales/
│   ├── amazon_prime_titles.csv
│   ├── disney_plus_titles.csv
│   ├── hulu_titles.csv
│   └── netflix_titles.csv
│
├── ratings/                   # Archivos de ratings de usuarios (1.csv - 8.csv)
│
├── ETL.ipynb                  # Notebook con el proceso de ETL
├── EDA.ipynb                  # Notebook con el Análisis Exploratorio de Datos
├── Sistema de recomendacion.ipynb  # Notebook con el desarrollo del modelo
├── plataformas.csv            # Dataset consolidado de plataformas
├── render.yaml                # Configuración de deploy en Render
└── README.md
```

---

## ⚙️ Pipeline de Datos

### 1. ETL (`ETL.ipynb`)

El proceso de Extracción, Transformación y Carga realiza las siguientes operaciones:

| Paso | Descripción |
|------|-------------|
| **Carga** | Lee los 4 CSV de plataformas y los 8 archivos de ratings |
| **ID único** | Genera un campo `id` con prefijo por plataforma (`a`, `d`, `h`, `n`) |
| **Nulos en rating** | Reemplaza valores nulos del campo `rating` por `"G"` |
| **Fechas** | Convierte `date_added` al formato `YYYY-MM-DD` |
| **Normalización** | Convierte todos los campos de texto a minúsculas |
| **Duration** | Separa el campo `duration` en `duration_int` (entero) y `duration_type` (texto) |
| **Plataforma** | Crea el campo `plataform` derivado del prefijo del ID |
| **Merge** | Une los datasets de plataformas con los scores de usuarios |

### 2. EDA (`EDA.ipynb`)

Análisis exploratorio de datos que incluye visualizaciones y estadísticas descriptivas del dataset unificado.

### 3. Sistema de Recomendación (`Sistema de recomendacion.ipynb`)

Desarrollo y experimentación del modelo de recomendación basado en:

- **Collaborative Filtering**: Matriz usuario-película con similitud coseno entre usuarios.
- **Content-Based Filtering** (implementado en la API): TF-IDF sobre features de contenido (`cast`, `plataform`, `duration_type`, `release_year`).

---

## 🚀 API — Endpoints

La API está construida con **FastAPI** y desplegada en **Render**. Documentación interactiva disponible en `/docs`.

### `GET /`
Endpoint raíz con información general de la API.

---

### `GET /get_max_duration`
Devuelve el título de la película con la duración máxima.

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `year` | `int` | ❌ | Año de lanzamiento |
| `platform` | `str` | ❌ | Plataforma (`amazon`, `disney_plus`, `hulu`, `netflix`) |
| `duration_type` | `str` | ❌ | Tipo de duración (`min` o `season`) |

**Ejemplo:** `/get_max_duration?year=2020&platform=netflix&duration_type=min`

---

### `GET /get_score_count`
Cuenta la cantidad de películas por plataforma con un score mayor o igual al valor dado, filtrado por año.

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `platform` | `str` | ✅ | Plataforma |
| `scored` | `float` | ✅ | Score mínimo |
| `year` | `int` | ✅ | Año de lanzamiento |

**Ejemplo:** `/get_score_count?platform=netflix&scored=3.5&year=2019`

---

### `GET /get_count_platform`
Devuelve la cantidad de películas/series disponibles en una plataforma.

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `platform` | `str` | ✅ | Plataforma |

**Ejemplo:** `/get_count_platform?platform=amazon`

---

### `GET /get_actor`
Devuelve el actor con más apariciones en una plataforma y año determinados.

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `platform` | `str` | ✅ | Plataforma |
| `year` | `int` | ✅ | Año de lanzamiento |

**Ejemplo:** `/get_actor?platform=netflix&year=2018`

---

### `GET /get_recommendations`
🤖 **Sistema de recomendación content-based.** Recibe un título y devuelve las N películas/series más similares.

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `title` | `str` | ✅ | Título de la película o serie |
| `n` | `int` | ❌ | Cantidad de recomendaciones (default: 5, máx: 20) |

**Ejemplo:** `/get_recommendations?title=the godfather&n=5`

**Respuesta:**
```json
{
  "title_consultado": "the godfather",
  "recomendaciones": [
    {
      "title": "...",
      "plataform": "...",
      "score": 4.2,
      "release_year": 1995,
      "duration": "120 min",
      "similarity": 0.8732
    }
  ]
}
```

---

## 🛠️ Tech Stack

| Tecnología | Uso |
|------------|-----|
| **Python 3.11** | Lenguaje principal |
| **FastAPI** | Framework para la API REST |
| **Uvicorn** | Servidor ASGI |
| **Pandas** | Manipulación y análisis de datos |
| **NumPy** | Operaciones numéricas |
| **Scikit-learn** | TF-IDF Vectorizer y similitud coseno |
| **FastParquet** | Lectura de archivos `.parquet` |
| **Render** | Plataforma de deployment |

---

## 🖥️ Instalación y Ejecución Local

### Prerrequisitos
- Python 3.10+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/facuberon/Machine_Learning_Ops_API_Deploy.git
cd Machine_Learning_Ops_API_Deploy

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r API/requirements.txt

# 4. Ejecutar la API
cd API
uvicorn main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000` y la documentación interactiva en `http://127.0.0.1:8000/docs`.

---

## ☁️ Deploy en Render

El proyecto incluye un archivo `render.yaml` preconfigurado para deployment automático en [Render](https://render.com):

```yaml
services:
  - type: web
    name: mlops-api
    runtime: python
    rootDir: API
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    plan: free
```

Para desplegar, conectá tu repositorio de GitHub a Render y el servicio se configurará automáticamente.

---

## 📊 Datasets

Los datos provienen de las siguientes plataformas de streaming:

| Plataforma | Archivo | Registros aprox. |
|------------|---------|-----------------|
| Amazon Prime Video | `amazon_prime_titles.csv` | ~9.700 |
| Disney Plus | `disney_plus_titles.csv` | ~1.450 |
| Hulu | `hulu_titles.csv` | ~3.070 |
| Netflix | `netflix_titles.csv` | ~8.800 |

Además se incluyen **8 archivos de ratings** con calificaciones de usuarios para el sistema de recomendación.

---

