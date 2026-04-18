
from fastapi import FastAPI
import pandas as pd
import numpy as np
import fastparquet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import os

# Obtener la ruta absoluta del directorio actual donde está main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "final.parquet")

# Cargo los datos del Dataset final con la variable df
df = pd.read_parquet(PARQUET_PATH)
#creacion de una aplicacion
# aumenta el tiempo de espera a 60 segundos para todas las rutas
app = FastAPI(
    title="MLOps API - Plataformas de Streaming",
    description="API para consultar datos de películas y series de plataformas de streaming, con sistema de recomendación.",
    version="2.0.0",
)

@app.get("/")
async def root():
    return {
        "message": "Bienvenido a la API de MLOps - Plataformas de Streaming",
        "docs": "/docs",
        "endpoints": [
            "/get_max_duration",
            "/get_score_count",
            "/get_count_platform",
            "/get_actor",
            "/get_recommendations",
        ],
    }

@app.get("/get_max_duration")
async def get_max_duration(year:int= None, platform: str = None, duration_type: str = None):
    #Esta funcion devuelve el título de la película con la duración máxima.
    #Recibe los parametros year, platform y duration_type (siendo las tres opcionales)
    df = pd.read_parquet(PARQUET_PATH)
      #filtro por parametros 
    if year:
        df = df[df['release_year'] == year]
    
    if platform:
        df = df[df['plataform'] == platform]
    
    if duration_type:
        if duration_type == 'min':
            max_duration = df.sort_values(by='duration_type', ascending=True).iloc[0]['title']
        elif duration_type == 'season':
            max_duration = df.sort_values(by='duration_type', ascending=True).iloc[0]['title']
        else:
            return {'error': 'Invalid duration type'}

  #Si no se coloca parametro devuelve el titulo con maxima duracion
    else:
        max_duration = df.loc[df['duration_int'].idxmax(), 'title']

  # obtengo el título de la película con duración máxima
        
    return {'title': max_duration}

@app.get("/get_score_count")
async def get_score_count(platform: str, scored: float, year: int):
    #Esta funcion cuenta la cantidad de películas que cumplen con los criterios 
    #ingresados en "platform, scored, year" y muestra el total.
    #Resive los parametros platform, scored, year (que no son opcionales)

    #selecciona las peliculas que cumplan con el criterio de los parametros
    selec = df.loc[(df['plataform'] == platform) & (df['score_y'] >= scored) & (df['release_year'] == year)]

    #contar las peliculas y retornar el resultado
    contar = selec['title']

    #Seteo los duplicados y los cuento con len
    contar= set(contar)

    return len(contar)


@app.get("/get_count_platform")
async def get_count_platform(platform: str):
    #Esta Funcion filtra el Dataset por la plataforma especificada, luego las
    #setea para evitar duplicados y retorna la cantidad de películas.
    #Recibe el parametro platform(no opcional)

    #filtro las películas a la plataforma especificada
    peliculas_filtradas = df[df["plataform"] == platform]
  
    #cuento los titulos de la plataforma filtrada
    contar = peliculas_filtradas['title']

    #Seteo la cantidad de peliculas encontradas y retorno el numero total sin duplicados
    cantidad_peliculas= set(contar)

    return len(cantidad_peliculas)

@app.get("/get_actor")
async def get_actor(platform: str, year: int):
    #Esta Funcion filtra el Dataset por la plataforma especificada y 
    # el año de lanzamiento, luego retorna el nombre del actor mas repetido.
    #Recibe los parametros platform y year (no son opcionales)

    # Filtro por plataforma y año
    filtro = df[(df['plataform'] == platform) & (df['release_year'] == year)]

    #Reemplazo los valores nulos en la columna "cast" por "ningun actor"
    filtro['cast'].fillna(value='ningun actor', inplace=True)

    # Creo una lista de actores
    actores = filtro['cast'].str.split(', ')

    # Creo un Dataset a partir de la lista de actores
    actores = pd.DataFrame({'actor': [actor for actors in actores for actor in actors]})

    # Filtro por actores distintos
    actores_filtro = actores[actores['actor'] != 'ningun actor'].groupby('actor').size().reset_index(name='count')

    # Ordeno de mayor a menor y obtengo el nombre del actor con más apariciones
    actor = actores_filtro.sort_values('count', ascending=False)['actor'].iloc[0]

    return actor


# ============================================================
# SISTEMA DE RECOMENDACIÓN (Content-Based Filtering)
# ============================================================
# Utiliza TF-IDF sobre las características de contenido (cast, 
# plataforma, tipo de duración) combinadas con el score para
# recomendar títulos similares de forma eficiente.
# ============================================================

def _build_content_features(dataframe: pd.DataFrame) -> pd.Series:
    """Construye una columna de texto combinando las features de contenido
    para crear el perfil de cada título."""
    cast_clean = dataframe['cast'].fillna('')
    platform = dataframe['plataform'].fillna('')
    duration = dataframe['duration_type'].fillna('')
    year_bin = (dataframe['release_year'] // 5 * 5).astype(str)  # agrupa por lustros
    
    return (cast_clean + ' ' + platform + ' ' + duration + ' ' + year_bin)


# Pre-computo la matriz TF-IDF al inicio para respuestas rápidas
_df_rec = df.drop_duplicates(subset='title').reset_index(drop=True)
_content = _build_content_features(_df_rec)
_tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
_tfidf_matrix = _tfidf.fit_transform(_content)
_title_to_idx = {title: idx for idx, title in enumerate(_df_rec['title'])}


@app.get("/get_recommendations")
async def get_recommendations(title: str, n: int = 5):
    """Sistema de recomendación content-based.
    
    Recibe el título de una película/serie y devuelve las N más similares
    basándose en cast, plataforma, tipo y época de lanzamiento.
    
    Parámetros:
    - title: título de la película o serie (debe existir en el dataset)
    - n: cantidad de recomendaciones (por defecto 5, máximo 20)
    """
    # Limito n para evitar respuestas enormes
    n = min(n, 20)
    
    title_lower = title.lower().strip()
    
    if title_lower not in _title_to_idx:
        # Buscar coincidencias parciales
        matches = [t for t in _title_to_idx.keys() if title_lower in t]
        if not matches:
            return {
                'error': f'Título "{title}" no encontrado.',
                'sugerencia': 'Verificá que el título esté escrito correctamente (en minúsculas).'
            }
        # Usar la primera coincidencia parcial
        title_lower = matches[0]
    
    idx = _title_to_idx[title_lower]
    
    # Calculo similitud coseno solo contra el título seleccionado (eficiente)
    sim_scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    
    # Combino con el score de rating para dar peso a películas mejor valoradas
    score_weight = _df_rec['score_y'].fillna(0).values / 5.0  # normalizo a [0,1]
    combined = sim_scores * 0.8 + score_weight * 0.2
    
    # Obtengo los N indices más similares (excluyendo el título mismo)
    similar_indices = combined.argsort()[::-1][1:n+1]
    
    recommendations = []
    for i in similar_indices:
        row = _df_rec.iloc[i]
        recommendations.append({
            'title': row['title'],
            'plataform': row['plataform'],
            'score': round(float(row['score_y']), 2) if pd.notna(row['score_y']) else None,
            'release_year': int(row['release_year']),
            'duration': f"{row['duration_int']} {row['duration_type']}",
            'similarity': round(float(combined[i]), 4),
        })
    
    return {
        'title_consultado': title_lower,
        'recomendaciones': recommendations
    }
