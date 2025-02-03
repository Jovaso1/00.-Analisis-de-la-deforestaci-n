import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cargar_datos(archivo=None, url=None):
    """
    Carga los datos desde un archivo CSV o una URL, convierte las columnas de tipo `object` a `str` o `category`,
    y asegura la compatibilidad de las columnas numéricas con Arrow.

    Args:
        archivo (str): Ruta del archivo CSV local.
        url (str): URL del archivo CSV en línea.

    Returns:
        pd.DataFrame: Datos cargados con tipos de datos compatibles con Arrow.
    """
    if archivo:
        df = pd.read_csv(archivo)
    elif url:
        df = pd.read_csv(url)
    else:
        return None
    
    # Normalizar los nombres de las columnas (eliminar espacios y convertir a minúsculas)
    df.columns = df.columns.str.strip().str.lower()
    
    # Verificar los nombres de las columnas disponibles
    st.write("Columnas disponibles:", df.columns.tolist())
    
    # Convertir las columnas de tipo 'object' a 'str' o 'category'
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('str')  # Convertir a str, o usa 'category' si aplica
    
    # Asegurarnos de que las columnas numéricas como latitud y longitud sean de tipo float
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')  # Convertir a numérico y manejar errores
    
    # Limpiar valores faltantes mediante interpolación
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    
    return df

def generar_mapa(df):
    """
    Genera un mapa interactivo basado en las coordenadas de latitud y longitud y otras variables seleccionadas.

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.
    """
    # Selección de columnas disponibles para las variables
    lat_col = st.selectbox("Selecciona la columna de latitud", df.columns.tolist())
    lon_col = st.selectbox("Selecciona la columna de longitud", df.columns.tolist())
    
    # Verificar si las columnas de latitud y longitud están disponibles
    if lat_col not in df.columns or lon_col not in df.columns:
        st.error("Las columnas seleccionadas no existen en los datos")
        return
    
    # Selección de la columna adicional para personalizar el mapa
    var_col = st.selectbox("Selecciona la columna para visualizar en el mapa", df.columns.tolist())
    
    # Crear GeoDataFrame con las coordenadas seleccionadas
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))

    # Plotear el mapa
    gdf.plot(column=var_col, legend=True, figsize=(10, 8))
    st.pyplot()

def analisis_cluster(df):
    """
    Realiza un análisis de clúster para las superficies deforestadas, utilizando KMeans.

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.
    """
    # Seleccionar columnas para el análisis
    variables = st.multiselect("Selecciona las variables para el análisis de clúster", df.columns.tolist())
    
    # Verificar si hay al menos dos variables seleccionadas
    if len(variables) < 2:
        st.error("Se deben seleccionar al menos dos variables para el análisis de clúster.")
        return
    
    # Preprocesar los datos para el análisis
    df_cluster = df[variables].dropna()  # Eliminar filas con valores faltantes
    
    # Estandarizar los datos
    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)
    
    # Realizar el análisis de clúster con KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_cluster_scaled)
    
    # Visualizar los clústeres
    st.write("Clústeres identificados:", df['cluster'].value_counts())
    
    # Graficar los clústeres
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[variables[0]], df[variables[1]], c=df['cluster'], cmap='viridis')
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_title("Análisis de Clúster de Deforestación")
    st.pyplot(fig)

def grafico_torta(df):
    """
    Genera un gráfico de torta que muestra la distribución de tipos de vegetación.

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.
    """
    # Seleccionar la columna para el gráfico de torta
    veg_col = st.selectbox("Selecciona la columna de tipo de vegetación", df.columns.tolist())
    
    # Generar el gráfico de torta
    vegetacion = df[veg_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(vegetacion, labels=vegetacion.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Asegurar que el gráfico sea circular
    ax.set_title("Distribución de Tipos de Vegetación")
    st.pyplot(fig)

def app():
    """
    Función principal de la aplicación.
    Permite cargar los datos, elegir variables para análisis y generar mapas y gráficos.
    """
    st.title("Análisis de la Deforestación")
    
    # Subir archivo CSV
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    # Cargar datos
    if archivo:
        df = cargar_datos(archivo=archivo)
        
        # Mostrar los primeros registros
        st.write("Primeros registros de los datos", df.head())
        
        # Mostrar tipos de datos
        st.write("Tipos de datos de las columnas:", df.dtypes)
        
        # Generar mapa
        if df is not None:
            generar_mapa(df)
            
        # Realizar análisis de clúster
        if df is not None:
            analisis_cluster(df)
        
        # Mostrar gráfico de torta
        if df is not None:
            grafico_torta(df)

if __name__ == "__main__":
    app()
