import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Función para cargar datos
def cargar_datos(archivo=None, url=None):
    """
    Carga los datos desde un archivo CSV o una URL.

    Args:
        archivo (str): Ruta del archivo CSV local.
        url (str): URL del archivo CSV en línea.

    Returns:
        pd.DataFrame: Datos cargados.
    """
    if archivo:
        df = pd.read_csv(archivo)
    elif url:
        df = pd.read_csv(url)
    else:
        return None
    
    # Limpiar valores faltantes
    df = df.dropna()
    return df

# Función para generar mapa por variable
def generar_mapa_variable(df, variable):
    """
    Genera un mapa de las zonas deforestadas según una variable seleccionada (vegetación, altitud, etc.).

    Args:
        df (pd.DataFrame): El dataframe con la información de deforestación.
        variable (str): Nombre de la variable a mapear (vegetación, altitud, etc.).
    """
    if variable in df.columns:
        # Verificar si la variable es numérica o categórica
        if pd.api.types.is_numeric_dtype(df[variable]):
            # Generar mapa de color continuo si la variable es numérica
            fig, ax = plt.subplots()
            df.plot.scatter(x='longitud', y='latitud', c=variable, cmap='viridis', ax=ax, alpha=0.7)
            ax.set_title(f'Zonas Deforestadas por {variable}')
            st.pyplot(fig)
        else:
            # Generar mapa de color discreto si la variable es categórica
            fig, ax = plt.subplots()
            df[variable] = df[variable].astype('category')
            df.plot.scatter(x='longitud', y='latitud', c=variable, cmap='Set1', ax=ax, alpha=0.7)
            ax.set_title(f'Zonas Deforestadas por {variable}')
            st.pyplot(fig)

# Función para generar gráfico de torta según tipo de vegetación
def generar_grafico_torta(df, columna):
    """
    Genera un gráfico de torta para una columna categórica.

    Args:
        df (pd.DataFrame): El dataframe con la columna categórica.
        columna (str): Nombre de la columna categórica.
    """
    categoria_counts = df[columna].value_counts()
    fig, ax = plt.subplots()
    categoria_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    ax.set_title(f'Distribución de {columna}')
    st.pyplot(fig)

# Función para análisis de clúster
def analisis_cluster(df, variables):
    """
    Realiza un análisis de clúster sobre las superficies deforestadas.

    Args:
        df (pd.DataFrame): El dataframe con los datos de superficie deforestada.
        variables (list): Lista de columnas a usar para el análisis de clúster.
    """
    # Seleccionamos las variables de interés
    df_cluster = df[variables].dropna()

    # Escalamos las variables
    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    # Aplicamos KMeans
    kmeans = KMeans(n_clusters=3)  # Número de clústeres (puedes ajustarlo)
    df['cluster'] = kmeans.fit_predict(df_cluster_scaled)

    # Mostrar clústeres en un mapa
    fig, ax = plt.subplots()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitud'], df['latitud']))
    gdf.plot(column='cluster', cmap='viridis', legend=True, ax=ax)
    ax.set_title('Clúster de Zonas Deforestadas')
    st.pyplot(fig)

# Configuración de la interfaz de Streamlit
def app():
    """
    Función principal que maneja la interfaz y el flujo de la aplicación.
    """
    st.title('Análisis de Deforestación con Imágenes de Satélite')

    df = None  # Asegurarse de que df esté inicializado

    # Opción para cargar archivo o leer desde URL
    opcion_carga = st.selectbox("Seleccione cómo cargar los datos:", ['Cargar archivo', 'Leer desde URL'])
    
    if opcion_carga == 'Cargar archivo':
        archivo = st.file_uploader("Subir archivo CSV", type='csv')
        if archivo:
            df = cargar_datos(archivo=archivo)
            st.write("Datos cargados:", df.head())
            st.write("Tipos de datos de cada columna:", df.dtypes)
    elif opcion_carga == 'Leer desde URL':
        url = st.text_input("Introduzca la URL del archivo CSV")
        if url:
            df = cargar_datos(url=url)
            st.write("Datos cargados:", df.head())
            st.write("Tipos de datos de cada columna:", df.dtypes)

    if df is not None:
        # Selección de análisis a realizar
        analisis_seleccionados = st.multiselect(
            'Seleccione los análisis a realizar:',
            ['Generar mapa por variable', 'Generar gráfico de torta', 'Análisis de clúster'],
            default=['Generar mapa por variable']
        )

        # Generar análisis según selección
        if 'Generar mapa por variable' in analisis_seleccionados:
            variable_seleccionada = st.selectbox("Seleccione la variable para el mapa:", ['vegetacion', 'altitud', 'precipitacion'])
            generar_mapa_variable(df, variable_seleccionada)

        if 'Generar gráfico de torta' in analisis_seleccionados:
            columna_torta = st.selectbox("Seleccione la columna para el gráfico de torta:", df.columns)
            generar_grafico_torta(df, columna_torta)

        if 'Análisis de clúster' in analisis_seleccionados:
            variables_cluster = st.multiselect("Seleccione las variables para el análisis de clúster:", df.columns)
            if len(variables_cluster) > 0:
                analisis_cluster(df, variables_cluster)

if __name__ == "__main__":
    app()
