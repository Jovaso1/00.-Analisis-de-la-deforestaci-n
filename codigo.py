import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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
        return pd.read_csv(archivo)
    elif url:
        return pd.read_csv(url)
    return None

# Función para realizar interpolación lineal
def interpolar_datos(df):
    """
    Realiza la interpolación lineal para los valores faltantes en el dataframe.

    Args:
        df (pd.DataFrame): El dataframe con valores faltantes.

    Returns:
        pd.DataFrame: El dataframe con los valores faltantes interpolados.
    """
    return df.interpolate(method='linear', axis=0)

# Función para generar mapa
def generar_mapa(df):
    """
    Genera un mapa con GeoPandas a partir de las coordenadas de latitud y longitud seleccionadas por el usuario.

    Args:
        df (pd.DataFrame): Datos con columnas de latitud y longitud.
    """
    # Pedir al usuario seleccionar las columnas de latitud y longitud
    lat_col = st.selectbox("Seleccione la columna de latitud:", df.columns)
    lon_col = st.selectbox("Seleccione la columna de longitud:", df.columns)

    if lat_col and lon_col:
        try:
            # Asegurarse de que las columnas sean numéricas
            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

            # Eliminar filas con valores nulos
            df = df.dropna(subset=[lat_col, lon_col])

            # Crear un GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))
            gdf.plot(marker='o', color='red', markersize=5)
            plt.show()
        except KeyError as e:
            st.error(f"Error al crear el mapa: {e}")



# Función para crear gráfico de torta
def generar_grafico_torta(df, columna):
    """
    Genera un gráfico de torta para una columna categórica.

    Args:
        df (pd.DataFrame): El dataframe con la columna categórica.
        columna (str): Nombre de la columna categórica.
    """
    categoria_counts = df[columna].value_counts()
    categoria_counts.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.ylabel('')
    plt.title(f'Distribución de {columna}')
    plt.show()

# Configuración de la interfaz de Streamlit
def app():
    """
    Función principal que maneja la interfaz y el flujo de la aplicación.
    """
    st.title('Análisis de Deforestación con Imágenes de Satélite')

    # Opción para cargar archivo o leer desde URL
    opcion_carga = st.selectbox("Seleccione cómo cargar los datos:", ['Cargar archivo', 'Leer desde URL'])
    
    if opcion_carga == 'Cargar archivo':
        archivo = st.file_uploader("Subir archivo CSV", type='csv')
        if archivo:
            df = cargar_datos(archivo=archivo)
            st.write("Datos cargados:", df.head())
    elif opcion_carga == 'Leer desde URL':
        url = st.text_input("Introduzca la URL del archivo CSV")
        if url:
            df = cargar_datos(url=url)
            st.write("Datos cargados:", df.head())

    if df is not None:
        # Interpolación de los datos
        df_interpolado = interpolar_datos(df)

        # Selección de análisis a realizar
        analisis_seleccionados = st.multiselect(
            'Seleccione los análisis a realizar:',
            ['Generar mapa', 'Generar gráfico de torta'],
            default=['Generar mapa']
        )

        # Generar análisis según selección
        if 'Generar mapa' in analisis_seleccionados:
            generar_mapa(df_interpolado)

        if 'Generar gráfico de torta' in analisis_seleccionados:
            columna_torta = st.selectbox("Seleccione la columna para el gráfico de torta:", df_interpolado.columns)
            generar_grafico_torta(df_interpolado, columna_torta)

if __name__ == "__main__":
    app()
