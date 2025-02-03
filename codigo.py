import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt

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
    
    # Limpiar valores faltantes
    df = df.dropna()
    
    return df


def generar_mapa(df, lat_col, lon_col, variable_col):
    """
    Genera un mapa de la deforestación según las variables seleccionadas por el usuario.

    Args:
        df (pd.DataFrame): El DataFrame con los datos cargados.
        lat_col (str): El nombre de la columna con las coordenadas de latitud.
        lon_col (str): El nombre de la columna con las coordenadas de longitud.
        variable_col (str): El nombre de la columna que contiene la variable para el mapa.
    """
    # Verificar que las columnas existan en el DataFrame
    if lat_col not in df.columns or lon_col not in df.columns or variable_col not in df.columns:
        st.error("Las columnas seleccionadas no existen en los datos.")
        return
    
    # Crear un GeoDataFrame con las coordenadas de latitud y longitud
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))
    
    # Asignar la variable seleccionada como la de visualización en el mapa
    gdf[variable_col] = df[variable_col]
    
    # Mostrar el mapa
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=variable_col, ax=ax, legend=True,
             legend_kwds={'label': f"Valores de {variable_col}", 'orientation': "horizontal"})
    st.pyplot(fig)


def app():
    # Cargar los datos desde un archivo o URL
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = cargar_datos(archivo=archivo)

        # Verificar que los datos hayan sido cargados correctamente
        if df is not None:
            st.write("Primeras filas de los datos cargados:")
            st.write(df.head())

            # Permitir al usuario seleccionar las columnas para el mapa
            latitud_col = st.selectbox("Selecciona la columna de latitud", df.columns)
            longitud_col = st.selectbox("Selecciona la columna de longitud", df.columns)
            variable_col = st.selectbox("Selecciona la variable para visualizar en el mapa", df.columns)
            
            # Generar el mapa
            if latitud_col and longitud_col and variable_col:
                generar_mapa(df, latitud_col, longitud_col, variable_col)


if __name__ == "__main__":
    app()
