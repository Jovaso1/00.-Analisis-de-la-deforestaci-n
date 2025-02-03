import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path: str):
    """Carga los datos de deforestación desde un archivo CSV.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path)

def generate_map(df: pd.DataFrame, lat_column: str, lon_column: str, filters: dict):
    """Genera un mapa de zonas deforestadas con los filtros seleccionados.

    Args:
        df (pd.DataFrame): DataFrame con datos de deforestación.
        lat_column (str): Nombre de la columna de latitud.
        lon_column (str): Nombre de la columna de longitud.
        filters (dict): Filtros para seleccionar las variables a mostrar en el mapa.
    """
    filtered_df = df
    for column, (min_val, max_val) in filters.items():
        filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]
    
    gdf = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df[lon_column], filtered_df[lat_column]))
    gdf.plot(marker='o', color='red', markersize=5)
    plt.title("Mapa de Zonas Deforestadas")
    plt.show()

def analyze_deforestation(df: pd.DataFrame):
    """Analiza los datos de deforestación, como la superficie deforestada y tasas de deforestación.

    Args:
        df (pd.DataFrame): DataFrame con los datos de deforestación.

    Returns:
        dict: Resultados del análisis, incluyendo superficie y tasa de deforestación.
    """
    total_deforestation = df['area_deforestada'].sum()
    deforestation_rate = df['area_deforestada'].pct_change().mean() * 100
    return {
        'Total Deforestation Area': total_deforestation,
        'Deforestation Rate': deforestation_rate
    }

def cluster_analysis(df: pd.DataFrame, n_clusters: int = 3):
    """Realiza un análisis de clústeres en las superficies deforestadas utilizando KMeans.

    Args:
        df (pd.DataFrame): DataFrame con las superficies deforestadas.
        n_clusters (int, optional): Número de clústeres a identificar. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame con los clústeres asignados.
    """
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['area_deforestada', 'latitud', 'longitud']])
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    return df

def plot_vegetation_pie_chart(df: pd.DataFrame):
    """Genera un gráfico de torta que muestra la distribución de la deforestación por tipo de vegetación.

    Args:
        df (pd.DataFrame): DataFrame con los tipos de vegetación y superficie deforestada.
    """
    vegetation_counts = df['tipo_vegetacion'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(vegetation_counts, labels=vegetation_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Distribución de la Deforestación por Tipo de Vegetación")
    plt.show()

# Función principal de Streamlit
def main():
    st.title("Análisis de Deforestación con Imágenes de Satélite")

    # Cargar datos
    file_path = st.text_input("Ruta del archivo CSV:", "deforestacion.csv")
    df = load_data(file_path)

    # Filtros de usuario para el mapa
    st.sidebar.header("Filtros para el Mapa")
    lat_min, lat_max = st.sidebar.slider("Latitud", min_value=-90, max_value=90, value=(-5, 5))
    lon_min, lon_max = st.sidebar.slider("Longitud", min_value=-180, max_value=180, value=(-10, 10))
    filters = {
        'latitud': (lat_min, lat_max),
        'longitud': (lon_min, lon_max)
    }
    generate_map(df, 'latitud', 'longitud', filters)

    # Análisis de deforestación
    analysis_results = analyze_deforestation(df)
    st.subheader("Análisis de Deforestación")
    st.write(analysis_results)

    # Análisis de clúster
    df_clustered = cluster_analysis(df)
    st.subheader("Análisis de Clúster")
    st.write(df_clustered[['latitud', 'longitud', 'cluster']].head())

    # Gráfico de torta por tipo de vegetación
    plot_vegetation_pie_chart(df)

if __name__ == "__main__":
    main()
