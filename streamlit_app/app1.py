import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===========================================
# RUTA A LA CARPETA DATA
# ===========================================
DATA_FOLDER = "/app/data"  # Montada desde docker-compose o local

# ===========================================
# FUNCIONES
# ===========================================

def cargar_csvs_desde_data():
    """
    Carga los CSV desde la carpeta DATA_FOLDER.
    Retorna un diccionario de DataFrames.
    """
    rutas = {
        "clientes": os.path.join(DATA_FOLDER, "clientes.csv"),
        "metodos_pago": os.path.join(DATA_FOLDER, "metodos_pago.csv"),
        "categorias": os.path.join(DATA_FOLDER, "categorias.csv"),
        "productos": os.path.join(DATA_FOLDER, "productos.csv"),
        "ventas": os.path.join(DATA_FOLDER, "ventas.csv"),
    }

    dict_dfs = {}

    for nombre, ruta in rutas.items():
        if os.path.exists(ruta):
            dict_dfs[nombre] = pd.read_csv(ruta)
        else:
            st.warning(f"⚠ El archivo {ruta} no existe.")

    return dict_dfs


def entender_datos(df: pd.DataFrame):
    """
    Muestra información básica, estadísticos y nulos de un DataFrame.
    """
    st.write("### Info")
    st.write(df.info())
    st.write("### Descripción")
    st.write(df.describe(include='all'))
    st.write("### Nulos")
    st.write(df.isnull().sum())


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica de un DataFrame:
    - Quita duplicados
    - Elimina filas completamente nulas
    - Llena nulos numéricos con la mediana
    """
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df.fillna(df.median(numeric_only=True))
    return df

# ===========================================
# ERROR ORIGINAL: HBase / happybase
# ===========================================
# La siguiente sección se comentó porque Streamlit Cloud NO SOPORTA `happybase`
# ni conexiones a HBase. Cualquier intento de importar o usar happybase falla.
# import happybase
# def cargar_multiples_tablas_hbase(dict_dfs: dict, host: str):
#     ...

# ===========================================
# STREAMLIT APP
# ===========================================

st.title("Pipeline CRISP-DM con CSV (HBase eliminado)")
st.markdown("---")

# 1️⃣ Cargar CSV automáticamente
st.header("1. Cargar CSVs desde carpeta /data")

if st.button("Cargar Datos"):
    dict_dfs = cargar_csvs_desde_data()
    if len(dict_dfs) == 0:
        st.error("No se encontraron archivos CSV en la carpeta /data.")
    else:
        st.success("Datos cargados desde carpeta /data.")
        st.session_state["dict_raw"] = dict_dfs

        for nombre, df in dict_dfs.items():
            st.subheader(f"Dataset: {nombre}")
            st.write(df.head())

# 2️⃣ Entendimiento de Datos
if "dict_raw" in st.session_state:
    st.header("2. Entendimiento de Datos (CRISP-DM)")
    for nombre, df in st.session_state["dict_raw"].items():
        st.subheader(f"Dataset: {nombre}")
        entender_datos(df)

# 3️⃣ Limpieza de Datos
if "dict_raw" in st.session_state:
    if st.button("Limpiar Datos"):
        dict_limpio = {n: limpiar_datos(df) for n, df in st.session_state["dict_raw"].items()}
        st.session_state["dict_clean"] = dict_limpio
        st.success("Datos limpiados correctamente.")

# 4️⃣ Ver Datos Limpiados
if "dict_clean" in st.session_state:
    if st.button("Ver datos limpios"):
        for nombre, df in st.session_state["dict_clean"].items():
            st.subheader(f"{nombre}")
            st.dataframe(df)

# 5️⃣ SECCIÓN DE HBASE ELIMINADA
# Originalmente se intentaba subir datos a HBase usando happybase.
# Esto no funciona en Streamlit Cloud, por eso se eliminó.

# 6️⃣ Modelado Predictivo
if "dict_clean" in st.session_state:
    st.header("3. Modelado Predictivo (CRISP-DM)")

    df_v = st.session_state["dict_clean"].get("ventas")
    df_p = st.session_state["dict_clean"].get("productos")
    df_c = st.session_state["dict_clean"].get("categorias")
    df_m = st.session_state["dict_clean"].get("metodos_pago")
    df_cl = st.session_state["dict_clean"].get("clientes")

    if df_v is not None and df_p is not None:
        df = df_v.merge(df_p, on="ID_Producto", how="left")
        if "Categoría" in df.columns and "Categoría" in df_c.columns:
            df = df.merge(df_c, on="Categoría", how="left")
        if "ID_Cliente" in df.columns and "ID_Cliente" in df_cl.columns:
            df = df.merge(df_cl, on="ID_Cliente", how="left")
        if "Método_Pago" in df.columns and "Método" in df_m.columns:
            df["Método_Pago"] = df["Método_Pago"].astype(str)
            df_m["Método"] = df_m["Método"].astype(str)
            df = df.merge(df_m, left_on="Método_Pago", right_on="Método", how="left")

        st.subheader("Dataset final")
        st.dataframe(df.head())

        # Variables y modelo
        y = df["Cantidad"]
        variables = [v for v in ["Precio_Unitario","Stock","Categoría","Método_Pago","Estado","Región","Nombre_producto"] if v in df.columns]
        X = df[variables]

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(include="number").columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        modelo = Pipeline(steps=[("prep", preprocessor), ("reg", LinearRegression())])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.subheader("Predicción vs Real")
        st.write(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}).head())

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("MSE", round(mse,4))
        st.metric("R² Score", round(r2,4))

        # Visualización
        st.header("Visualización")
        st.subheader("Cantidad por Categoría")
        fig1, ax1 = plt.subplots()
        df.groupby("Categoría")["Cantidad"].sum().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Ventas por Método de Pago")
        fig2, ax2 = plt.subplots()
        df.groupby("Método_Pago")["Cantidad"].sum().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Precio vs Cantidad Vendida")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["Precio_Unitario"], df["Cantidad"])
        ax3.set_xlabel("Precio Unitario")
        ax3.set_ylabel("Cantidad")
        st.pyplot(fig3)
