import streamlit as st
import pandas as pd
import happybase
from sqlalchemy import create_engine

# =============================
# FUNCIONES CRISP-DM + HBASE
# =============================

def cargar_csvs_multiple(rutas: dict) -> dict:
    dict_dfs = {nombre: pd.read_csv(ruta) for nombre, ruta in rutas.items()}
    return dict_dfs


def entender_datos(df: pd.DataFrame):
    st.write("### Info")
    st.write(df.info())
    st.write("### Descripción")
    st.write(df.describe(include='all'))
    st.write("### Nulos")
    st.write(df.isnull().sum())


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df = df.fillna(df.median(numeric_only=True))
    return df


def cargar_multiples_tablas_hbase(dict_dfs: dict, host: str):
    # 👇 CORRECCIÓN: especificar puerto de Thrift (9091 en Docker)
    connection = happybase.Connection(host=host, port=9091)
    connection.open()

    familias = {"cf1": dict()}
    tablas_existentes = connection.tables()

    for nombre_tabla, df in dict_dfs.items():
        if nombre_tabla.encode() not in tablas_existentes:
            connection.create_table(nombre_tabla, familias)

        t = connection.table(nombre_tabla)

        for i, row in df.iterrows():
            data_dict = {}
            for col in df.columns:
                valor = row[col]
                if pd.notnull(valor):
                    data_dict[f"cf1:{col}"] = str(valor).encode()
            t.put(str(i).encode(), data_dict)

        st.success(f"Tabla '{nombre_tabla}' cargada en HBase.")

    connection.close()



# ==================================================================
# APLICACIÓN STREAMLIT
# ==================================================================

st.title("Pipeline CRISP-DM con Upload a HBase")
st.markdown("---")

# 1. SUBIR CSVs
st.header("1. Cargar CSVs para procesar")
clientes = st.file_uploader("Clientes", type=["csv"])
metodos = st.file_uploader("Métodos de Pago", type=["csv"])
categorias = st.file_uploader("Categorías", type=["csv"])
productos = st.file_uploader("Productos", type=["csv"])
ventas = st.file_uploader("Ventas", type=["csv"])

if st.button("Procesar CSVs"):
    rutas = {}

    if clientes: rutas["clientes"] = clientes
    if metodos: rutas["metodos_pago"] = metodos
    if categorias: rutas["categorias"] = categorias
    if productos: rutas["productos"] = productos
    if ventas: rutas["ventas"] = ventas

    if len(rutas) == 0:
        st.error("Debe subir al menos un archivo CSV.")
    else:
        dict_dfs = {name: pd.read_csv(file) for name, file in rutas.items()}

        st.success("Archivos CSV cargados correctamente.")
        st.session_state["dict_raw"] = dict_dfs

# 2. ENTENDIMIENTO DE DATOS
if "dict_raw" in st.session_state:
    st.header("2. Entendimiento de Datos (CRISP-DM)")
    for nombre, df in st.session_state["dict_raw"].items():
        st.subheader(f"Dataset: {nombre}")
        st.write(df.head())
        entender_datos(df)

# 3. LIMPIEZA DE DATOS
if "dict_raw" in st.session_state:
    if st.button("Limpiar Datos"):
        dict_limpio = {n: limpiar_datos(df) for n, df in st.session_state["dict_raw"].items()}
        st.session_state["dict_clean"] = dict_limpio
        st.success("Datos limpiados correctamente.")

# ⭐⭐⭐ NUEVO BOTÓN: VER DATOS LIMPIOS ⭐⭐⭐
if "dict_clean" in st.session_state:
    if st.button("Ver datos limpios antes de subir"):
        st.subheader("📌 Datos limpios")
        for nombre, df in st.session_state["dict_clean"].items():
            st.write(f"### {nombre}")
            st.dataframe(df)

# 4. SUBIR A HBASE
if "dict_clean" in st.session_state:
    st.header("4. Subir Datos Limpiados a HBase")
    host = st.text_input("Host HBase", "localhost")

    if st.button("Subir a HBase"):
        cargar_multiples_tablas_hbase(st.session_state["dict_clean"], host)
        st.success("Proceso completo CRISP-DM ejecutado.")

# ================================
# 5. MODELADO COMPLETO CON TODAS LAS TABLAS
# ================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

if "dict_clean" in st.session_state:

    st.header("5. Modelado Predictivo con Todas las Tablas")

    # Cargar data limpia
    df_v = st.session_state["dict_clean"].get("ventas")
    df_p = st.session_state["dict_clean"].get("productos")
    df_c = st.session_state["dict_clean"].get("categorias")
    df_m = st.session_state["dict_clean"].get("metodos_pago")
    df_cl = st.session_state["dict_clean"].get("clientes")

    # Validar tablas
    if df_v is not None and df_p is not None:

        # ---------------------------------------
        # 1️⃣ JOIN: ventas + productos
        # ---------------------------------------
        df = df_v.merge(df_p, on="ID_Producto", how="left")

        # ---------------------------------------
        # 2️⃣ JOIN: productos + categorias
        # ---------------------------------------
        if "Categoría" in df.columns and "Categoría" in df_c.columns:
            df = df.merge(df_c, on="Categoría", how="left")

        # ---------------------------------------
        # 3️⃣ JOIN: ventas + clientes
        # ---------------------------------------
        if "ID_Cliente" in df.columns and "ID_Cliente" in df_cl.columns:
            df = df.merge(df_cl, on="ID_Cliente", how="left")

        # ---------------------------------------
        # 4️⃣ JOIN: ventas + metodos_pago
        # Coincide por nombre del método
        # ---------------------------------------
        if "Método_Pago" in df.columns and "Método" in df_m.columns:

        # 👇 CORRECCIÓN: asegurar que ambas columnas sean strings
            df["Método_Pago"] = df["Método_Pago"].astype(str)
            df_m["Método"] = df_m["Método"].astype(str)

            df = df.merge(df_m, left_on="Método_Pago", right_on="Método", how="left")


        st.subheader("Dataset unificado final")
        st.dataframe(df.head())

        # ===========================
        # 5️⃣ Selección de variables
        # ===========================
        y = df["Cantidad"]  # Variable objetivo

        # Variables predictoras reales disponibles
        variables = [
            "Precio_Unitario",
            "Stock",
            "Categoría",
            "Método_Pago",
            "Estado",
            "Región",
            "Nombre_producto"
        ]

        # Filtrar solo columnas que existen
        variables = [v for v in variables if v in df.columns]

        X = df[variables]

        # Columnas categóricas y numéricas
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(include="number").columns.tolist()

        # ===========================
        # 6️⃣ Pipeline de modelado
        # ===========================
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        modelo = Pipeline(steps=[
            ("prep", preprocessor),
            ("reg", LinearRegression())
        ])

        # ===========================
        # 7️⃣ Entrenamiento
        # ===========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        st.subheader("Predicción vs Real")
        st.write(pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}).head())

        # ===========================
        # 8️⃣ Evaluación del modelo
        # ===========================
        st.header("6. Evaluación del Modelo")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.metric("MSE", round(mse, 4))
        st.metric("R² Score", round(r2, 4))


        # ===========================
        # 9️⃣ Gráficas finales
        # ===========================
        st.header("7. Visualización")

        # --- Gráfico 1: Cantidad por Categoría ---
        st.subheader("1️⃣ Cantidad Total por Categoría")
        fig1, ax1 = plt.subplots()
        df.groupby("Categoría")["Cantidad"].sum().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Categoría")
        ax1.set_ylabel("Cantidad Total")
        st.pyplot(fig1)

        # --- Gráfico 2: Ventas por Método de Pago ---
        st.subheader("2️⃣ Ventas por Método de Pago")
        fig2, ax2 = plt.subplots()
        df.groupby("Método_Pago")["Cantidad"].sum().plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Método de Pago")
        ax2.set_ylabel("Cantidad Vendida")
        st.pyplot(fig2)

        # --- Gráfico 3: Precio vs Cantidad Vendida ---
        st.subheader("3️⃣ Relación Precio Unitario vs Cantidad Vendida")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["Precio_Unitario"], df["Cantidad"])
        ax3.set_xlabel("Precio Unitario")
        ax3.set_ylabel("Cantidad")
        st.pyplot(fig3)

    else:
        st.error("Debes subir al menos las tablas 'ventas' y 'productos'.")
