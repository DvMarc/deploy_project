import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from potencial_yac import IPR_curve_methods, Qo, Qb, aof, j
from analisis_nodal import pwf_darcy, pwf_vogel, f_darcy, sg_oil, sg_avg, gradient_avg
import base64

def image_to_base64(img):
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

icon = Image.open('icono.jpg')
icon_base64 = image_to_base64(icon)
st.set_page_config(page_title="Project-2 App", page_icon=icon, layout="wide")
rofer=Image.open('Rofer_corporation.png')

# Menú lateral
if "opcion" not in st.session_state:
    st.session_state["opcion"] = "Información del Proyecto"

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{icon_base64}" style="width: 200px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Menu")

    # Botones para navegación
    if st.button(":newspaper: Información del Proyecto"):
        st.session_state["opcion"] = "Información del Proyecto"
    if st.button(":droplet: Campo VOLVE"):
        st.session_state["opcion"] = "Campo VOLVE"
    if st.button(":abacus: Cálculos Petrofísicos"):
        st.session_state["opcion"] = "Cálculos Petrofísicos"
    if st.button(":chart_with_upwards_trend: Análisis Nodal"):
        st.session_state["opcion"] = "Análisis Nodal"

# Renderizar la página seleccionada
opcion = st.session_state["opcion"]

# Página 1: Información del Proyecto
if opcion == "Información del Proyecto":
    st.title("Información del Proyecto")
    st.write("---")
    st.write("""
    ### Descripción
    Esta aplicación permite realizar análisis petrofísicos y de producción con datos de pozos petroleros. 
    Podrás cargar archivos de datos, visualizar gráficas de producción y realizar cálculos de potencial de producción.

    #### Funcionalidades:
    - Carga de archivos en formato Excel para análisis de datos.
    - Cálculos de curvas IPR utilizando diferentes métodos.
    - Visualización de gráficos de producción de petróleo y agua.
    - Futuro desarrollo de análisis nodal.

    #### Contacto:
    Si tienes dudas o sugerencias, puedes contactar al equipo de desarrollo.
    """)

# Página 2: Carga de Archivos
elif opcion == "Campo VOLVE":
    st.title("Carga de Archivos")
    st.write("---")

    uploaded_file = st.file_uploader("Carga tu archivo de Excel con datos de producción del campo VOLVE", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)

        st.write("Vista previa de los datos cargados:")
        st.dataframe(data.head())

        wells = data['WELL_BORE_CODE'].unique()
        selected_well = st.selectbox("Selecciona un pozo para graficar:", wells)

        well_data = data[data['WELL_BORE_CODE'] == selected_well]

        well_data['DATEPRD'] = pd.to_datetime(well_data['DATEPRD'], errors='coerce')

        well_data = well_data.sort_values(by='DATEPRD')

        # Graficar Qo vs t
        st.subheader("Gráfico: Qo vs t (año)")
        plt.figure(figsize=(10, 5))
        plt.plot(well_data['DATEPRD'], well_data['BORE_OIL_VOL'], label='Qo (Producción de Petróleo)', color='blue')
        plt.xlabel('Fecha')
        plt.ylabel('Volumen de Petróleo (Qo)')
        plt.title(f'Producción de Petróleo para el Pozo {selected_well}')
        plt.legend()
        st.pyplot(plt)

        # Graficar Qw vs t
        st.subheader("Gráfico: Qw vs t (año)")
        plt.figure(figsize=(10, 5))
        plt.plot(well_data['DATEPRD'], well_data['BORE_WAT_VOL'], label='Qw (Producción de Agua)', color='green')
        plt.xlabel('Fecha')
        plt.ylabel('Volumen de Agua (Qw)')
        plt.title(f'Producción de Agua para el Pozo {selected_well}')
        plt.legend()
        st.pyplot(plt)

        # Graficar Qo y Qw juntos
        st.subheader("Gráfico: Qo y Qw vs t (año)")
        plt.figure(figsize=(10, 5))
        plt.plot(well_data['DATEPRD'], well_data['BORE_OIL_VOL'], label='Qo (Producción de Petróleo)', color='blue')
        plt.plot(well_data['DATEPRD'], well_data['BORE_WAT_VOL'], label='Qw (Producción de Agua)', color='green')
        plt.xlabel('Fecha')
        plt.ylabel('Volumen (Qo y Qw)')
        plt.title(f'Producción de Petróleo y Agua para el Pozo {selected_well}')
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("Por favor, carga un archivo de Excel para comenzar.")

# Página 3: Cálculos Petrofísicos
elif opcion == "Cálculos Petrofísicos":
    st.title("Cálculos Petrofísicos")

    st.write("Ingresa los parámetros necesarios para calcular el potencial de producción:")

    q_test = st.number_input("Caudal de prueba (bpd)", value=500.0)
    pwf_test = st.number_input("Presión de fondo fluyente durante la prueba (psia)", value=3000.0)
    pr = st.number_input("Presión inicial del yacimiento (psia)", value=4000.0)
    pb = st.number_input("Presión de burbuja (psia)", value=2500.0)
    ko = st.number_input("Permeabilidad (mD)", value=50.0)
    h = st.number_input("Altura neta productiva (ft)", value=50.0)
    bo = st.number_input("Factor de volumen del petróleo (rb/stb)", value=1.2)
    uo = st.number_input("Viscosidad del petróleo (cp)", value=2.0)
    re = st.number_input("Radio de drenaje (ft)", value=1000.0)
    rw = st.number_input("Radio del pozo (ft)", value=0.5)
    s = st.number_input("Skin (daño en la formación)", value=0.0)
    metodo = st.selectbox("Método para cálculo de Qo", ["Darcy", "Vogel", "Standing", "IPR_compuesto"])

    pwf_array = np.arange(pr, 0, -400)

    if st.button("Calcular"):
        with st.spinner("Calculando..."):
            st.subheader("Curva IPR (Influencia Presión - Producción)")
            IPR_curve_methods(q_test, pwf_test, pr, pwf_array, pb, method=metodo)
            st.pyplot(plt)

            st.write("### Resultados:")
            st.write(f"- Índice de productividad (J): {j(q_test, pwf_test, pr, pb):.4f} stb/d/psi")
            st.write(f"- Caudal al punto de burbuja (Qb): {Qb(q_test, pwf_test, pr, pb):.2f} bpd")
            st.write(f"- Caudal máximo de producción (AOF): {aof(q_test, pwf_test, pr, pb):.2f} bpd")
            st.write(f"- Caudal Qo (bpd) según {metodo}: {Qo(q_test, pwf_test, pr, 2000, pb):.2f} bpd")

# Página 4: Análisis Nodal
elif opcion == "Análisis Nodal":
    st.title("Análisis Nodal")
    st.write("---")

    # Entradas del usuario
    st.subheader("Parámetros de entrada")
    Pr = st.number_input("Presión inicial del yacimiento (Pr) [psia]", value=3000.0)
    Pb = st.number_input("Presión de burbuja (Pb) [psia]", value=2300.0)
    Qt = st.number_input("Caudal total de prueba (Qt) [bpd]", value=1500.0)
    Pwft = st.number_input("Presión de fondo fluyente (Pwft) [psia]", value=2400.0)
    THP = st.number_input("Presión en la cabeza del pozo (THP) [psia]", value=360.0)
    wc = st.number_input("Porcentaje de corte de agua (wc)", value=0.9)
    API = st.number_input("Gravedad API del petróleo (API)", value=20.0)
    sg_h2o = st.number_input("Gravedad específica del agua (sg_h2o)", value=1.09)
    ID = st.number_input("Diámetro interno de la tubería (ID) [in]", value=3.5)
    tvd = st.number_input("Profundidad vertical verdadera (TVD) [ft]", value=9000.0)
    md = st.number_input("Profundidad medida (MD) [ft]", value=10500.0)
    C = st.number_input("Factor de Hazen-Williams (C)", value=120.0)

    if st.button("Calcular"):
        with st.spinner("Calculando..."):
            st.subheader("Resultados del Análisis Nodal")
            columns = ['Q(bpd)', 'Pwf(psia)', 'THP(psia)', 'Pgravity(psia)', 'f', 'F(ft)', 'Pf(psia)', 'Po(psia)', 'Psys(psia)']
            df = pd.DataFrame(columns=columns)

            df[columns[0]] = np.linspace(0, 7500, 10)  # Rango de tasas de flujo
            df[columns[1]] = df['Q(bpd)'].apply(lambda x: pwf_darcy(Qt, Pwft, x, Pr, Pb))
            df[columns[2]] = THP
            df[columns[3]] = gradient_avg(API, wc, sg_h2o) * tvd
            df[columns[4]] = df['Q(bpd)'].apply(lambda x: f_darcy(x, ID, C))
            df[columns[5]] = df['f'] * md
            df[columns[6]] = gradient_avg(API, wc, sg_h2o) * df['F(ft)']
            df[columns[7]] = df['THP(psia)'] + df['Pgravity(psia)'] + df['Pf(psia)']
            df[columns[8]] = df['Po(psia)'] - df['Pwf(psia)']

            # Mostrar tabla
            st.write("Tabla de resultados:")
            st.dataframe(df)

            # Graficar resultados
            st.subheader("Gráficos del Análisis Nodal")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Q(bpd)'], df['Pwf(psia)'], label='IPR', color='red')
            ax.plot(df['Q(bpd)'], df['Po(psia)'], label='VLP', color='green')
            ax.plot(df['Q(bpd)'], df['Psys(psia)'], label='Curva del Sistema', color='blue')
            ax.set_xlabel('Tasa de Flujo (Q) [bpd]')
            ax.set_ylabel('Presión (Pwf) [psia]')
            ax.legend()
            ax.grid()
            st.pyplot(fig)
