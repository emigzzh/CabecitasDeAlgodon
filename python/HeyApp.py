import streamlit as st
import pandas as pd
from modeloComercioLightGBM import runPredictiveModel
from datetime import datetime, timedelta

idxClient = 0
fecha_referencia = datetime(2023, 1, 31)

def on_number_input_change():
    st.session_state.flag = False

def main():
    if 'flag' not in st.session_state:
        st.session_state.flag = True

    if 'idClient' not in st.session_state:
        st.session_state.idClient = 0

    st.set_page_config(page_title="Hey, prophet", layout="wide")
    
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* sidebar */
        section[data-testid="stSidebar"] {
            background-color: #2b2b2b;
            padding: 0;
        }
                        
        /* botones */
        section[data-testid="stSidebar"] button {
            width: 100%;
            min-height: auto !important;
            height: auto !important;
            padding: 10px 20px !important;
                
            border: none !important;
            background-color: transparent !important;
            
            text-align: left !important;
            color: #a6a6a6 !important;
            font-size: 1rem;
            margin: 0 !important;
            border-radius: 0;
            border: none !important;
            box-shadow: none !important;
            border-bottom: none !important;
            transition: background-color 0.2s;
            display: block !important;
        }
        
        /* efecto hover botones */
        section[data-testid="stSidebar"] button:hover {
            background-color: #4f4f4f !important;
            color: #8da5af !important;
        }
        
        /* Reset del contenedor flex interno */
        section[data-testid="stSidebar"] button > div {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }
                
        section[data-testid="stSidebar"] button div p {
            font-family: 'Manrope', serif !important;
            font-weight: bold !important;
            font-size: 1.2rem;
        }
        
        /* Ajuste del texto dentro del botón */
        section[data-testid="stSidebar"] button span {
            margin: 0 !important;
            padding: 0 !important;
            display: block !important;
            text-align: left !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("DcA_FINAL\logo-heypago.png", width=120)
        
        if st.button("Datos Crudos"):
            st.session_state.page = "option1"
            st.session_state.flag = True

        if st.button("Usuario Final"):
            st.session_state.page = "option2"
            st.session_state.flag = True

        valor_input = st.number_input(
            "Inserte Índice Manualmente",
            min_value=0,
            max_value=961,
            value=500,
            step=1,
            key="inputID",
            on_change=on_number_input_change
        )
    
        # Hay 961 inputs disponibles en los datasets entregados filtrados
        if valor_input > 961:
            valor_input = 961

        if valor_input < 0:
            valor_input = 0
        
    if 'page' not in st.session_state:
        st.session_state.page = "option0"
    
    if st.session_state.page == "option1" and st.session_state.flag == True:
        st.session_state.idClient = st.session_state.inputID
        show_option1()
    elif st.session_state.page == "option2" and st.session_state.flag == True:
        show_option2()
    elif st.session_state.page == "option0" or st.session_state.flag == False:
        show_option0()

def show_option0():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Fondo blanco para toda la aplicación */
        html, body, #root, .main, .stApp {
            background-color: white !important;
            color: #000000 !important;
        }
        
        /* Fuente Manrope para todos los textos */
        body, h1, h2, h3, h4, h5, h6, p, div, span {
            font-family: 'Manrope', sans-serif !important;
        }
        
        /* Estilo para el saludo y divider */
        div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown > p:contains("Hola")) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Divider ajustado */
        hr {
            border-top: 1px solid #e0e0e0 !important;
            margin: 4px 0 12px 0 !important;
        }
        
        /* Títulos */
        h1 {
            font-weight: 700 !important;
            color: #222222 !important;
        }
        
        h2 {
            font-weight: 600 !important;
            color: #252525 !important;
            margin-top: 1.5rem !important;
        }
        
        /* Contenedores */
        .stContainer {
            background-color: white;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def show_option1():
    try:
        fig1, fig2, fig3, idCliente, f1s, prcs, resultados = runPredictiveModel(st.session_state.idClient)
            
    except Exception as e:
        st.error(f"Error al ejecutar el modelo: {str(e)}")

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Fondo blanco para toda la aplicación */
        html, body, #root, .main, .stApp {
            background-color: white !important;
            color: #000000 !important;
        }
        
        /* Fuente Manrope para todos los textos */
        body, h1, h2, h3, h4, h5, h6, p, div, span {
            font-family: 'Manrope', sans-serif !important;
        }
        
        /* Estilo para el saludo y divider */
        div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown > p:contains("Hola")) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Divider ajustado */
        hr {
            border-top: 1px solid #e0e0e0 !important;
            margin: 4px 0 12px 0 !important;
        }
        
        /* Títulos */
        h1 {
            font-weight: 700 !important;
            color: #222222 !important;
        }
        
        h2 {
            font-weight: 600 !important;
            color: #252525 !important;
            margin-top: 1.5rem !important;
        }
        
        /* Contenedores */
        .stContainer {
            background-color: white;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("User ID: ", idCliente)
    st.title("Modelo Predictivo")

    st.divider()
    st.subheader("Probabilidad de Gasto Recurrente (%)")
    if fig1 is not None:
        st.pyplot(fig1)
    
    st.divider()
    st.subheader("Monto Promedio por Gasto ($)")
    if fig2 is not None:
        st.pyplot(fig2)

    st.divider()
    st.subheader("Frecuencia de Compras (Periodicidad)")
    if fig3 is not None:
        st.pyplot(fig3)

    st.divider()
    st.subheader("Datos y Métricas")
    st.write("Precisión (Precisión): ", prcs)
    st.write("Puntaje F1 (f1-score): ", f1s)

def show_option2():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Fondo blanco para toda la aplicación */
        html, body, #root, .main, .stApp {
            background-color: white !important;
            color: #000000 !important;
        }
        
        /* Fuente Manrope para todos los textos */
        body, h1, h2, h3, h4, h5, h6, p, div, span {
            font-family: 'Manrope', sans-serif !important;
        }
        
        /* Estilo para el saludo y divider */
        div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown > p:contains("Hola")) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Divider ajustado */
        hr {
            border-top: 1px solid #e0e0e0 !important;
            margin: 4px 0 12px 0 !important;
        }
        
        /* Títulos */
        h1 {
            font-weight: 700 !important;
            color: #222222 !important;
        }
        
        h2 {
            font-weight: 600 !important;
            color: #252525 !important;
            margin-top: 1.5rem !important;
        }
        
        /* Contenedores */
        .stContainer {
            background-color: white;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    try:
        fig1, fig2, fig3, idCliente, f1s, prcs, resultados = runPredictiveModel(st.session_state.idClient)
            
    except Exception as e:
        st.error(f"Error al ejecutar el modelo: {str(e)}")

    st.write("Hola, <username>!")
    st.write("---")
    st.title("Resumen Predictivo de Compras")
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        for _, row in resultados.iterrows():
            st.write(f"""
            ### {row['comercio_nombre'].upper()}  
            - Probabilidad de gasto recurrente: {row['probabilidad']:.1%}  
            - Monto promedio: ${row['monto_promedio']:.2f}  
            - Próxima compra estimada: {row['fecha_estimada'].strftime('%d/%m/%Y')}  
            - Frecuencia: cada {row['frecuencia_dias']:.0f} días  
            - Total transacciones históricas: {row['total_transacciones']}  
            - Gasto total histórico: ${row['monto_total_historico']:.2f}
            """)
            st.write("---")
    with col2:
        st.subheader("Pred. Próxima semana")
        st.write(f"({fecha_referencia.strftime('%d/%m/%Y')} al {(fecha_referencia + timedelta(days=7)).strftime('%d/%m/%Y')})")
    
        pagos_proxima_semana = resultados[
            (resultados['fecha_estimada'] >= fecha_referencia) &
            (resultados['fecha_estimada'] <= fecha_referencia + timedelta(days=7))
        ].sort_values('fecha_estimada')
        
        if len(pagos_proxima_semana) > 0:
            st.write("**Próximos pagos programados:**")
            for _, pago in pagos_proxima_semana.iterrows():
                dias_restantes = (pago['fecha_estimada'] - fecha_referencia).days
                st.write(f"""
                - **{pago['fecha_estimada'].strftime('%A %d/%m/%Y')}** 
                ({f"Dentro de {dias_restantes} días" if dias_restantes > 0 else "Hoy"})
                - Comercio: {pago['comercio_nombre'].upper()}
                - Monto estimado: ${pago['monto_promedio']:.2f}
                - Probabilidad: {pago['probabilidad']:.1%}
                """)
                st.write("---")
        else:
            st.warning("No hay pagos programados para la próxima semana")

if __name__ == "__main__":
    main()
