import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import warnings
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración inicial
warnings.filterwarnings('ignore', category=UserWarning)
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')


# Paso 1. Cargar y limpiar datos
# Se espera un dataFrame de clientes y transacciones limpios
def cargar_datos(ruta_clientes, ruta_transacciones):
    print("\n -------> P1. Limpiando Datos...")
    try:
        # Selección de columnas a usarse
        clientes = pd.read_csv(ruta_clientes, usecols=[
            'id', 'fecha_nacimiento', 'fecha_alta', 'actividad_empresarial'
        ])
        
        transacciones = pd.read_csv(ruta_transacciones, usecols=[
            'id', 'fecha', 'comercio', 'monto'
        ])
        
        clientes = clientes.dropna().copy()
        transacciones = transacciones.dropna().copy()
        
        # Conversión de fechas
        clientes['fecha_nacimiento'] = pd.to_datetime(clientes['fecha_nacimiento'], dayfirst=True, errors='coerce')
        clientes['fecha_alta'] = pd.to_datetime(clientes['fecha_alta'], dayfirst=True, errors='coerce')
        transacciones['fecha'] = pd.to_datetime(transacciones['fecha'], dayfirst=True, errors='coerce')
        
        # Eliminación de fechas inválidas
        clientes = clientes[clientes['fecha_nacimiento'].notna() & clientes['fecha_alta'].notna()]
        transacciones = transacciones[transacciones['fecha'].notna()]
        
        print("--> Datos cargados y limpios correctamente")
        print(f" Comercios únicos encontrados: {transacciones['comercio'].nunique()}")
        print(f" Tipos de comercio: {list(transacciones['comercio'].unique())}")
        
        return clientes, transacciones
    
    except Exception as e:
        print(f" Error al cargar datos: {str(e)}")
        raise

# Paso 2: Procesamiento por comercio
def procesar_caracteristicas_por_comercio(clientes, transacciones):
    print("\n -------> P.2 Procesando características por comercio...")
    
    try:
        # Edad y antigüedad del cliente
        hoy = datetime.now()
        clientes['edad'] = (hoy - clientes['fecha_nacimiento']).dt.days // 365
        clientes['antiguedad'] = (hoy - clientes['fecha_alta']).dt.days // 30
        
        # Codificación de actividad empresarial
        def codificar_comercio(texto):
            texto = str(texto).upper().strip()
            
            # 1. E-commerce y Marketplace
            if any(x in texto for x in ['AMAZON', 'MERCADO PAGO', 'MERCADOPAGO', 'ALIEXPRESS', 'TEMU', 'SHEIN']):
                return 1
            
            # 2. Delivery y Transporte
            if any(x in texto for x in ['RAPPI', 'RAPPIPRO', 'SOFT RAPPI', 'UBER', 'UBER EATS', 'DIDI', 'DIDIFOOD', 'DIDI FOOD', 'DIDI RIDES']):
                return 2
            
            # 3. Tiendas de Conveniencia
            if any(x in texto for x in ['OXXO', '7 ELEVEN', '7ELEVEN', 'UNDOSTRES']):
                return 3
            
            # 4. Supermercados y Grandes Superficies
            if any(x in texto for x in ['SORIANA', 'HEB', 'WALMART', 'WAL-MART', 'SUPERCENTER', 'SUPERAMA', 'CHEDRAUI', 'ALSUPER']):
                return 4
            
            # 5. Clubes de Precio y Mayoristas
            if any(x in texto for x in ['COSTCO', 'SAMS CLUB']):
                return 5
            
            # 6. Farmacias
            if any(x in texto for x in ['FARMACIAS DEL AHORRO', 'FARMACIAS GUADALAJARA', 'FARMACIAS SIMILARES']):
                return 6
            
            # 7. Departamentales y Retail
            if any(x in texto for x in ['LIVERPOOL', 'SEARS', 'COPPEL']):
                return 7
            
            # 8. Streaming y Entretenimiento
            if any(x in texto for x in ['SPOTIFY', 'NETFLIX', 'DISNEY PLUS', 'MAX', 'CRUNCHYROLL', 'VIX', 'CINEPOLIS']):
                return 8
            
            # 9. Tecnología y Software
            if any(x in texto for x in ['APPLE', 'ITUNES', 'MICROSOFT', 'GOOGLE', 'GOOGLE ONE', 'GOOGLE YOUTUBE', 'GOOGLE YOUTUBEPREMIUM', 'GOOGLE AMAZON', 'OPENAI', 'ADOBE', 'CANVA', 'PLAYSTATION NETWORK', 'ROKU', 'AUDIBLE']):
                return 9
            
            # 10. Telecomunicaciones e Internet
            if any(x in texto for x in ['TELCEL', 'TELMEX', 'IZZI', 'TOTALPLAY', 'TOTAL PLAY', 'AT&T', 'ATT', 'MI ATT', 'MEGACABLE', 'TELEFONICA', 'CABLEYCOMUN']):
                return 10
            
            # 11. Servicios Públicos
            if any(x in texto for x in ['CFE', 'SERV AGUA DREN', 'ROTOPLAS', 'METROBUS']):
                return 11
            
            # 12. Gasolineras
            if any(x in texto for x in ['OXXO GAS', 'COSTCO GAS']):
                return 12
            
            # 13. Restaurantes y Comida
            if any(x in texto for x in ['STARBUCKS', 'CARLS JR']):
                return 13
            
            # 14. Fintech y Pagos
            if any(x in texto for x in ['APLAZO', 'APLAZ', 'KUESKI PAY', 'CASHI ECOMMERCE', 'UBRPAGOSMEX']):
                return 14
            
            # 15. Fitness y Bienestar
            if any(x in texto for x in ['SMARTFIT', 'SMART FIT']):
                return 15
            
            # 16. Seguros
            if 'ALLIANZ MEXICO' in texto:
                return 16
            
            # 17. Apuestas y Juegos
            if any(x in texto for x in ['CALIENTE', 'BET365', 'TULOTERO']):
                return 17
            
            # 18. Redes Sociales
            if 'FACEBOOK' in texto:
                return 18
            
            # 19. Otros Servicios Diversos
            if any(x in texto for x in ['NAYAX', 'BAIT', 'MELIMAS', 'BAE', 'SMART', 'PARCO', 'TOTAL PASS', 'RENTAMOVISTAR', 'URBANI']):
                return 19
            
            # 20. No clasificado
            return 20
        
        clientes['actividad_cod'] = clientes['actividad_empresarial'].apply(codificar_comercio)
        
        # Procesamiento específico por comercio
        transacciones = transacciones.sort_values(['id', 'comercio', 'fecha'])
        
        # Diferencia de días entre transacciones cliente-comercio
        transacciones['dias_entre_trans'] = transacciones.groupby(['id', 'comercio'])['fecha'].diff().dt.days
        
        # Estadística por comercio
        features_por_comercio = transacciones.groupby(['id', 'comercio']).agg(
            frecuencia_promedio=('dias_entre_trans', 'mean'),
            desviacion_frecuencia=('dias_entre_trans', 'std'),
            monto_promedio=('monto', 'mean'),
            monto_maximo=('monto', 'max'),
            monto_minimo=('monto', 'min'),
            cantidad_transacciones=('monto', 'size'),
            primera_fecha=('fecha', 'min'),
            ultima_fecha=('fecha', 'max'),
            monto_total=('monto', 'sum')
        ).reset_index()
        
        # Métricas adicionales por comercio
        features_por_comercio['rango_fechas'] = (
            features_por_comercio['ultima_fecha'] - features_por_comercio['primera_fecha']
        ).dt.days
        
        features_por_comercio['consistencia'] = (
            features_por_comercio['cantidad_transacciones'] / 
            (features_por_comercio['rango_fechas'] / 30 + 1)  # Transacciones por mes
        ).fillna(0)
        
        # Estacionalidad por comercio
        transacciones['mes'] = transacciones['fecha'].dt.month
        estacionalidad = transacciones.groupby(['id', 'comercio', 'mes']).size().reset_index(name='trans_por_mes')
        coef_variacion_mensual = estacionalidad.groupby(['id', 'comercio'])['trans_por_mes'].agg(['std', 'mean']).reset_index()
        coef_variacion_mensual['variabilidad_mensual'] = coef_variacion_mensual['std'] / (coef_variacion_mensual['mean'] + 0.01)
        
        features_por_comercio = features_por_comercio.merge(
            coef_variacion_mensual[['id', 'comercio', 'variabilidad_mensual']], 
            on=['id', 'comercio'], 
            how='left'
        )
        
        features_filtradas = []
        
        for comercio in features_por_comercio['comercio'].unique():
            print(f"\n ----- Procesando comercio: {comercio} ------")
            
            datos_comercio = features_por_comercio[features_por_comercio['comercio'] == comercio].copy()
            
            # Umbrales adaptativos por tipo de comercio (basado en los 20 grupos definidos)
            if comercio in [3, 4, 5, 6]:  # Tiendas de conveniencia, Supermercados, Clubes de precio, Farmacias
                # Comercios de necesidad básica - más frecuentes
                umbral_frecuencia = 14  # 2 semanas
                min_transacciones = 3
            elif comercio in [13]:  # Restaurantes y Comida
                # Restaurantes - frecuencia media
                umbral_frecuencia = 21  # 3 semanas
                min_transacciones = 2
            elif comercio in [7]:  # Departamentales y Retail (ropa, moda)
                # Ropa - menos frecuente
                umbral_frecuencia = 60  # 2 meses
                min_transacciones = 2
            elif comercio in [11, 14, 16]:  # Servicios públicos, Fintech, Seguros
                # Servicios bancarios/públicos - muy regulares
                umbral_frecuencia = 30  # 1 mes
                min_transacciones = 2
            elif comercio in [10]:  # Telecomunicaciones
                # Servicios de telecomunicaciones - regulares mensuales
                umbral_frecuencia = 35  # ~1 mes
                min_transacciones = 2
            elif comercio in [8, 9]:  # Streaming y Tecnología
                # Suscripciones digitales - muy regulares
                umbral_frecuencia = 32  # ~1 mes
                min_transacciones = 2
            elif comercio in [1, 2]:  # E-commerce y Delivery
                # Compras online y delivery - frecuencia variable
                umbral_frecuencia = 25  # ~3-4 semanas
                min_transacciones = 2
            elif comercio in [12]:  # Gasolineras
                # Combustible - frecuencia regular
                umbral_frecuencia = 20  # ~3 semanas
                min_transacciones = 2
            elif comercio in [15]:  # Fitness
                # Gimnasios - muy regulares
                umbral_frecuencia = 7   # semanal
                min_transacciones = 4
            elif comercio in [17]:  # Apuestas
                # Apuestas - frecuencia variable
                umbral_frecuencia = 15  # ~2 semanas
                min_transacciones = 2
            else:
                # Otros comercios - criterio general (grupos 18, 19, 20)
                umbral_frecuencia = 45
                min_transacciones = 2
            
            # Aplicar filtros específicos
            mask = (
                (datos_comercio['frecuencia_promedio'] <= umbral_frecuencia) &
                (datos_comercio['cantidad_transacciones'] >= min_transacciones) &
                (datos_comercio['desviacion_frecuencia'].fillna(0) <= umbral_frecuencia * 0.7)
            )
            
            datos_filtrados = datos_comercio[mask]
            print(f" >>>> Patrones identificados: {len(datos_filtrados)} de {len(datos_comercio)}")
            
            if len(datos_filtrados) > 0:
                features_filtradas.append(datos_filtrados)
        
        if features_filtradas:
            features_finales = pd.concat(features_filtradas, ignore_index=True)
            print(f"\n -->>>> Características procesadas. Total de patrones: {len(features_finales)}")
            print(f" Distribución por comercio:")
            
            # Diccionario para mapear códigos a nombres descriptivos
            nombres_comercio = {
                1: "E-commerce/Marketplace",
                2: "Delivery/Transporte", 
                3: "Tiendas de Conveniencia",
                4: "Supermercados",
                5: "Clubes de Precio",
                6: "Farmacias",
                7: "Departamentales/Retail",
                8: "Streaming/Entretenimiento",
                9: "Tecnología/Software",
                10: "Telecomunicaciones",
                11: "Servicios Públicos",
                12: "Gasolineras",
                13: "Restaurantes/Comida",
                14: "Fintech/Pagos",
                15: "Fitness/Bienestar",
                16: "Seguros",
                17: "Apuestas/Juegos",
                18: "Redes Sociales",
                19: "Otros Servicios",
                20: "No Clasificado"
            }
            
            distribucion = features_finales['comercio'].value_counts().sort_index()
            for codigo_comercio, cantidad in distribucion.items():
                nombre_comercio = nombres_comercio.get(codigo_comercio, f"Código {codigo_comercio}")
                print(f"   {nombre_comercio} (Código {codigo_comercio}): {cantidad} patrones")
            
            return clientes, features_finales
        else:
            print(" No se encontraron patrones suficientes")
            return clientes, pd.DataFrame()

    except Exception as e:
        print(f" Error en procesamiento: {str(e)}")
        raise

# Optimización de hiperparámetros de LightGBM usando Optuna
def optimizar_lightgbm(X_train, y_train, n_trials=20):
    def objetivo(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'is_unbalance': True,
            'verbosity': -1,
            'random_state': 42
        }
        model = LGBMClassifier(**params)
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objetivo, n_trials=n_trials)
    return study.best_params

# Paso 3: Modelado por comercio, entrenamiento de modelos
def entrenar_modelos_por_comercio(clientes, features):
    print("\n-----> Entrenando modelos por comercio...")
    
    try:
        modelos = {}
        datos_completos = {}
        
        # Diccionario para mapear códigos a nombres descriptivos
        nombres_comercio = {
            1: "E-commerce/Marketplace",
            2: "Delivery/Transporte", 
            3: "Tiendas de Conveniencia",
            4: "Supermercados",
            5: "Clubes de Precio",
            6: "Farmacias",
            7: "Departamentales/Retail",
            8: "Streaming/Entretenimiento",
            9: "Tecnología/Software",
            10: "Telecomunicaciones",
            11: "Servicios Públicos",
            12: "Gasolineras",
            13: "Restaurantes/Comida",
            14: "Fintech/Pagos",
            15: "Fitness/Bienestar",
            16: "Seguros",
            17: "Apuestas/Juegos",
            18: "Redes Sociales",
            19: "Otros Servicios",
            20: "No Clasificado"
        }
        
        datos_base = features.merge(
            clientes[['id', 'edad', 'antiguedad', 'actividad_cod']],
            on='id',
            how='left'
        ).dropna()
        
        for comercio in datos_base['comercio'].unique():
            nombre_comercio = nombres_comercio.get(comercio, f"Código {comercio}")
            print(f"\n Entrenando modelo para: {nombre_comercio} (Código {comercio})")
            
            datos_comercio = datos_base[datos_base['comercio'] == comercio].copy()
            
            if len(datos_comercio) < 10:
                print(f"    Datos insuficientes ({len(datos_comercio)} registros)")
                continue
            
            # Definir objetivo específico por comercio
            hoy = datetime.now()
            datos_comercio['dias_desde_ultima'] = (hoy - datos_comercio['ultima_fecha']).dt.days
            
            # Umbrales adaptativos por comercio
            freq_mediana = datos_comercio['frecuencia_promedio'].median()
            dias_mediana = datos_comercio['dias_desde_ultima'].median()
            
            datos_comercio['objetivo'] = (
                (datos_comercio['cantidad_transacciones'] >= 2) &
                (datos_comercio['frecuencia_promedio'] <= freq_mediana * 1.5) &
                (datos_comercio['dias_desde_ultima'] <= dias_mediana * 1.2)
            ).astype(int)
            
            # Verificar distribución de clases
            class_dist = datos_comercio['objetivo'].value_counts()
            print(f" Distribución: {dict(class_dist)}")
            
            if class_dist.get(1, 0) < 5:
                print(f"   Muy pocos casos positivos, ajustando criterios...")
                datos_comercio['objetivo'] = (
                    datos_comercio['cantidad_transacciones'] >= 2
                ).astype(int)
                class_dist = datos_comercio['objetivo'].value_counts()
                print(f" Nueva distribución: {dict(class_dist)}")
            
            # Preparar features
            feature_cols = [
                'frecuencia_promedio', 'desviacion_frecuencia', 'monto_promedio',
                'consistencia', 'variabilidad_mensual', 'edad', 'antiguedad', 'actividad_cod'
            ]
            
            # Asegurar que todas las columnas existen
            for col in feature_cols:
                if col not in datos_comercio.columns:
                    datos_comercio[col] = 0
            
            X = datos_comercio[feature_cols].fillna(0)
            y = datos_comercio['objetivo']
            
            # Entrenar modelo con manejo inteligente de casos pequeños
            if len(np.unique(y)) > 1:  # Verificar que hay al menos 2 clases
                # Verificar si hay suficientes muestras para estratificación
                class_counts = np.bincount(y)
                min_class_size = np.min(class_counts[class_counts > 0])
                
                if min_class_size < 2:
                    print(f"   Clase minoritaria muy pequeña ({min_class_size} muestra). Usando split simple...")
                    # No usar estratificación para casos muy pequeños
                    if len(X) >= 6:  # Mínimo 6 muestras para hacer split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=42, shuffle=True
                        )
                    else:
                        # Si hay muy pocas muestras, usar todas para entrenamiento
                        print(f"   Muy pocas muestras ({len(X)}). Entrenando con todos los datos...")
                        X_train, y_train = X, y
                        X_test, y_test = X, y  # Usar los mismos datos para evaluación
                
                elif min_class_size >= 2 and len(X) >= 10:
                    # Usar estratificación normal
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                else:
                    # Casos intermedios - split simple
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, shuffle=True
                    )
                
                if len(X_train) < 10:
                    # Modelo simple para pocos datos
                    modelo = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=3,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=1
                    )
                else:
                    # Modelo normal
                    modelo = LGBMClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.05,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1  # Silencia logs
                    )
                
                modelo.fit(X_train, y_train)
                
                # Evaluar modelo
                try:
                    score = modelo.score(X_test, y_test)
                    print(f" Precisión: {score:.3f}")
                except:
                    print(f" Modelo entrenado (evaluación no disponible)")
                
                modelos[comercio] = modelo
                datos_completos[comercio] = datos_comercio
                print(f" -----> Modelo creado exitosamente")
            else:
                print(f"   Solo una clase disponible, modelo no viable")
        
        print(f"\n ------ Modelos entrenados para {len(modelos)} tipos de comercio")

        def evaluar_modelo(modelo, X_test, y_test):
            y_pred = modelo.predict(X_test)
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

        metricas_lgbm = evaluar_modelo(modelo, X_test, y_test)
        print(f" >>>>>>> LightGBM - Precision: {metricas_lgbm['precision']:.3f}, F1: {metricas_lgbm['f1']:.3f}")

        return modelos, datos_completos, metricas_lgbm['f1'], metricas_lgbm['precision']
    
    except Exception as e:
        print(f" Error en entrenamiento: {str(e)}")
        raise

# Pasito 4. Predicción por comercio para un cliente específico
def predecir_gastos_por_comercio(modelos, clientes, features, cliente_id):
    print(f"\n-----> Prediciendo gastos por comercio para cliente {cliente_id}...")
    
    nombres_comercio = {
        1: "E-commerce/Marketplace", 2: "Delivery/Transporte", 3: "Tiendas de Conveniencia",
        4: "Supermercados", 5: "Clubes de Precio", 6: "Farmacias", 7: "Departamentales/Retail",
        8: "Streaming/Entretenimiento", 9: "Tecnología/Software", 10: "Telecomunicaciones",
        11: "Servicios Públicos", 12: "Gasolineras", 13: "Restaurantes/Comida",
        14: "Fintech/Pagos", 15: "Fitness/Bienestar", 16: "Seguros", 17: "Apuestas/Juegos",
        18: "Redes Sociales", 19: "Otros Servicios", 20: "No Clasificado"
    }
    
    try:
        # Verificar existencia del cliente
        if cliente_id not in clientes['id'].values:
            print(f" Cliente {cliente_id} no encontrado")
            return pd.DataFrame()
        
        cliente_data = clientes[clientes['id'] == cliente_id].iloc[0]
        resultados_finales = []
        
        for comercio_cod, modelo in modelos.items():
            nombre_comercio = nombres_comercio.get(comercio_cod, f"Código {comercio_cod}")
            print(f"\n--- Analizando {nombre_comercio}...")
            
            patrones_comercio = features[
                (features['id'] == cliente_id) & 
                (features['comercio'] == comercio_cod)
            ].copy()
            
            if patrones_comercio.empty:
                print(f" x--x Sin historial en {nombre_comercio}")
                continue
            
            # Preparar datos para predicción
            feature_cols = [
                'frecuencia_promedio', 'desviacion_frecuencia', 'monto_promedio',
                'consistencia', 'variabilidad_mensual'
            ]
            
            X_pred = patrones_comercio[feature_cols].fillna(0).copy()
            X_pred['edad'] = cliente_data['edad']
            X_pred['antiguedad'] = cliente_data['antiguedad']
            X_pred['actividad_cod'] = cliente_data['actividad_cod']
            
            # Realizar predicción
            if hasattr(modelo, 'predict_proba'):
                probabilidad = modelo.predict_proba(X_pred)[:, 1][0]
            else:
                probabilidad = modelo.predict(X_pred)[0]
            
            ultima_fecha = patrones_comercio['ultima_fecha'].iloc[0]
            frecuencia_promedio = patrones_comercio['frecuencia_promedio'].iloc[0]
            fecha_estimada = ultima_fecha + pd.Timedelta(days=frecuencia_promedio)
            
            resultado = {
                'comercio_codigo': comercio_cod,
                'comercio_nombre': nombre_comercio,
                'probabilidad': probabilidad,
                'fecha_estimada': fecha_estimada,
                'monto_promedio': patrones_comercio['monto_promedio'].iloc[0],
                'frecuencia_dias': frecuencia_promedio,
                'ultima_compra': ultima_fecha,
                'total_transacciones': patrones_comercio['cantidad_transacciones'].iloc[0],
                'monto_total_historico': patrones_comercio['monto_total'].iloc[0]
            }
            
            resultados_finales.append(resultado)
            print(f"-----> Probabilidad: {probabilidad:.2%}")
            
        if resultados_finales:
            df_resultados = pd.DataFrame(resultados_finales)
            df_resultados = df_resultados.sort_values('probabilidad', ascending=False)
            return df_resultados
        else:
            print(" No se encontraron predicciones para ningún comercio")
            return pd.DataFrame()
    
    except Exception as e:
        print(f" Error en predicción: {str(e)}")
        return pd.DataFrame()

# Pason 5. Gráficas
def visualizar_predicciones_por_comercio(resultados, cliente_id):
    if resultados.empty:
        print(" No hay resultados para visualizar")
        return

    

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resultados)))
    plt.style.use('ggplot')  # Estilo profesional


    # Gráfico 1: Probabilidades de gasto recurrente
    graph1 = plt.figure(1, figsize=(10, 6))
    bars1 = plt.barh(resultados['comercio_nombre'], resultados['probabilidad']*100, 
                     color=colors, alpha=0.8)
    plt.xlabel('Probabilidad (%)', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for i, (_, row) in enumerate(resultados.iterrows()):
        plt.text(row['probabilidad']*100 + 1, i, f"{row['probabilidad']*100:.1f}%", 
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    

    # Gráfico 2: Montos promedio
    graph2 = plt.figure(2, figsize=(10, 6))
    bars2 = plt.bar(range(len(resultados)), resultados['monto_promedio'], 
                    color=colors, alpha=0.8)
    plt.xticks(range(len(resultados)), resultados['comercio_nombre'], 
              rotation=45, ha='right', fontsize=9)
    plt.ylabel('Monto Promedio ($)', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for i, (_, row) in enumerate(resultados.iterrows()):
        plt.text(i, row['monto_promedio'] + 0.5, f"${row['monto_promedio']:.2f}", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    

    # Gráfico 3: Frecuencia de compras
    graph3 = plt.figure(3, figsize=(10, 6))
    bars3 = plt.bar(range(len(resultados)), resultados['frecuencia_dias'], 
                    color=colors, alpha=0.8)
    plt.xticks(range(len(resultados)), resultados['comercio_nombre'], 
              rotation=45, ha='right', fontsize=9)
    plt.ylabel('Días entre compras', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for i, (_, row) in enumerate(resultados.iterrows()):
        plt.text(i, row['frecuencia_dias'] + 0.5, f"{row['frecuencia_dias']:.0f} días", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    

    print(f" ---->>>> RESUMEN DETALLADO - CLIENTE {cliente_id}")
    
    for _, row in resultados.iterrows():
        print(f"\n {row['comercio_nombre'].upper()}")
        print(f"   Probabilidad de gasto recurrente: {row['probabilidad']:.1%}")
        print(f"   Monto promedio: ${row['monto_promedio']:.2f}")
        print(f"   Próxima compra estimada: {row['fecha_estimada'].strftime('%d/%m/%Y')}")
        print(f"   Frecuencia: cada {row['frecuencia_dias']:.0f} días")
        print(f"   Total transacciones históricas: {row['total_transacciones']}")
        print(f"   Gasto total histórico: ${row['monto_total_historico']:.2f}")

    return graph1, graph2, graph3

# Análisis completo de un cliente con predicciones
def analizar_cliente_completo(modelos, clientes, features, cliente_id):
    nombres_comercio = {
        1: "E-commerce/Marketplace", 2: "Delivery/Transporte", 3: "Tiendas de Conveniencia",
        4: "Supermercados", 5: "Clubes de Precio", 6: "Farmacias", 7: "Departamentales/Retail",
        8: "Streaming/Entretenimiento", 9: "Tecnología/Software", 10: "Telecomunicaciones",
        11: "Servicios Públicos", 12: "Gasolineras", 13: "Restaurantes/Comida",
        14: "Fintech/Pagos", 15: "Fitness/Bienestar", 16: "Seguros", 17: "Apuestas/Juegos",
        18: "Redes Sociales", 19: "Otros Servicios", 20: "No Clasificado"
    }
    
    print(f" ----->>>> ANÁLISIS COMPLETO CLIENTE {cliente_id}")
    
    if cliente_id in clientes['id'].values:
        cliente_info = clientes[clientes['id'] == cliente_id].iloc[0]
        print(f" Edad: {cliente_info['edad']} años")
        print(f" Antigüedad: {cliente_info['antiguedad']} meses")
        print(f" Actividad: {cliente_info['actividad_empresarial']}")
    
    # Historial
    comercios_cliente = features[features['id'] == cliente_id]['comercio'].unique()
    print(f" --- Comercios frecuentados: {len(comercios_cliente)}")
    for comercio_cod in comercios_cliente:
        nombre = nombres_comercio.get(comercio_cod, f"Código {comercio_cod}")
        print(f"  • {nombre}")
    
    # Predicciones
    resultados = predecir_gastos_por_comercio(modelos, clientes, features, cliente_id)
    
    if not resultados.empty:
        graph1, graph2, graph3 = visualizar_predicciones_por_comercio(resultados, cliente_id)
        return resultados, graph1, graph2, graph3, resultados
    else:
        print(" No se pudieron generar predicciones para este cliente")
        return pd.DataFrame()

# Método mandado a llamarse por la WebAPP
def runPredictiveModel(index):
    try:      
        clientes, transacciones = cargar_datos(
            ruta_clientes="DcA_FINAL/base_clientes_final.csv",
            ruta_transacciones="DcA_FINAL/base_transacciones_final.csv"
        )
        
        clientes, features = procesar_caracteristicas_por_comercio(clientes, transacciones)

        if features.empty:
            print(" No hay datos suficientes para entrenar modelos")
            exit()
        
        modelos, datos_completos, f1s, prcs = entrenar_modelos_por_comercio(clientes, features)
        
        if not modelos:
            print(" No se pudieron entrenar modelos")
            exit()
    
        clientes_disponibles = features['id'].unique()
        if len(clientes_disponibles) > 0:
            cliente_ejemplo = clientes_disponibles[index]
            print(f"\n ----- Realizando análisis completo para cliente ejemplo: {cliente_ejemplo}")
            
            resultados, graph1, graph2, graph3, resultados = analizar_cliente_completo(modelos, clientes, features, cliente_ejemplo)
            
            # Mostrar top 3 comercios con mayor probabilidad
            if not resultados.empty:
                print(f"\n >>>> TOP 3 COMERCIOS CON MAYOR PROBABILIDAD:")
                top_3 = resultados.head(3)
                for i, (_, row) in enumerate(top_3.iterrows(), 1):
                    print(f"{i}. {row['comercio_nombre']}: {row['probabilidad']:.1%} "
                        f"(${row['monto_promedio']:.2f} cada {row['frecuencia_dias']:.0f} días)")
        
        print("\n ***** Análisis completado exitosamente")
        return graph1, graph2, graph3, cliente_ejemplo, f1s, prcs, resultados
    
    except Exception as e:
        print(f"\n ERROR CRÍTICO: {str(e)}")
        exit()