import pandas as pd, numpy as np, os, sqlite3

metadatos = ['id_ubicacion', 'especie', 'tratamiento', 'salinidad', 'nombre_archivo', 'estadio_larval', 'obs',
             'video_analizado', 'larva_unica']
datos_trackeo = ['t', 'x', 'y']
coordenadas_centradas = {'x_centrada', 'y_centrada'}
variables_base = [
    'r', 'r_centrado', 'v', 'v_centrado', 'a', 'a_centrado',
    'dr', 'dr_centrado', 'theta_r', 'theta_r_centrado',
    'v_angular', 'v_angular_centrado', 'a_angular', 'a_angular_centrado',
    'curvatura', 'curvatura_centrada', 'theta_v', 'theta_v_centrado']

variables_promedio = [
    'v_promedio', 'v_centrado_promedio',
    'a_promedio', 'a_centrado_promedio',
    'r_promedio', 'r_centrado_promedio',
    'dr_promedio', 'dr_centrado_promedio',
    'theta_r_promedio', 'theta_r_centrado_promedio',
    'v_angular_promedio', 'v_angular_centrado_promedio',
    'a_angular_promedio', 'a_angular_centrado_promedio',
    'curvatura_promedio', 'curvatura_centrada_promedio',
    'theta_v_promedio', 'theta_v_centrado_promedio']

variables_mediana = [
    'v_mediana', 'v_centrado_mediana',
    'a_mediana', 'a_centrado_mediana',
    'r_mediana', 'r_centrado_mediana',
    'dr_mediana', 'dr_centrado_mediana',
    'theta_r_mediana', 'theta_r_centrado_mediana',
    'v_angular_mediana', 'v_angular_centrado_mediana',
    'a_angular_mediana', 'a_angular_centrado_mediana',
    'curvatura_mediana', 'curvatura_centrada_mediana',
    'theta_v_mediana', 'theta_v_centrado_mediana']

variables_resumen = list(set(variables_promedio).union(set(variables_mediana)))

variables_ordenadas = [
    'r_promedio', 'r_mediana',
    'r_centrado_promedio', 'r_centrado_mediana',

    'v_promedio', 'v_mediana',
    'v_centrado_promedio', 'v_centrado_mediana',

    'a_promedio', 'a_mediana',
    'a_centrado_promedio', 'a_centrado_mediana',

    'dr_promedio', 'dr_mediana',
    'dr_centrado_promedio', 'dr_centrado_mediana',

    'theta_r_promedio', 'theta_r_mediana',
    'theta_r_centrado_promedio', 'theta_r_centrado_mediana',

    'v_angular_promedio', 'v_angular_mediana',
    'v_angular_centrado_promedio', 'v_angular_centrado_mediana',

    'a_angular_promedio', 'a_angular_mediana',
    'a_angular_centrado_promedio', 'a_angular_centrado_mediana',

    'theta_v_promedio', 'theta_v_mediana',
    'theta_v_centrado_promedio', 'theta_v_centrado_mediana',

    'curvatura_promedio', 'curvatura_mediana',
    'curvatura_centrada_promedio', 'curvatura_centrada_mediana']

#para obtener la ruta absoluta del archivo de base de datos, independientemente del sistema operativo
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_DB = os.path.join(BASE_DIR, "data", "data.db")
if not os.path.exists(RUTA_DB):
    RUTA_DB = os.path.join(BASE_DIR, "data", "data_simulada.db")

QUERY_TRACKEO_TODOS = 'SELECT * FROM datos_trackeo'
QUERY_RESUMEN_TODOS = 'SELECT * FROM datos_resumen'
QUERY_TRACKEO_ANALIZADOS = 'SELECT * FROM datos_trackeo WHERE video_analizado = 1'
QUERY_RESUMEN_ANALIZADOS = 'SELECT * FROM datos_resumen WHERE video_analizado = 1'

def definir_dtypes(df_input):
    df = df_input.copy()
    tipos = {'nombre_archivo': 'string'}
    for col in metadatos:
        tipos[col] = 'category'
    for col in datos_trackeo + variables_resumen:
        tipos[col] = 'float64'
    tipos_filtrados = {col: dtype for col, dtype in tipos.items() if col in df.columns}
    return df.astype(tipos_filtrados)

def calcular_coordenadas_centradas(df_input): #debo hacer esta función antes de agregar info angular
    df = df_input.copy()
    cols_interp = ['x', 'y', 't'] #interpolo los valores de columnas esenciales (por ahora)
    for col in cols_interp:
        if col in df.columns:
            df[col] = df.groupby('id')[col].transform(lambda s: s.interpolate(method='pchip')) #asegura suavidad de la primera derivada al menos, mejor que lineal
    if 'id' in df.columns and 'x' in df.columns and 'y' in df.columns:
        df['x_centrada'] = df.groupby('id')['x'].transform(lambda x: x - x.iloc[0])
        df['y_centrada'] = df.groupby('id')['y'].transform(lambda y: y - y.iloc[0])
    else:
        print("Columnas no encontradas. Revisar.")
    return df

def agregar_datos_calculados(df_input):
    df = df_input.copy()
    columnas_necesarias = ['id', 't', 'x', 'y', 'x_centrada', 'y_centrada']
    if not all(col in df.columns for col in columnas_necesarias):
        raise KeyError(f"Faltan columnas en el dataframe: {set(columnas_necesarias) - set(df.columns)}")

    df = df.sort_values(['id', 't'])

    def _procesar_grupo(gr):
        n = len(gr)
        t = gr['t'].values
        x = gr['x'].values
        y = gr['y'].values
        x_centr = gr['x_centrada'].values
        y_centr = gr['y_centrada'].values

        # Velocidad y aceleración
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)

        vx_c = np.gradient(x_centr, t)
        vy_c = np.gradient(y_centr, t)
        ax_c = np.gradient(vx_c, t)
        ay_c = np.gradient(vy_c, t)

        v = np.sqrt(vx**2 + vy**2)
        a = np.sqrt(ax**2 + ay**2)
        v_centrado = np.sqrt(vx_c**2 + vy_c**2)
        a_centrado = np.sqrt(ax_c**2 + ay_c**2)

        r = np.sqrt(x**2 + y**2)
        r_centrado = np.sqrt(x_centr**2 + y_centr**2)
        dr = np.gradient(r, t)
        dr_centrado = np.gradient(r_centrado, t)

        theta_r = np.arctan2(y, x)
        theta_r_centrado = np.arctan2(y_centr, x_centr)

        theta_r_unwrapped = np.unwrap(theta_r)
        theta_r_centrado_unwrapped = np.unwrap(theta_r_centrado)

        v_ang = np.gradient(theta_r_unwrapped, t)
        a_ang = np.gradient(v_ang, t)

        v_ang_centrado = np.gradient(theta_r_centrado_unwrapped, t)
        a_ang_centrado = np.gradient(v_ang_centrado, t)

        theta_v = np.arctan2(vy, vx)
        theta_v_centrado = np.arctan2(vy_c, vx_c)

        # Curvatura (como ya lo tenías)
        x_m1, x_0, x_p1 = x[:-2], x[1:-1], x[2:]
        y_m1, y_0, y_p1 = y[:-2], y[1:-1], y[2:]
        lado_a = np.sqrt((x_p1 - x_0)**2 + (y_p1 - y_0)**2)
        b = np.sqrt((x_p1 - x_m1)**2 + (y_p1 - y_m1)**2)
        c = np.sqrt((x_0 - x_m1)**2 + (y_0 - y_m1)**2)
        s = (lado_a + b + c) / 2
        area_arg = s * (s - lado_a) * (s - b) * (s - c)
        area = np.sqrt(np.maximum(area_arg, 0))
        area = np.maximum(area, 1e-10)
        R = (lado_a * b * c) / (4 * area)
        k = np.divide(1, R, out=np.zeros_like(R), where=R != 0)
        k[(lado_a + b <= c) | (lado_a + c <= b) | (b + c <= lado_a)] = 0
        curvatura = np.empty(n)
        curvatura[:] = np.nan
        curvatura[1:-1] = k
        curvatura = np.where((curvatura > 1e-6) & (curvatura < 5000), curvatura, 0)

        # Curvatura centrada (mismo cálculo pero usando coordenadas centradas)
        x_m1, x_0, x_p1 = x_centr[:-2], x_centr[1:-1], x_centr[2:]
        y_m1, y_0, y_p1 = y_centr[:-2], y_centr[1:-1], y_centr[2:]
        lado_a = np.sqrt((x_p1 - x_0)**2 + (y_p1 - y_0)**2)
        b = np.sqrt((x_p1 - x_m1)**2 + (y_p1 - y_m1)**2)
        c = np.sqrt((x_0 - x_m1)**2 + (y_0 - y_m1)**2)
        s = (lado_a + b + c) / 2
        area_arg = s * (s - lado_a) * (s - b) * (s - c)
        area = np.sqrt(np.maximum(area_arg, 0))
        area = np.maximum(area, 1e-10)
        R = (lado_a * b * c) / (4 * area)
        k = np.divide(1, R, out=np.zeros_like(R), where=R != 0)
        k[(lado_a + b <= c) | (lado_a + c <= b) | (b + c <= lado_a)] = 0
        curvatura_centrada = np.empty(n)
        curvatura_centrada[:] = np.nan
        curvatura_centrada[1:-1] = k
        curvatura_centrada = np.where((curvatura_centrada > 1e-6) & (curvatura_centrada < 5000), curvatura_centrada, 0)

        # Asignar variables nuevas
        gr = gr.assign(
            v=v, v_centrado=v_centrado,
            a=a, a_centrado=a_centrado,
            r=r, r_centrado=r_centrado,
            dr=dr, dr_centrado=dr_centrado,
            theta_r=theta_r_unwrapped, theta_r_centrado=theta_r_centrado_unwrapped,
            v_angular=v_ang, v_angular_centrado=v_ang_centrado,
            a_angular=a_ang, a_angular_centrado=a_ang_centrado,
            theta_v=theta_v, theta_v_centrado=theta_v_centrado,
            curvatura=curvatura, curvatura_centrada=curvatura_centrada)
        return gr

    df = df.groupby('id', group_keys=False).apply(_procesar_grupo).reset_index(drop=True)
    return df

def resumir_por_id(df):
    df = df.copy()
    columnas = variables_base
    estadisticas=('mean', 'median')
    sufijos = {'mean': 'promedio', 'median': 'mediana'}
    df_resumen = df.groupby('id')[columnas].agg(estadisticas)
    df_resumen.columns = [f"{col}_{sufijos.get(stat, stat)}" for col, stat in df_resumen.columns]
    df_resumen = df_resumen.reset_index()
    df_resumen = df_resumen.apply(pd.to_numeric, errors='coerce')
    return df_resumen
    
def cargar_datos(ruta_db=RUTA_DB,
                       query_trackeo=QUERY_TRACKEO_ANALIZADOS,
                       query_resumen=QUERY_RESUMEN_ANALIZADOS,
                       incluir_resumen=True):
    with sqlite3.connect(ruta_db) as conn:
        df_trackeo = pd.read_sql_query(query_trackeo, conn)
        if incluir_resumen:
            df_resumen = pd.read_sql_query(query_resumen, conn)

    print(f"Datos cargados. Trackeo: {df_trackeo.shape}, Resumen: {df_resumen.shape if incluir_resumen else 'No incluido'}")
    return (pd.DataFrame(df_trackeo), pd.DataFrame(df_resumen)) if incluir_resumen else df_trackeo

def obtener_orden_salinidades(df):
    if 'salinidad' in df.columns:
        return sorted(df['salinidad'].unique())
    else:
        print("Columna 'salinidad' no encontrada.")
        return []

def obtener_orden_niveles_salinidad(df):
    if 'nivel_salinidad' in df.columns:
        return ['baja', 'normal', 'alta']
    else:
        print("Columna 'nivel_salinidad' no encontrada.")
        return []

def clasificar_salinidad(salinidad_str):
    if salinidad_str in ["40", "45", "50", "55", "60"]:
        return "alta"
    elif salinidad_str in ["35-N", "35-R", "35"]:
        return "normal"
    elif salinidad_str in ['15', '20', '25', '30']:
        return "baja"
    else:
        return None




