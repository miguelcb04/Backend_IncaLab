# import pymysql
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from CoolProp.CoolProp import PropsSI

# MODELOS_STD = {
#     't_ev': 2, 't_cd': 2, 'rec': 2, 'subf': 2, 'dt_ev': 1,
#     'dt_cd': 1, 'dta_evap': 1, 'dta_cond': 1, 'cop': 0.5,
#     'ef_comp': 0.05, 'pot_frig': 100
# }

# def get_connection():
#     return pymysql.connect(
#         host="5.134.116.201",
#         port=3306,
#         user="juandeeu_digital",
#         password="ContraseN.4",
#         database="juandeeu_db"
#     )

# def fetch_last_record():
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT
#             fecha, pa, pb, t_asp, t_des, t_liq,
#             ta_in_cond AS t_amb,
#             ta_in_evap AS t_cam,
#             ta_out_cond, ta_out_evap, pot_abs
#         FROM incalab
#         ORDER BY id DESC
#         LIMIT 1
#     """)
#     row = cur.fetchone()
#     conn.close()
#     keys = ['fecha','pa','pb','t_asp','t_des','t_liq','t_amb','t_cam','ta_out_cond','ta_out_evap','pot_abs']
#     return dict(zip(keys, row))

# def fetch_historical_records(limit=1800):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT fecha, pa, pb, t_asp, t_des, t_liq,
#                ta_in_cond AS t_amb, ta_in_evap AS t_cam,
#                ta_out_cond, ta_out_evap, pot_abs
#         FROM incalab
#         ORDER BY id DESC
#         LIMIT %s
#     """, (limit,))
#     rows = cur.fetchall()
#     columns = ['fecha','pa','pb','t_asp','t_des','t_liq','t_amb','t_cam','ta_out_cond','ta_out_evap','pot_abs']
#     conn.close()
#     df = pd.DataFrame(rows, columns=columns)
#     df = df.fillna(0)
#     return df.to_dict(orient="records")


# # def fetch_historical_records(limit=300000):
# #     conn = get_connection()
# #     cur = conn.cursor()
# #     cur.execute("""
# #         SELECT fecha, pa, pb, t_asp, t_des, t_liq,
# #                ta_in_cond AS t_amb, ta_in_evap AS t_cam,
# #                ta_out_cond, ta_out_evap, pot_abs
# #         FROM incalab
# #         ORDER BY id DESC
# #         LIMIT %s
# #     """, (limit,))
# #     rows = cur.fetchall()
# #     columns = ['fecha','pa','pb','t_asp','t_des','t_liq','t_amb','t_cam','ta_out_cond','ta_out_evap','pot_abs']
# #     conn.close()
# #     df = pd.DataFrame(rows, columns=columns)
# #     df = df.fillna(0)
# #     return df.to_dict(orient="records")

# def calcular_variables(df: pd.DataFrame):
#     df = df.copy()
#     for col in df.columns:
#         if col != "fecha":
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     df = df.fillna(0)

#     pa = df.iloc[0]["pa"]
#     pb = df.iloc[0]["pb"]
#     t_asp = df.iloc[0]["t_asp"]
#     t_liq = df.iloc[0]["t_liq"]
#     t_des = df.iloc[0]["t_des"]
#     t_cam = df.iloc[0]["t_cam"]
#     t_amb = df.iloc[0]["t_amb"]
#     ta_out_evap = df.iloc[0]["ta_out_evap"]
#     ta_out_cond = df.iloc[0]["ta_out_cond"]
#     pot_abs = df.iloc[0]["pot_abs"]

#     t_ev = np.round(PropsSI('T', 'P', (pb + 1) * 1e5, 'Q', 1, 'R290') - 273.15, 2)
#     t_cd = np.round(PropsSI('T', 'P', (pa + 1) * 1e5, 'Q', 0, 'R290') - 273.15, 2)

#     t_asp = max(t_asp, t_ev)
#     t_liq = min(t_liq, t_cd)

#     rec = np.round(t_asp - t_ev, 2)
#     subf = np.round(t_cd - t_liq, 2)
#     dt_ev = np.round(t_cam - t_ev, 2)
#     dt_cd = np.round(t_cd - t_amb, 2)
#     dta_evap = np.round(t_cam - ta_out_evap, 2)
#     dta_cond = np.round(ta_out_cond - t_amb, 2)

#     def entalpia(P_bar, T_C):
#         return PropsSI('H', 'P', (P_bar + 1) * 1e5, 'T', T_C + 273.15, 'R290') / 1000

#     h_liq = entalpia(pa, t_liq)
#     h_asp = entalpia(pb, t_asp)
#     h_des = entalpia(pa, t_des)

#     try:
#         cop_val = (h_asp - h_liq) / (h_des - h_asp)
#     except ZeroDivisionError:
#         cop_val = 0
#     cop = np.round(cop_val, 2)

#     try:
#         s_asp = PropsSI('S', 'P', (pb + 1) * 1e5, 'T', t_asp + 273.15, 'R290') / 1000
#         h_des_iso = PropsSI('H', 'P', (pa + 1) * 1e5, 'S', s_asp * 1000, 'R290') / 1000
#         ef_comp_val = (h_des_iso - h_asp) / (h_des - h_asp)
#     except:
#         ef_comp_val = 0
#     ef_comp = np.round(ef_comp_val, 3)

#     pot_frig = np.round(pot_abs * cop, 1)

#     resultado = {
#         't_ev': t_ev, 't_cd': t_cd, 'rec': rec, 'subf': subf,
#         'dt_ev': dt_ev, 'dt_cd': dt_cd, 'dta_evap': dta_evap,
#         'dta_cond': dta_cond, 'cop': cop, 'ef_comp': ef_comp,
#         'pot_frig': pot_frig, 'pot_abs': pot_abs, 'pa': pa, 'pb': pb,
#         't_asp': t_asp, 't_liq': t_liq, 't_des': t_des, 't_cam': t_cam,
#         't_amb': t_amb, 'ta_out_evap': ta_out_evap, 'ta_out_cond': ta_out_cond
#     }
#     if 'fecha' in df.columns:
#         resultado['fecha'] = df.iloc[0]['fecha']

#     return resultado

# def calcular_tabla_comparativa(df: pd.DataFrame):
#     registros = []

#     for _, row in df.iterrows():
#         fila = pd.DataFrame([row])

#         for col in fila.columns:
#             if col != "fecha":
#                 fila[col] = pd.to_numeric(fila[col], errors="coerce")
#         fila = fila.fillna(0)

#         real = calcular_variables(fila)
#         esperado = {k: round(real[k] * 0.95, 2) if k in MODELOS_STD else real[k] for k in real}

#         real['registro'] = 'real'
#         esperado['registro'] = 'esperado'
#         registros.append(real)
#         registros.append(esperado)

#         desviacion = {'registro': 'desviación'}
#         for var in MODELOS_STD:
#             desviacion[var] = round(real[var] - esperado[var], 2)
#         registros.append(desviacion)

#         n_sd = {'registro': 'n_sd'}
#         for var in MODELOS_STD:
#             std = MODELOS_STD[var]
#             n_sd[var] = round(desviacion[var] / std if std != 0 else 0, 2)
#         registros.append(n_sd)

#     return registros



import os
import pickle
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
import pymysql

MODELS_DIR = "models_vee"
MODELOS_PKL = {
    't_ev': 'model_t_ev.pkl',
    't_cd': 'model_t_cd.pkl',
    'rec': 'model_rec.pkl',
    't_des': 'model_t_des.pkl',
    'subf': 'model_subf.pkl',
    'dta_evap': 'model_dta_evap.pkl',
    'dta_cond': 'model_dta_cond.pkl',
    'pot_abs': 'model_pot_abs.pkl',
    'cop': 'model_cop.pkl',
    'ef_comp': 'model_ef_comp.pkl'
}
MODELOS_STD = {
    't_ev': 2, 't_cd': 2, 'rec': 2, 't_des': 5, 'subf': 2, 'dt_ev': 1,
    'dt_cd': 1, 'dta_evap': 1, 'dta_cond': 1, 'pot_abs': 20, 'pot_frig': 100,
    'cop': 0.5, 'ef_comp': 0.05
}

def get_connection():
    return pymysql.connect(
        host="5.134.116.201",
        port=3306,
        user="juandeeu_digital",
        password="ContraseN.4",
        database="juandeeu_db"
    )

def fetch_last_record():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            fecha, pa, pb, t_asp, t_des, t_liq,
            ta_in_cond AS t_amb,
            ta_in_evap AS t_cam,
            ta_out_cond, ta_out_evap, pot_abs
        FROM incalab
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    keys = ['fecha','pa','pb','t_asp','t_des','t_liq','t_amb','t_cam','ta_out_cond','ta_out_evap','pot_abs']
    return dict(zip(keys, row))

def fetch_historical_records(limit=1800):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT fecha, pa, pb, t_asp, t_des, t_liq,
               ta_in_cond AS t_amb, ta_in_evap AS t_cam,
               ta_out_cond, ta_out_evap, pot_abs
        FROM incalab
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    columns = ['fecha','pa','pb','t_asp','t_des','t_liq','t_amb','t_cam','ta_out_cond','ta_out_evap','pot_abs']
    conn.close()
    df = pd.DataFrame(rows, columns=columns)
    df = df.fillna(0)
    return df.to_dict(orient="records")

def cargar_modelo(nombre):
    path = os.path.join(MODELS_DIR, nombre)
    with open(path, 'rb') as f:
        return pickle.load(f)

modelos = {k: cargar_modelo(v) for k, v in MODELOS_PKL.items()}

def calcular_variables(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        if col != "fecha":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    df['t_ev'] = np.round(PropsSI('T', 'P', (df['pb'] + 1)*1e5, 'Q', 1, 'R290') - 273.15, 2)
    df['t_cd'] = np.round(PropsSI('T', 'P', (df['pa'] + 1)*1e5, 'Q', 0, 'R290') - 273.15, 2)
    df['t_asp'] = np.maximum(df['t_asp'], df['t_ev'])
    df['t_liq'] = np.minimum(df['t_liq'], df['t_cd'])
    df['rec'] = df['t_asp'] - df['t_ev']
    df['subf'] = df['t_cd'] - df['t_liq']
    df['dt_ev'] = df['t_cam'] - df['t_ev']
    df['dt_cd'] = df['t_cd'] - df['t_amb']
    df['dta_evap'] = df['t_cam'] - df['ta_out_evap']
    df['dta_cond'] = df['ta_out_cond'] - df['t_amb']
    df['cop'] = (df['rec'] / 10) + 3.0
    df['ef_comp'] = 0.85 + 0.01 * (df['t_asp'] - df['t_amb'])
    df['pot_frig'] = df['pot_abs'] * df['cop']

    return df.iloc[0].to_dict()

def calcular_tabla_comparativa(df: pd.DataFrame):
    fila = df.copy().fillna(0)
    real = calcular_variables(fila)
    X = pd.DataFrame([real])

    esperado = {
        't_ev': modelos['t_ev'].predict(X[['t_cam', 't_amb']])[0],
        't_cd': modelos['t_cd'].predict(X[['t_cam', 't_amb']])[0],
    }
    X['t_ev'], X['t_cd'] = esperado['t_ev'], esperado['t_cd']
    esperado['rec'] = modelos['rec'].predict(X[['t_cam', 't_amb', 't_ev', 't_cd']])[0]
    X['rec'] = esperado['rec']
    esperado['t_des'] = modelos['t_des'].predict(X[['t_ev', 't_cd', 'rec', 't_amb']])[0]
    X['t_des'] = esperado['t_des']
    esperado['subf'] = modelos['subf'].predict(X[['t_amb', 't_cd', 't_des']])[0]
    esperado['dta_evap'] = modelos['dta_evap'].predict(X[['t_cam', 't_ev']])[0]
    esperado['dta_cond'] = modelos['dta_cond'].predict(X[['t_cd', 't_amb']])[0]
    esperado['pot_abs'] = modelos['pot_abs'].predict(X[['t_des', 't_ev', 't_cd', 't_amb', 't_cam']])[0]
    esperado['cop'] = modelos['cop'].predict(X[['t_ev', 't_cd', 't_des', 'rec']])[0]
    esperado['ef_comp'] = modelos['ef_comp'].predict(X[['t_ev', 't_cd', 't_des', 't_amb', 't_cam']])[0]
    esperado['dt_ev'] = X['t_cam'].iloc[0] - esperado['t_ev']
    esperado['dt_cd'] = esperado['t_cd'] - X['t_amb'].iloc[0]
    esperado['pot_frig'] = esperado['cop'] * esperado['pot_abs']

    real['registro'] = 'real'
    esperado['registro'] = 'esperado'
    desviacion = {'registro': 'desviación'}
    n_sd = {'registro': 'n_sd'}

    for key in MODELOS_STD:
        desviacion[key] = round(real.get(key, 0) - esperado.get(key, 0), 2)
        n_sd[key] = round(desviacion[key] / MODELOS_STD[key], 2)

    return [real, esperado, desviacion, n_sd]
