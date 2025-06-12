from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime 
from .services import (
    fetch_last_record,
    fetch_historical_records,
    calcular_variables,
    calcular_tabla_comparativa
)

app = FastAPI(title="Detector R290 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    pa: float
    pb: float
    t_asp: float
    t_des: float
    t_liq: float
    ta_in_cond: float
    ta_in_evap: float
    ta_out_cond: float
    ta_out_evap: float
    pot_abs: float
    t_amb: float
    t_cam: float

@app.get("/")
def root():
    return {"status": "ok", "cors": "aplicado correctamente"}

@app.get("/predict")
def predict_get():
    try:
        data = fetch_last_record()
        df = pd.DataFrame([data])
        df["fecha"] = data.get("fecha", datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        return calcular_variables(df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/predict")
def predict_post(payload: Payload):
    try:
        df = pd.DataFrame([payload.dict()])
        return calcular_variables(df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/registros")
def registros():
    try:
        raw_data = fetch_historical_records()
        enriched = []
        for row in raw_data:
            try:
                df = pd.DataFrame([row])
                calculado = calcular_variables(df)
                if "error" not in calculado:
                    calculado["fecha"] = row["fecha"]
                    enriched.append(calculado)
            except Exception as e:
                print(f"[ERROR EN FILA]: {e}")
        return enriched
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}




@app.get("/tabla-comparativa")
def tabla_comparativa():
    try:
        data = fetch_last_record()
        df = pd.DataFrame([data])
        df["fecha"] = data.get("fecha", datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        tabla = calcular_tabla_comparativa(df)
        return tabla
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    

