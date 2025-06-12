import os, gdown, pickle

MODEL_DIR = "models_vee"
os.makedirs(MODEL_DIR, exist_ok=True)

MODELOS_DRIVE = {
    'model_t_ev.pkl':    '18NG91i8EyCr8TkaxIFYh1RS5ujtqIwCx',
    'model_t_cd.pkl':    '18vZzV90q8Vt628cVVv5tCGQWYOzEBtMf',
    'model_rec.pkl':     '1KJJLNqysxW1PR2Ptl0KZ1grawJCxcqmW',
    'model_t_des.pkl':   '1r58Ab80yNa1A9ejRN4YXYHD0yL7ktZiQ',
    'model_subf.pkl':    '11Gp7cnknk-qsao51VLPq0lvgMh3UVSvc',
    'model_dta_evap.pkl':'1dr0DHRch8aeabjL57CbZREI8H8xujthx',
    'model_dta_cond.pkl':'1P-CCsm_bL_k-kdqPw34iskYUPmUIp9uY',
    'model_pot_abs.pkl': '1mRv5Fiw7K7j8DLUvEnqJ2q54vcR0Cq9i',
    'model_cop.pkl':     '1aJuLDtz5bcZK7r2L2r34UUYyBLuq9EC2',
    'model_ef_comp.pkl': '148EVPmYV5xK8Hp7jYHQijsSdcuCfS9Ia'
}

# Descargar y cargar todos los modelos en un dict
models = {}
for fname, file_id in MODELOS_DRIVE.items():
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    key = fname.replace('.pkl','')
    with open(path,'rb') as f:
        models[key] = pickle.load(f)