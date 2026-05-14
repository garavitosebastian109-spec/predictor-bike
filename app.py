# ============================================================
#  BACKEND — Flask API
#  Dataset: Bike Sharing (UCI) — Clasificación binaria
#
#  Estructura:
#    bike_predictor/
#    ├── app.py                  ← Este archivo
#    ├── regresion_logistica.py  ← Entrena modelo RL
#    ├── red_neuronal.py         ← Entrena modelo RNA
#    ├── dataset.csv             ← Dataset original
#    ├── feature_names.json      ← Generado al entrenar
#    ├── static/
#    │   ├── style.css
#    │   └── script.js
#    └── templates/
#        └── index.html
#
#  PASOS:
#  1. Entrena los modelos:
#       python regresion_logistica.py
#       python red_neuronal.py
#  2. Corre el servidor:
#       python app.py
#  3. Abre: http://127.0.0.1:5000
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
#  RUTAS DE MODELOS
# ─────────────────────────────────────────────────────────────
ARCHIVOS = {
    "rl": {
        "modelo": "modelo_rl.pkl",
        "scaler": "scaler_rl.pkl",
    },
    "rna": {
        "modelo": "modelo_rna.h5",
        "scaler": "scaler_rna.pkl",
    }
}

_cache = {}   # Cache para no recargar el modelo en cada petición


def cargar_modelo(tipo):
    """Carga modelo y scaler. Usa cache para no recargar cada vez."""
    if tipo in _cache:
        return _cache[tipo]["modelo"], _cache[tipo]["scaler"]

    rutas = ARCHIVOS[tipo]

    if not os.path.exists(rutas["modelo"]):
        raise FileNotFoundError(
            f"Modelo '{rutas['modelo']}' no encontrado. "
            f"Ejecuta primero "
            f"{'regresion_logistica.py' if tipo == 'rl' else 'red_neuronal.py'}"
        )

    scaler = joblib.load(rutas["scaler"])

    if tipo == "rl":
        modelo = joblib.load(rutas["modelo"])
    else:
        from tensorflow.keras.models import load_model as tf_load
        modelo = tf_load(rutas["modelo"])

    _cache[tipo] = {"modelo": modelo, "scaler": scaler}
    print(f"  ✓ Modelo '{tipo}' cargado en memoria")
    return modelo, scaler


def cargar_feature_names():
    """Lee los nombres de features guardados al entrenar."""
    if os.path.exists("feature_names.json"):
        with open("feature_names.json") as f:
            data = json.load(f)
        return data.get("features", []), data.get("mediana_cnt", None)
    return [], None


# ─────────────────────────────────────────────────────────────
#  RUTAS
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    print("--> Solicitud recibida en /")
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200


@app.route("/estado", methods=["GET"])
def estado():
    """Informa qué modelos están listos (archivos .pkl / .h5 existen)."""
    disponibles = {}
    for tipo, rutas in ARCHIVOS.items():
        modelo_ok = os.path.exists(rutas["modelo"])
        scaler_ok = os.path.exists(rutas["scaler"])
        disponibles[tipo] = {
            "modelo": modelo_ok,
            "scaler": scaler_ok,
            "listo":  modelo_ok and scaler_ok
        }
    # También devuelve los nombres de las features y la mediana
    features, mediana = cargar_feature_names()
    return jsonify({
        "modelos"    : disponibles,
        "features"   : features,
        "mediana_cnt": mediana
    })


@app.route("/predecir/individual", methods=["POST"])
def predecir_individual():
    """
    Recibe JSON:
      { "modelo": "rl" | "rna", "features": [v1, v2, ..., vN] }
    Devuelve:
      { "prediccion": 0|1, "prob_clase0": float,
        "prob_clase1": float, "etiqueta": str, "modelo_usado": str }
    """
    try:
        datos    = request.get_json()
        tipo     = datos.get("modelo")
        features = datos.get("features", [])

        if tipo not in ("rl", "rna"):
            return jsonify({"error": "Modelo inválido. Usa 'rl' o 'rna'"}), 400
        if not features:
            return jsonify({"error": "No se recibieron features"}), 400

        modelo, scaler = cargar_modelo(tipo)

        X    = np.array(features, dtype=float).reshape(1, -1)
        X_sc = scaler.transform(X)

        if tipo == "rl":
            pred  = int(modelo.predict(X_sc)[0])
            proba = modelo.predict_proba(X_sc)[0]
            prob0, prob1 = float(proba[0]), float(proba[1])
        else:
            prob1 = float(modelo.predict(X_sc, verbose=0)[0][0])
            prob0 = 1.0 - prob1
            pred  = 1 if prob1 >= 0.5 else 0

        return jsonify({
            "prediccion":   pred,
            "prob_clase0":  round(prob0, 4),
            "prob_clase1":  round(prob1, 4),
            "etiqueta":     "Alta Demanda" if pred == 1 else "Baja Demanda",
            "modelo_usado": "Regresión Logística" if tipo == "rl"
                            else "Red Neuronal Artificial"
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


@app.route("/predecir/lotes", methods=["POST"])
def predecir_lotes():
    """
    Recibe un CSV (multipart/form-data) con features + columna 'target'.
    Devuelve métricas de desempeño y la matriz de confusión.
    """
    try:
        tipo    = request.form.get("modelo")
        archivo = request.files.get("archivo")

        if tipo not in ("rl", "rna"):
            return jsonify({"error": "Modelo inválido"}), 400
        if not archivo:
            return jsonify({"error": "No se recibió archivo CSV"}), 400

        df = pd.read_csv(archivo)

        # Columna objetivo: busca "target", si no usa la última columna
        target_col = "target" if "target" in df.columns else df.columns[-1]
        y_real = df[target_col].values
        X      = df.drop(columns=[target_col]).values.astype(float)

        modelo, scaler = cargar_modelo(tipo)
        X_sc = scaler.transform(X)

        if tipo == "rl":
            y_pred  = modelo.predict(X_sc)
            y_proba = modelo.predict_proba(X_sc)[:, 1]
        else:
            y_proba = modelo.predict(X_sc, verbose=0).flatten()
            y_pred  = (y_proba >= 0.5).astype(int)

        acc  = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred, zero_division=0)
        rec  = recall_score(y_real, y_pred, zero_division=0)
        f1   = f1_score(y_real, y_pred, zero_division=0)
        auc  = roc_auc_score(y_real, y_proba)
        cm   = confusion_matrix(y_real, y_pred).tolist()

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        return jsonify({
            "accuracy":         round(acc,  4),
            "precision":        round(prec, 4),
            "recall":           round(rec,  4),
            "f1_score":         round(f1,   4),
            "auc_roc":          round(auc,  4),
            "n_muestras":       int(len(y_real)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                                 "fn": int(fn), "tp": int(tp)},
            "modelo_usado":     "Regresión Logística" if tipo == "rl"
                                else "Red Neuronal Artificial"
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
