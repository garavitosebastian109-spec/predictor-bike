// ============================================================
//  script.js — Bike Sharing ML Predictor
//  Ubicación: bike_predictor/static/script.js
//  Se comunica con app.py (Flask) en http://127.0.0.1:5000
// ============================================================

const API = "";   // Vacío = misma dirección que sirve Flask

let modeloActual = null;
let archivoLotes = null;
let featuresNames = [];

// Campos del formulario individual con etiquetas descriptivas y rangos
// Se populan dinámicamente desde /estado, pero aquí tenemos los labels
// y los valores mínimos/máximos esperados del dataset Bike Sharing
const FIELD_CONFIG = {
  season:     { label: "Temporada (1-4)",       min: 1,   max: 4,    step: 1,    placeholder: "1=Primav, 2=Verano, 3=Otoño, 4=Invierno" },
  yr:         { label: "Año (0=2011, 1=2012)",  min: 0,   max: 1,    step: 1,    placeholder: "0 ó 1" },
  mnth:       { label: "Mes (1-12)",             min: 1,   max: 12,   step: 1,    placeholder: "1=Enero … 12=Diciembre" },
  holiday:    { label: "Festivo (0/1)",          min: 0,   max: 1,    step: 1,    placeholder: "0=No, 1=Sí" },
  weekday:    { label: "Día de semana (0-6)",    min: 0,   max: 6,    step: 1,    placeholder: "0=Dom … 6=Sáb" },
  workingday: { label: "Día laboral (0/1)",      min: 0,   max: 1,    step: 1,    placeholder: "0=No, 1=Sí" },
  weathersit: { label: "Clima (1-4)",            min: 1,   max: 4,    step: 1,    placeholder: "1=Despejado … 4=Tormenta" },
  temp:       { label: "Temp. norm. (0-1)",      min: 0,   max: 1,    step: 0.01, placeholder: "ej. 0.34" },
  atemp:      { label: "Sens. térmica (0-1)",    min: 0,   max: 1,    step: 0.01, placeholder: "ej. 0.36" },
  hum:        { label: "Humedad norm. (0-1)",    min: 0,   max: 1,    step: 0.01, placeholder: "ej. 0.80" },
  windspeed:  { label: "Viento norm. (0-1)",     min: 0,   max: 1,    step: 0.01, placeholder: "ej. 0.16" },
};

// ════════════════════════════════════════════════════════════
//  INICIALIZACIÓN
// ════════════════════════════════════════════════════════════
window.addEventListener("DOMContentLoaded", () => {
  verificarEstado();
});


// ════════════════════════════════════════════════════════════
//  VERIFICAR ESTADO DE MODELOS Y CARGAR FEATURES
// ════════════════════════════════════════════════════════════
async function verificarEstado() {
  const badge = document.getElementById("estado-badge");
  try {
    const res  = await fetch(`${API}/estado`);
    const data = await res.json();
    const rl   = data.modelos.rl.listo;
    const rna  = data.modelos.rna.listo;

    if (rl && rna) {
      badge.textContent = "● Ambos modelos listos";
      badge.classList.add("listo");
    } else if (rl || rna) {
      badge.textContent = rl
        ? "● RL listo · RNA pendiente"
        : "● RNA lista · RL pendiente";
      badge.classList.add("listo");
    } else {
      badge.textContent = "● Modelos no entrenados aún";
      badge.classList.add("error");
    }

    // Usar los nombres de features del servidor (feature_names.json)
    if (data.features && data.features.length > 0) {
      generarCampos(data.features);
    } else {
      // Fallback: usar FIELD_CONFIG directamente
      generarCampos(Object.keys(FIELD_CONFIG));
    }

  } catch {
    badge.textContent = "● Servidor desconectado";
    badge.classList.add("error");
    // Generar campos con los defaults si el servidor no responde
    generarCampos(Object.keys(FIELD_CONFIG));
  }
}


// ════════════════════════════════════════════════════════════
//  SELECCIÓN DE MODELO
// ════════════════════════════════════════════════════════════
function seleccionarModelo(tipo) {
  modeloActual = tipo;

  document.getElementById("card-rl").classList.remove("activo-rl", "activo-rna");
  document.getElementById("card-rna").classList.remove("activo-rl", "activo-rna");
  document.getElementById("card-" + tipo).classList.add(
    tipo === "rl" ? "activo-rl" : "activo-rna"
  );

  const btn = document.getElementById("btn-predecir-individual");
  btn.className = "btn btn-" + tipo;

  esconder("alerta-modelo");
  esconder("alerta-no-entrenado");
}


// ════════════════════════════════════════════════════════════
//  CAMBIO DE TAB
// ════════════════════════════════════════════════════════════
function cambiarTab(tab) {
  document.getElementById("tab-individual").classList.toggle("activo", tab === "individual");
  document.getElementById("tab-lotes").classList.toggle("activo", tab === "lotes");

  if (tab === "individual") {
    document.getElementById("panel-individual").classList.remove("oculto");
    document.getElementById("panel-lotes").classList.add("oculto");
  } else {
    document.getElementById("panel-lotes").classList.remove("oculto");
    document.getElementById("panel-individual").classList.add("oculto");
  }
}


// ════════════════════════════════════════════════════════════
//  GENERAR CAMPOS DEL FORMULARIO (adaptados al dataset)
// ════════════════════════════════════════════════════════════
function generarCampos(nombres) {
  const contenedor = document.getElementById("form-individual");
  contenedor.innerHTML = "";
  featuresNames = [];

  nombres.forEach((nombre, i) => {
    featuresNames.push(nombre);
    const cfg = FIELD_CONFIG[nombre] || {};

    const grupo = document.createElement("div");
    grupo.className = "form-group";
    grupo.innerHTML = `
      <label for="feat-${i}" title="${cfg.placeholder || ''}">${cfg.label || nombre}</label>
      <input
        type="number"
        id="feat-${i}"
        placeholder="${cfg.placeholder || 'ej. 0.00'}"
        step="${cfg.step || 'any'}"
        min="${cfg.min !== undefined ? cfg.min : ''}"
        max="${cfg.max !== undefined ? cfg.max : ''}"
      />
    `;
    contenedor.appendChild(grupo);
  });
}


// ════════════════════════════════════════════════════════════
//  PREDICCIÓN INDIVIDUAL
// ════════════════════════════════════════════════════════════
async function predecirIndividual() {
  if (!validarModelo()) return;

  const features = [];
  for (let i = 0; i < featuresNames.length; i++) {
    const val = parseFloat(document.getElementById(`feat-${i}`).value);
    features.push(isNaN(val) ? 0 : val);
  }

  const btn = document.getElementById("btn-predecir-individual");
  btn.disabled = true;
  mostrar("spinner-individual");
  esconder("resultado-individual");
  esconder("alerta-no-entrenado");

  try {
    const res = await fetch(`${API}/predecir/individual`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ modelo: modeloActual, features })
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      mostrarError("alerta-no-entrenado", "❌ " + (data.error || "Error al predecir."));
      return;
    }

    mostrarResultadoIndividual(data);

  } catch {
    mostrarError("alerta-no-entrenado",
      "❌ No se pudo conectar con el servidor. ¿Está corriendo app.py?");
  } finally {
    btn.disabled = false;
    esconder("spinner-individual");
  }
}

function mostrarResultadoIndividual(data) {
  const esPos  = data.prediccion === 1;
  const prob1  = (data.prob_clase1 * 100).toFixed(1);
  const prob0  = (data.prob_clase0 * 100).toFixed(1);

  const claseEl = document.getElementById("res-clase");
  // Etiquetas específicas para Bike Sharing
  claseEl.textContent = esPos ? "🚲 Alta Demanda" : "📉 Baja Demanda";
  claseEl.className   = "resultado-clase " + (esPos ? "clase-positivo" : "clase-negativo");

  document.getElementById("res-detalle").innerHTML =
    `Modelo: ${data.modelo_usado}<br>` +
    `Probabilidad Alta Demanda: ${prob1}%<br>` +
    `Probabilidad Baja Demanda: ${prob0}%<br>` +
    `Umbral de decisión: 0.50`;

  const barra = document.getElementById("barra-fill");
  barra.style.width      = prob1 + "%";
  barra.style.background = esPos ? "var(--accent-rl)" : "var(--error)";

  document.getElementById("barra-label").textContent =
    `${prob1}% probabilidad de alta demanda de bicicletas`;

  mostrar("resultado-individual");
}


// ════════════════════════════════════════════════════════════
//  MANEJO DE ARCHIVO PARA LOTES
// ════════════════════════════════════════════════════════════
function manejarArchivo(event) {
  const file = event.target.files[0];
  if (!file) return;
  archivoLotes = file;

  const zona = document.getElementById("upload-zona");
  zona.classList.add("cargado");
  document.getElementById("upload-titulo").textContent = "✓ " + file.name;
  document.getElementById("upload-sub").textContent =
    `${(file.size / 1024).toFixed(1)} KB · Listo para evaluar`;
}


// ════════════════════════════════════════════════════════════
//  PREDICCIÓN POR LOTES
// ════════════════════════════════════════════════════════════
async function evaluarLotes() {
  if (!validarModelo()) return;

  if (!archivoLotes) {
    alert("Primero carga un archivo CSV.");
    return;
  }

  const btn = document.getElementById("btn-evaluar");
  btn.disabled = true;
  mostrar("spinner-lotes");
  esconder("lotes-resultado");
  esconder("alerta-no-entrenado");

  const formData = new FormData();
  formData.append("modelo",  modeloActual);
  formData.append("archivo", archivoLotes);

  try {
    const res  = await fetch(`${API}/predecir/lotes`, {
      method: "POST",
      body:   formData
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      mostrarError("alerta-no-entrenado",
        "❌ " + (data.error || "Error al evaluar. Revisa el formato del CSV."));
      return;
    }

    mostrarResultadosLotes(data);

  } catch {
    mostrarError("alerta-no-entrenado",
      "❌ No se pudo conectar con el servidor. ¿Está corriendo app.py?");
  } finally {
    btn.disabled = false;
    esconder("spinner-lotes");
  }
}

function mostrarResultadosLotes(data) {
  document.getElementById("m-accuracy").textContent  = pct(data.accuracy);
  document.getElementById("m-precision").textContent = pct(data.precision);
  document.getElementById("m-recall").textContent    = pct(data.recall);
  document.getElementById("m-f1").textContent        = pct(data.f1_score);
  document.getElementById("m-auc").textContent       = data.auc_roc.toFixed(3);
  document.getElementById("m-muestras").textContent  = data.n_muestras;

  const cm = data.confusion_matrix;
  document.getElementById("cm-tn").textContent = cm.tn;
  document.getElementById("cm-fp").textContent = cm.fp;
  document.getElementById("cm-fn").textContent = cm.fn;
  document.getElementById("cm-tp").textContent = cm.tp;

  document.getElementById("modelo-usado-label").textContent =
    "→ Evaluado con: " + data.modelo_usado;

  mostrar("lotes-resultado");
}


// ════════════════════════════════════════════════════════════
//  UTILIDADES
// ════════════════════════════════════════════════════════════
function validarModelo() {
  if (!modeloActual) {
    mostrar("alerta-modelo");
    return false;
  }
  esconder("alerta-modelo");
  return true;
}

function pct(val)          { return (val * 100).toFixed(1) + "%"; }
function mostrar(id)       { document.getElementById(id)?.classList.remove("oculto"); }
function esconder(id)      { document.getElementById(id)?.classList.add("oculto"); }
function mostrarError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.remove("oculto"); }
}
