const API_BASE_URL = window.API_BASE_URL;

const SELECT_OPTIONS = {
  job: ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown"],
  marital: ["divorced", "married", "single", "unknown"],
  education: ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
              "professional.course", "university.degree", "unknown"],
  default: ["no", "yes", "unknown"],
  housing: ["no", "yes", "unknown"],
  loan: ["no", "yes", "unknown"],
  contact: ["cellular", "telephone"],
  month: ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
  day_of_week: ["mon", "tue", "wed", "thu", "fri"],
  poutcome: ["failure", "nonexistent", "success"],
};

const DEFAULTS = {
  job: "technician",
  marital: "married",
  education: "university.degree",
  default: "no",
  housing: "yes",
  loan: "no",
  contact: "cellular",
  month: "may",
  day_of_week: "thu",
  poutcome: "nonexistent",
};

function populateSelects() {
  for (const [name, options] of Object.entries(SELECT_OPTIONS)) {
    const el = document.querySelector(`select[name="${name}"]`);
    el.innerHTML = options
      .map((opt) => `<option value="${opt}" ${opt === DEFAULTS[name] ? "selected" : ""}>${opt}</option>`)
      .join("");
  }
}

async function loadInfo() {
  const infoEl = document.getElementById("info-content");
  const apiLink = document.getElementById("api-link");
  apiLink.href = API_BASE_URL;
  apiLink.textContent = API_BASE_URL;

  try {
    const res = await fetch(`${API_BASE_URL}/info`);
    if (!res.ok) {
      infoEl.textContent = `API responded ${res.status} (artifact may be unavailable).`;
      return;
    }
    const data = await res.json();
    const meta = data.metadata;
    infoEl.innerHTML = `
      <p>${meta.description}</p>
      <ul>
        <li><strong>Model:</strong> ${meta.model_type} (custom transformer: ${meta.custom_transformer})</li>
        <li><strong>scikit-learn:</strong> ${meta.sklearn_version}</li>
        <li><strong>Trained on:</strong> ${meta.n_train_rows} rows, built ${new Date(meta.built_at).toLocaleString()}</li>
        <li><strong>Held-out ROC AUC:</strong> ${meta.test_roc_auc.toFixed(4)}</li>
        <li><strong>Baseline training subscribe rate:</strong> ${(data.target_rate * 100).toFixed(2)}%</li>
      </ul>
    `;
  } catch (err) {
    infoEl.textContent = `Could not reach API: ${err}`;
  }
}

function collectPayload(form) {
  const fd = new FormData(form);
  const payload = {};
  for (const [key, value] of fd.entries()) {
    if (["age", "campaign", "pdays", "previous"].includes(key)) {
      payload[key] = parseInt(value, 10);
    } else if (["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"].includes(key)) {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  }
  return payload;
}

function renderResult(data) {
  const resultEl = document.getElementById("result");
  const errorEl = document.getElementById("error");
  errorEl.hidden = true;

  const pct = (data.subscribe_probability * 100).toFixed(2);
  resultEl.innerHTML = `
    <div class="prob">${pct}% predicted subscribe probability</div>
    <div>Predicted label: <strong>${data.predicted_label}</strong></div>
    <div class="muted">Baseline training rate: ${(data.baseline_training_rate * 100).toFixed(2)}%</div>
  `;
  resultEl.hidden = false;
}

function renderError(message) {
  const resultEl = document.getElementById("result");
  const errorEl = document.getElementById("error");
  resultEl.hidden = true;
  errorEl.textContent = message;
  errorEl.hidden = false;
}

async function handleSubmit(evt) {
  evt.preventDefault();
  const form = evt.target;
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Predicting…";

  try {
    const payload = collectPayload(form);
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 422) {
      const body = await res.json();
      renderError(`Validation error (422): ${JSON.stringify(body.detail)}`);
    } else if (res.status === 503) {
      const body = await res.json();
      renderError(`Model unavailable (503): ${body.detail}`);
    } else if (!res.ok) {
      renderError(`Unexpected error (${res.status})`);
    } else {
      const data = await res.json();
      renderResult(data);
    }
  } catch (err) {
    renderError(`Network error calling API: ${err}`);
  } finally {
    button.disabled = false;
    button.textContent = "Predict subscription probability";
  }
}

populateSelects();
loadInfo();
document.getElementById("predict-form").addEventListener("submit", handleSubmit);
