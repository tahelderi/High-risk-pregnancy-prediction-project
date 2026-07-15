"use strict";

const DIRECT_FIELDS = [
  "age", "weight_kg", "height_cm",
  "systolic_bp", "diastolic_bp", "heart_rate", "blood_sugar",
  "previous_complications", "preexisting_diabetes", "gestational_diabetes", "mental_health",
  "gravida_n", "tt_injection_n", "pregnancy_weeks", "fetal_heartbeat_bpm",
  "urine_sugar_yes", "vdrl_positive", "hrsag_positive",
];

const form = document.getElementById("risk-form");
const errorBox = document.getElementById("form-error");
const resultEl = document.getElementById("result");

function updateBmi() {
  const w = parseFloat(document.getElementById("weight_kg").value);
  const h = parseFloat(document.getElementById("height_cm").value);
  const out = document.getElementById("bmi_display");
  out.value = (w > 0 && h > 0) ? (w / ((h / 100) ** 2)).toFixed(1) : "";
}
document.getElementById("weight_kg").addEventListener("input", updateBmi);
document.getElementById("height_cm").addEventListener("input", updateBmi);

function collectPayload() {
  const payload = {};
  for (const name of DIRECT_FIELDS) {
    const el = form.elements[name];
    const v = el ? (el.value || "").trim() : "";
    if (v !== "") payload[name] = v;
  }
  // body temperature is entered in Celsius and sent in Fahrenheit
  const c = (form.elements["body_temp_c"].value || "").trim();
  if (c !== "") {
    const f = parseFloat(c) * 9 / 5 + 32;
    if (!Number.isNaN(f)) payload["body_temp_f"] = f.toFixed(1);
  }
  return payload;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function miniStat(label, pct) {
  return `<div class="mini-stat">${label}: <b>${pct}%</b></div>`;
}

function renderResult(data) {
  const elevated = data.main.predicted_class === 1;
  const pct = data.main.probability_pct;

  resultEl.classList.toggle("is-elevated", elevated);
  resultEl.classList.toggle("is-clear", !elevated);

  document.getElementById("result-badge").textContent =
    elevated ? "סיכון מוגבר" : "ללא סיכון מוגבר";
  document.getElementById("result-mode").textContent = data.mode_label;
  document.getElementById("result-title").textContent = data.main.risk_label;

  const fill = document.getElementById("result-meter-fill");
  fill.style.width = "0%";
  requestAnimationFrame(() => { fill.style.width = pct + "%"; });

  document.getElementById("result-percent").textContent =
    `לפי הנתונים שהוזנו, מידת הדמיון למקרים בסיכון מוגבר היא כ-${pct}%.`;

  document.getElementById("result-explain").textContent = elevated
    ? "ההערכה מצביעה על דמיון גבוה יחסית למקרים שסווגו כהריון בסיכון מוגבר. אין משמעות הדבר שקיים מצב רפואי כלשהו — זוהי הערכה סטטיסטית בלבד, שנועדה להפנות אותך לבירור נוסף."
    : "ההערכה אינה מצביעה על דמיון גבוה למקרים בסיכון מוגבר, על פי הפרטים שהוזנו. זוהי הערכה סטטיסטית בלבד, והיא אינה ערובה ואינה מחליפה מעקב רפואי.";

  document.getElementById("result-next").textContent = elevated
    ? "מומלץ לפנות לצוות רפואי להמשך בדיקה והתייעצות."
    : "הכלי אינו מחליף מעקב רפואי שגרתי. המשיכי במעקב ההריון הרגיל שלך.";

  const more = [];
  if (data.mode === "combined") {
    more.push(miniStat("הערכה משולבת", pct));
    if (data.details.data1) more.push(miniStat("לפי מדדי בריאות (Data1)", data.details.data1.probability_pct));
    if (data.details.data2) more.push(miniStat("לפי בדיקות הריון (Data2)", data.details.data2.probability_pct));
  } else if (data.mode === "data1_only" && data.details.data1) {
    more.push(miniStat("לפי מדדי בריאות (Data1)", data.details.data1.probability_pct));
  } else if (data.mode === "data2_only" && data.details.data2) {
    more.push(miniStat("לפי בדיקות הריון (Data2)", data.details.data2.probability_pct));
  }
  const moreEl = document.getElementById("result-more");
  if (more.length > 1) {
    document.getElementById("result-more-body").innerHTML =
      `<div class="result-more-grid">${more.join("")}</div>`;
    moreEl.hidden = false;
  } else {
    moreEl.hidden = true;
  }

  document.getElementById("result-disclaimer").textContent = data.disclaimer;

  resultEl.hidden = false;
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  resultEl.focus({ preventScroll: true });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;

  const payload = collectPayload();
  if (Object.keys(payload).length === 0) {
    showError("נא למלא לפחות חלק מהפרטים כדי לקבל הערכה.");
    return;
  }

  const submitBtn = form.querySelector('button[type="submit"]');
  const prevText = submitBtn.textContent;
  submitBtn.disabled = true;
  submitBtn.textContent = "מחשבת…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      resultEl.hidden = true;
      showError(data.error || "אירעה שגיאה. נא לנסות שוב.");
      return;
    }
    renderResult(data);
  } catch (err) {
    showError("לא ניתן היה להתחבר לשרת. ודאי שהאתר פועל ונסי שוב.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = prevText;
  }
});

form.addEventListener("reset", () => {
  errorBox.hidden = true;
  resultEl.hidden = true;
  document.getElementById("bmi_display").value = "";
});
