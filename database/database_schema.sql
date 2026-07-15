PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS data_sources (
    source_dataset TEXT PRIMARY KEY CHECK (source_dataset IN ('Data1', 'Data2')),
    cleaned_file TEXT NOT NULL,
    loaded_row_count INTEGER NOT NULL CHECK (loaded_row_count >= 0),
    loaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS pregnancies (
    pregnancy_id INTEGER PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_row_id INTEGER NOT NULL CHECK (source_row_id >= 0),
    target_risk INTEGER NOT NULL CHECK (target_risk IN (0, 1)),
    age REAL CHECK (age IS NULL OR age BETWEEN 10 AND 65),
    pregnancy_weeks REAL CHECK (pregnancy_weeks IS NULL OR pregnancy_weeks BETWEEN 1 AND 45),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_dataset, source_row_id),
    FOREIGN KEY (source_dataset) REFERENCES data_sources(source_dataset)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS vitals (
    vitals_id INTEGER PRIMARY KEY,
    pregnancy_id INTEGER NOT NULL UNIQUE,
    systolic_bp REAL CHECK (systolic_bp IS NULL OR systolic_bp BETWEEN 70 AND 220),
    diastolic_bp REAL CHECK (diastolic_bp IS NULL OR diastolic_bp BETWEEN 40 AND 140),
    heart_rate REAL CHECK (heart_rate IS NULL OR heart_rate BETWEEN 40 AND 200),
    body_temp_f REAL CHECK (body_temp_f IS NULL OR body_temp_f BETWEEN 90 AND 110),
    weight_kg REAL CHECK (weight_kg IS NULL OR weight_kg BETWEEN 25 AND 200),
    height_cm REAL CHECK (height_cm IS NULL OR height_cm BETWEEN 120 AND 220),
    height_m REAL CHECK (height_m IS NULL OR height_m BETWEEN 1.2 AND 2.2),
    bmi REAL CHECK (bmi IS NULL OR bmi BETWEEN 10 AND 70),
    FOREIGN KEY (pregnancy_id) REFERENCES pregnancies(pregnancy_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS medical_history (
    history_id INTEGER PRIMARY KEY,
    pregnancy_id INTEGER NOT NULL UNIQUE,
    previous_complications INTEGER CHECK (previous_complications IS NULL OR previous_complications IN (0, 1)),
    preexisting_diabetes INTEGER CHECK (preexisting_diabetes IS NULL OR preexisting_diabetes IN (0, 1)),
    gestational_diabetes INTEGER CHECK (gestational_diabetes IS NULL OR gestational_diabetes IN (0, 1)),
    mental_health INTEGER CHECK (mental_health IS NULL OR mental_health IN (0, 1)),
    gravida_n INTEGER CHECK (gravida_n IS NULL OR gravida_n IN (1, 2, 3)),
    tt_injection_n INTEGER CHECK (tt_injection_n IS NULL OR tt_injection_n IN (1, 2, 3)),
    FOREIGN KEY (pregnancy_id) REFERENCES pregnancies(pregnancy_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lab_tests (
    lab_id INTEGER PRIMARY KEY,
    pregnancy_id INTEGER NOT NULL UNIQUE,
    blood_sugar REAL CHECK (blood_sugar IS NULL OR blood_sugar BETWEEN 2 AND 25),
    urine_sugar_yes INTEGER CHECK (urine_sugar_yes IS NULL OR urine_sugar_yes IN (0, 1)),
    vdrl_positive INTEGER CHECK (vdrl_positive IS NULL OR vdrl_positive IN (0, 1)),
    hrsag_positive INTEGER CHECK (hrsag_positive IS NULL OR hrsag_positive IN (0, 1)),
    FOREIGN KEY (pregnancy_id) REFERENCES pregnancies(pregnancy_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS obstetric_observations (
    obs_id INTEGER PRIMARY KEY,
    pregnancy_id INTEGER NOT NULL UNIQUE,
    fetal_heartbeat_bpm REAL CHECK (fetal_heartbeat_bpm IS NULL OR fetal_heartbeat_bpm BETWEEN 60 AND 220),
    FOREIGN KEY (pregnancy_id) REFERENCES pregnancies(pregnancy_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL CHECK (dataset_name IN ('Data1', 'Data2', 'Combined')),
    trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    auc REAL CHECK (auc IS NULL OR auc BETWEEN 0 AND 1),
    accuracy REAL CHECK (accuracy IS NULL OR accuracy BETWEEN 0 AND 1),
    precision REAL CHECK (precision IS NULL OR precision BETWEEN 0 AND 1),
    recall REAL CHECK (recall IS NULL OR recall BETWEEN 0 AND 1),
    f1 REAL CHECK (f1 IS NULL OR f1 BETWEEN 0 AND 1),
    threshold REAL CHECK (threshold IS NULL OR threshold BETWEEN 0 AND 1),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_run_id INTEGER,
    model_name TEXT NOT NULL,
    input_json TEXT NOT NULL,
    predicted_probability REAL NOT NULL CHECK (predicted_probability BETWEEN 0 AND 1),
    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    risk_label TEXT NOT NULL CHECK (risk_label IN ('high-risk pregnancy', 'not high-risk pregnancy')),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_pregnancies_source_dataset
    ON pregnancies(source_dataset);

CREATE INDEX IF NOT EXISTS idx_pregnancies_target_risk
    ON pregnancies(source_dataset, target_risk);

CREATE INDEX IF NOT EXISTS idx_model_runs_dataset
    ON model_runs(dataset_name);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions(created_at);

CREATE VIEW IF NOT EXISTS vw_data1_modeling AS
SELECT
    p.pregnancy_id,
    p.source_row_id,
    p.age,
    v.systolic_bp,
    v.diastolic_bp,
    l.blood_sugar,
    v.body_temp_f,
    v.bmi,
    h.previous_complications,
    h.preexisting_diabetes,
    h.gestational_diabetes,
    h.mental_health,
    v.heart_rate,
    p.target_risk AS risk
FROM pregnancies p
JOIN vitals v
    ON v.pregnancy_id = p.pregnancy_id
JOIN medical_history h
    ON h.pregnancy_id = p.pregnancy_id
JOIN lab_tests l
    ON l.pregnancy_id = p.pregnancy_id
WHERE p.source_dataset = 'Data1';

CREATE VIEW IF NOT EXISTS vw_data2_modeling AS
SELECT
    p.pregnancy_id,
    p.source_row_id,
    p.age,
    h.gravida_n,
    h.tt_injection_n,
    p.pregnancy_weeks,
    v.weight_kg,
    v.height_cm,
    v.height_m,
    v.systolic_bp,
    v.diastolic_bp,
    o.fetal_heartbeat_bpm,
    l.urine_sugar_yes,
    l.vdrl_positive,
    l.hrsag_positive,
    v.bmi AS bmi_calc,
    p.target_risk AS high_risk
FROM pregnancies p
JOIN vitals v
    ON v.pregnancy_id = p.pregnancy_id
JOIN medical_history h
    ON h.pregnancy_id = p.pregnancy_id
JOIN lab_tests l
    ON l.pregnancy_id = p.pregnancy_id
JOIN obstetric_observations o
    ON o.pregnancy_id = p.pregnancy_id
WHERE p.source_dataset = 'Data2';
