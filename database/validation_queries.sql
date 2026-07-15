-- Workstream 3 database validation queries.
-- Run with:
--   sqlite3 database/pregnancy_risk.db < database/validation_queries.sql

.headers on
.mode column

PRAGMA foreign_keys = ON;

SELECT 'foreign_key_check' AS check_name, COUNT(*) AS issue_count
FROM pragma_foreign_key_check;

SELECT 'integrity_check' AS check_name, integrity_check AS result
FROM pragma_integrity_check;

SELECT
    ds.source_dataset,
    ds.loaded_row_count AS expected_rows,
    COUNT(p.pregnancy_id) AS pregnancy_rows,
    COUNT(p.pregnancy_id) - ds.loaded_row_count AS row_count_difference
FROM data_sources ds
LEFT JOIN pregnancies p
    ON p.source_dataset = ds.source_dataset
GROUP BY ds.source_dataset, ds.loaded_row_count
ORDER BY ds.source_dataset;

SELECT
    source_dataset,
    target_risk,
    COUNT(*) AS rows
FROM pregnancies
GROUP BY source_dataset, target_risk
ORDER BY source_dataset, target_risk;

SELECT
    'missing_target_values' AS check_name,
    COUNT(*) AS issue_count
FROM pregnancies
WHERE target_risk IS NULL;

SELECT
    'vitals_rows' AS table_name,
    COUNT(*) AS rows
FROM vitals
UNION ALL
SELECT 'medical_history_rows', COUNT(*) FROM medical_history
UNION ALL
SELECT 'lab_tests_rows', COUNT(*) FROM lab_tests
UNION ALL
SELECT 'obstetric_observations_rows', COUNT(*) FROM obstetric_observations;

SELECT
    'pregnancies_without_vitals' AS check_name,
    COUNT(*) AS issue_count
FROM pregnancies p
LEFT JOIN vitals v
    ON v.pregnancy_id = p.pregnancy_id
WHERE v.pregnancy_id IS NULL
UNION ALL
SELECT
    'pregnancies_without_medical_history',
    COUNT(*)
FROM pregnancies p
LEFT JOIN medical_history h
    ON h.pregnancy_id = p.pregnancy_id
WHERE h.pregnancy_id IS NULL
UNION ALL
SELECT
    'pregnancies_without_lab_tests',
    COUNT(*)
FROM pregnancies p
LEFT JOIN lab_tests l
    ON l.pregnancy_id = p.pregnancy_id
WHERE l.pregnancy_id IS NULL
UNION ALL
SELECT
    'pregnancies_without_obstetric_observations',
    COUNT(*)
FROM pregnancies p
LEFT JOIN obstetric_observations o
    ON o.pregnancy_id = p.pregnancy_id
WHERE o.pregnancy_id IS NULL;

SELECT
    'orphan_vitals' AS check_name,
    COUNT(*) AS issue_count
FROM vitals v
LEFT JOIN pregnancies p
    ON p.pregnancy_id = v.pregnancy_id
WHERE p.pregnancy_id IS NULL
UNION ALL
SELECT
    'orphan_medical_history',
    COUNT(*)
FROM medical_history h
LEFT JOIN pregnancies p
    ON p.pregnancy_id = h.pregnancy_id
WHERE p.pregnancy_id IS NULL
UNION ALL
SELECT
    'orphan_lab_tests',
    COUNT(*)
FROM lab_tests l
LEFT JOIN pregnancies p
    ON p.pregnancy_id = l.pregnancy_id
WHERE p.pregnancy_id IS NULL
UNION ALL
SELECT
    'orphan_obstetric_observations',
    COUNT(*)
FROM obstetric_observations o
LEFT JOIN pregnancies p
    ON p.pregnancy_id = o.pregnancy_id
WHERE p.pregnancy_id IS NULL;

SELECT
    'Data1 modelling_rows' AS output_name,
    COUNT(*) AS rows
FROM vw_data1_modeling
UNION ALL
SELECT
    'Data2 modelling_rows',
    COUNT(*)
FROM vw_data2_modeling;
