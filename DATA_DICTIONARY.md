# Data Dictionary

## Workbook

`koh_2eorr_cleaned_variant_for_modeling.xlsx`

## Sheets

### 1. `README`
Workbook-level notes describing the purpose of the processed dataset variant.

### 2. `all_rows_with_flags`
Full processed table, including rows later retained or removed for the cleaned modeling variant.

### 3. `cleaned_modeling_data`
Main sheet used by the Python scripts.

### 4. `removed_rows`
Rows excluded from the cleaned modeling variant.

### 5. `literature_metadata`
Supporting metadata and source attribution for literature-derived samples.

## Main modeling columns used by the code

### Predictors
- `ID/IG` — Raman ID/IG ratio
- `Time` — treatment / carbonization time (h)
- `Temperature , C` — treatment / carbonization temperature (°C)
- `BET, m2/g` — BET specific surface area (m²/g)
- `Dataset source` — source label used as a categorical feature

### Target
- `H2O2 (%)` — H2O2 selectivity / percentage used as the regression target

## Additional metadata columns in the workbook

- `Original row id` — row identifier retained for traceability
- `Sample / Paper` — sample or literature label
- `Protected experimental row` — Boolean flag for the first nine original experimental rows
- `Keep for cleaned variant` — Boolean flag showing whether the row is retained
- `Removal reason` — text note for excluded rows

## Current source labeling convention in the processed workbook

- Rows `1–9`: `Original experiment`
- Rows after `9`: `Literature`

## Notes

- The code reads `cleaned_modeling_data` by default when the workbook is present.
- If the workbook is replaced by a CSV file, the scripts expect the same core column names.
