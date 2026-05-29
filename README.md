# 🥑 Avocado Price Elasticity & TPRC via Double Machine Learning

Causal estimation of **own-price elasticity (PE)** and **promotional demand multiplier (TPRC)** across 54 U.S. markets using Double Machine Learning — with a full pipeline debug that lifted nuisance model R² from **0.003 → 0.282** (94×).

> 📝 **Debugging write-up:** [How I lifted R² 94× by removing one preprocessing step →](https://medium.com/@a.takeuchi121/debugging-a-broken-causal-pipeline-how-a-frisch-waugh-lovell-insight-lifted-r%C2%B2-from-0-003-to-0-282-88a1559341bf)

---

## What This Is
Retailers and marketplace platforms lose revenue daily by pricing from intuition rather than causal evidence — because standard regression conflates correlation with causation (endogeneity). This project applies Double Machine Learning (DML) to the Kaggle Avocado Prices dataset (90,000+ weekly records, 54 U.S. markets) to recover causally valid own-price elasticity (PE) and promotional demand multiplier (TPRC) estimates for three avocado SKUs across Online and Offline channels.
 
Standard regression can't estimate price elasticity causally — price and demand are simultaneously determined (endogeneity). This project implements **Double Machine Learning (DML)** to recover a causal PE estimate:

1. **Stage 1 — Residualise:** Learn `E[lnP | X]` and `E[lnQ | X]` via K-fold cross-fitting. Subtract to get "unexpected" price and demand shocks.
2. **Stage 2 — Identify:** Regress demand residuals on price residuals. The slope is the causal PE, free of confounding.

**Why this matters at scale:** Price elasticity and promo lift estimation underpin pricing strategy at marketplace platforms like Instacart, DoorDash, Amazon Fresh, and similar platforms — where a 1% error in PE translates directly to revenue loss.

---

## Results

| Item | Channel | PE | 95% CI | TPRC | Flag |
|---|---|---|---|---|---|
| PLU_4046 | Offline | −1.199 | [−1.314, −1.084] | 0.915 | 🟢 Green |
| PLU_4046 | Online | −0.369 | [−0.512, −0.227] | 0.893 | 🟢 Green |
| PLU_4225 | Offline | −0.635 | [−0.706, −0.563] | 1.026 | 🟢 Green |
| PLU_4225 | Online | −0.128 | [−0.235, −0.021] | 1.098 | 🟢 Green |
| PLU_4770 | Offline | −1.096 | [−1.300, −0.892] | 1.027 | 🟢 Green |
| PLU_4770 | Online | −0.011 | [−0.220, +0.198] | 0.773 | 🟢 Green |

**Nuisance diagnostics:** `r2_price = 0.282` · `r2_qty = 0.696` · All 6 cells: Green

Each estimate carries HC3-robust standard errors and a traffic-light reliability flag gating on: sufficient weeks, price variation, discount observations, and nuisance R².

---

## Pipeline Architecture

```
Raw CSV → clean() → build_panel() → build_feature_matrix()
       → cross_fit_residuals()   → build_discounts()
       → estimate_pe_tprc()      → reliability_flag()
       → Results + Revenue Simulation
```

**Feature matrix X** (controls passed to both nuisance models):

| Group | Detail |
|---|---|
| Trend | Linear week index |
| Fourier seasonality | 3 harmonics × 52-week + 2 harmonics × 12-month |
| Channel dummy | Online / Offline |
| Geographic FE | One-hot city dummies (54 outlets) or target encoding |
| Year dummies | One-hot |

---

## The R² Debug: 0.003 → 0.282

Five root causes identified from EDA, each generating a targeted fix:

| Fix | EDA Finding | Solution | B-V Effect |
|---|---|---|---|
| **1** | Houston $1.00 vs SF $1.75 — city price gap is pure noise without identity | Geographic entity fixed effects (`onehot` / `target`) | ↓ Bias |
| **2** | Asymmetric harvest peaks can't be captured by a single sine wave | Multi-harmonic Fourier (`n=3`) | ↓ Bias |
| **3** | Online/Offline have fundamentally different price-setting processes | Channel-specific price nuisance model per fold | ↓ Bias |
| **4** | Outlier price spikes dominate squared-error loss | `HuberRegressor` option for price first stage | ↓ Variance |
| **5 ★** | Pre-demeaning by outlet + outlet dummies in X = double-removal of all geographic signal | `demean_mode='none'` — raw lnP as treatment; FWL theorem guarantees equivalence | ↓ Bias (decisive) |

Fix 5 was the decisive insight. The original pipeline computed `lnP_dm = lnP − mean(lnP per outlet)`, stripping all between-city variation before the nuisance model ran — then re-added outlet dummies to X. The **Frisch-Waugh-Lovell theorem** shows these are algebraically equivalent; doing both left only ~3 distinct price levels per outlet across 143 weeks, a near-flat signal no model can predict. Setting `demean_mode='none'` restored full signal to the nuisance model. → [Full write-up](https://medium.com/@a.takeuchi121/debugging-a-broken-causal-pipeline-how-a-frisch-waugh-lovell-insight-lifted-r%C2%B2-from-0-003-to-0-282-88a1559341bf)

---

## Key Design Choices

- **Time-ordered cross-fitting (`shuffle=False`)** — prevents future price leaking into nuisance training; produces uncontaminated out-of-fold residuals (Neyman orthogonality).

- **HC3 robust standard errors** — leverage-adjusted sandwich estimator for valid inference across 54 heterogeneous markets.

- **Configurable model backends** — `make_price_model()` / `make_qty_model()` factories allow independent model selection (`gbm | huber | ridge | rf`) without touching pipeline logic.

---

## Honest Limitations

- **`r2_price = 0.282` is marginal.** PE magnitudes are likely attenuated (weak-instrument analogy); treat as lower bounds on true elasticity.
- Wholesale costs, competitor pricing, and retailer promo calendars are unobservable in this dataset — a structural ceiling, not a modeling gap.
- Cross-outlet lagged price features are the most tractable path to push R² toward 0.40+.

---

## Quickstart

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

1. Download [Kaggle Avocado Prices](https://www.kaggle.com/datasets/neuromusic/avocado-prices), set `CONFIG['data_path']` in Cell 3
2. Run all cells in `avocado_pe_tprc_r2fix.ipynb`
3. Outputs → `outputs/results_pe_tprc.csv` · `outputs/nuisance_diagnostics.csv`

**Key config toggles:**

```python
CONFIG['demean_mode']             = 'none'    # 'none' | 'time' | 'outlet'
CONFIG['geo_encoding']            = 'onehot'  # 'onehot' | 'target'
CONFIG['fourier_harmonics']       = 3
CONFIG['channel_specific_models'] = True
CONFIG['price_model']             = 'gbm'     # 'gbm' | 'huber' | 'ridge' | 'rf'
```

---

## Stack

`Python 3.10+` · `scikit-learn` · `NumPy` · `pandas` · `SciPy` · `Matplotlib` · `Seaborn`

*Dataset: [Kaggle Avocado Prices 2015–2023](https://www.kaggle.com/datasets/neuromusic/avocado-prices) · 54 U.S. markets · 3 PLU codes · Weekly frequency*
