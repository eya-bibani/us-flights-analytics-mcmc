# ✈️ Stochastic Modelling & Operational Analytics: From Metropolis-Hastings to US Aviation Dynamics (2003–2007)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

Coursework project for **ST2195 – Programming for Data Science** (BSc Data Science & Business Analytics, University of London). It has two parts:

1. **Random Walk Metropolis-Hastings**: sampling from a Laplace distribution and diagnosing MCMC convergence with the Gelman-Rubin statistic.
2. **US flight analytics**: analysis of **~35 million flight records (2003–2007)** to find the best departure windows, test whether aircraft age affects delays, and model flight diversions with logistic regression.

Everything is implemented in **Python** and replicated in **R**.

---

## 📌 Key Findings

| Question | Finding |
|---|---|
| **Best time to fly** | Early departures (roughly **05:00–09:00**) have the lowest arrival delays. Delays build up after ~15:00 (knock-on effect). Saturdays and Sundays are more punctual than Mondays and Fridays. |
| **Does aircraft age matter?** | **No systematic effect.** Average delay shows no upward trend with age. Differences between years (2007 much higher than 2003) are far larger than differences between ages, which points to network-level factors such as congestion. |
| **What drives diversions?** | **Arrival hour** and **route distance** are the strongest risk-raising predictors; scheduled departure hour has a negative coefficient once the other variables are controlled for. Effects are consistent across the five years. |
| **Predictive power** | AUC-ROC of about **0.6** each year: a real but modest signal, as expected for rare events (<1% of flights). |

---

## 1️⃣ Part 1: Random Walk Metropolis-Hastings

**Target:** Laplace(0, 1), with density `f(x) = ½ e^{-|x|}`.

**Method:** a random walk Metropolis-Hastings sampler with a Normal proposal, using the log-acceptance criterion for numerical stability. `N = 10,000`, proposal standard deviation `s = 1`, burn-in of 1,000 iterations, starting at `x₀ = 0`, seed 42.

| Metric (Python run) | Estimate | Theoretical |
|---|---|---|
| Mean | ≈ 0.06 | 0 |
| Standard deviation | ≈ 1.32 | √2 ≈ 1.414 |
| Acceptance rate | ≈ 70.8% | n/a |

![Empirical vs theoretical density](images/histogram.png)

The empirical distribution matches the Laplace density closely, including the sharp peak at zero. The acceptance rate (~70%) is well above the ~44% optimum for 1-D random-walk Metropolis, which means the proposal is conservative: the chain moves in small steps, is autocorrelated, and the standard deviation converges more slowly than the mean.

### Gelman-Rubin convergence diagnostic

R̂ is computed over a grid of 50 proposal scales `s ∈ [0.001, 1]`, using 4 chains started from overdispersed values (−5, −2, 2, 5), 2,000 iterations each and a 500-iteration burn-in. Very small scales mix badly (R̂ is very large), and R̂ falls below the 1.05 threshold as soon as `s` becomes reasonable.

![Gelman-Rubin diagnostic](images/convergence.png)

---

## 2️⃣ Part 2: US Flights Analysis (2003–2007)

### Data and pipeline

- **Source:** Harvard Dataverse, [Airline On-Time Performance (ASA 2009 Data Expo)](https://doi.org/10.7910/DVN/HG7NV7). About 7M flights per year, about 35M in total.
- **Ingestion:** yearly `.csv.bz2` files read in **chunks of 500,000 rows** (Pandas) into a **SQLite** database, with indexes on year, tail number, origin, destination, carrier and diversion flag.
- **Reference tables:** `planes`, `airports`, `carriers`.
- **Feature engineering:** scheduled departure and arrival hour, aircraft age (flight year − manufacturing year), airport latitudes.
- **Cleaning:** cancelled flights excluded from delay analysis; missing `ArrDelay` removed; invalid manufacturing years dropped.

### (a) Best times and days to minimise delays

Average arrival delay by scheduled departure hour and day of week, per year. Slots with fewer than **1,000 flights** are excluded from the ranking, so that a handful of observations cannot produce a misleadingly good average.

![Delay heatmaps](images/delay_heatmaps.png)

A consistent "green zone" appears in the early morning: aircraft that overnight at the gate depart on time regardless of earlier disruptions. Later in the day, tight turnaround buffers let delays propagate through the network.

### (b) Impact of aircraft age on delays

![Aircraft age and delays](images/aircraft_age_delays.png)

There is no systematic upward trend with age. The increased variability at high ages likely reflects small samples and survivorship effects. This is consistent with modern maintenance regimes decoupling mechanical ageing from operational reliability.

### (c) Logistic regression for flight diversions

- **Model:** one logistic regression per year (2003–2007), with standardised features, an 80/20 stratified train/test split and `class_weight="balanced"` to handle the rarity of diversions (<1%).
- **Sampling:** 500,000 non-cancelled flights per year, drawn by selecting row IDs in Python instead of using the slow `ORDER BY RANDOM()` in SQLite (seed 42).
- **Features:** month, day of week, scheduled departure hour, scheduled arrival hour, distance, origin latitude, destination latitude, carrier (label-encoded).

![AUC-ROC by year](images/auc_by_year.png)

![Standardised coefficients](images/diversion_coefficients.png)

![Odds ratios](images/odds_ratios.png)

![Confusion matrices](images/confusion_matrices_5years.png)

The model has a high false-positive rate: structural conditions raise the risk, but actual diversions depend on unpredictable events (weather, incidents). The practical implication is that mitigation should target **system-level levers** (arrival schedule management, monitoring high-risk operating contexts) rather than individual aircraft.

---

## 📁 Repository Structure

```
.
├── README.md
├── part1_mcmc/
│   ├── part1.ipynb        # Python implementation
│   └── part1.Rmd          # R implementation
├── part2_flights/
│   ├── part2.ipynb        # Python implementation
│   └── part2.Rmd          # R implementation
├── images/                # Figures used in this README
└── report/                # Written report (PDF)
```

## ▶️ How to Reproduce

**Part 1** needs no external data.

**Part 2** requires the Data Expo files in a `data/` folder next to the notebook: `2003.csv.bz2` … `2007.csv.bz2`, `plane-data.csv`, `airports.csv`, `carriers.csv`. The first run builds the SQLite database `coursework.db`, which can take a long time (15–30 min in R); later runs reuse it.

```bash
# Python
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

```r
# R
install.packages(c("DBI", "RSQLite", "dplyr", "readr", "ggplot2", "tidyr",
                   "glue", "scales", "stringr", "caret", "pROC", "tibble", "forcats"))
rmarkdown::render("part2.Rmd", output_format = "html_document")
```

## ⚠️ Limitations

- Aircraft age relies on the `planes` table, where manufacturing year is missing or invalid for part of the fleet; those flights are dropped.
- Logistic regression assumes a linear effect on the log-odds; non-linear models could capture more structure.
- Diversions are rare and largely stochastic, so the discriminative power (AUC ≈ 0.6) is modest by nature.

## 👩‍💻 Author

**Eya Bibani**: BSc Data Science & Business Analytics, University of London
[LinkedIn](https://www.linkedin.com/) · [bibanieya1@gmail.com](mailto:bibanieya1@gmail.com)
