# London Weather Prediction Backtest — Project Specs

## Objectif

Backtester sur 1 an la stratégie suivante : comparer les prévisions de 2 modèles météo (GFS et ECMWF) avec la température réelle enregistrée à la station de résolution Polymarket (London City Airport — EGLC), puis simuler un paper trading basé sur ces prévisions.

Le but est de répondre à ces questions :
1. Quel modèle est le plus précis sur Londres ?
2. Quand les modèles convergent, le taux de réussite augmente-t-il ?
3. Une stratégie simple de "bet on model consensus" est-elle rentable ?
4. Quels sont les patterns saisonniers d'erreur des modèles ?

---

## Architecture

```
london-weather-backtest/
├── README.md
├── requirements.txt
├── config.py                  # Constantes, coordonnées, paramètres
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_forecasts.py     # Open-Meteo Previous Runs API
│   │   ├── fetch_actuals.py       # Open-Meteo Historical API (température réelle)
│   │   └── fetch_polymarket.py    # Optionnel : prix historiques PM
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── buckets.py             # Logique de tranches température PM
│   │   ├── strategy.py            # Stratégies de paper trading
│   │   └── engine.py              # Moteur de backtest principal
│   └── analysis/
│       ├── __init__.py
│       ├── model_accuracy.py      # Analyse erreurs modèles
│       ├── pnl.py                 # Calcul P/L et métriques
│       └── plots.py               # Visualisations matplotlib
├── data/                          # Données brutes et intermédiaires (gitignored)
│   ├── raw/
│   └── processed/
├── output/                        # Graphiques et rapports générés
│   ├── plots/
│   └── reports/
├── notebooks/                     # Optionnel : Jupyter pour exploration
│   └── exploration.ipynb
└── main.py                        # Point d'entrée CLI
```

---

## Sources de données

### 1. Prévisions modèles — Open-Meteo Previous Runs API

C'est la source principale. Cette API fournit les prévisions archivées des modèles météo avec un décalage de lead-time (ce que le modèle prédisait 1, 2, 3 jours avant).

**API endpoint :** `https://previous-runs-api.open-meteo.com/v1/forecast`

**Paramètres clés :**
```
latitude=51.5054    # London City Airport (EGLC)
longitude=0.0553    # London City Airport (EGLC)
start_date=2025-03-14
end_date=2026-03-14
daily=temperature_2m_max
timezone=Europe/London
past_days_offset=1   # Prévision faite 1 jour avant (J-1)
```

**Modèles à requêter :**
- GFS : `models=ncep_gfs025` — disponible depuis avril 2021
- ECMWF IFS : `models=ecmwf_ifs025` — disponible depuis ~janvier 2024

**Lead times à collecter :**
- `past_days_offset=1` (prévision J-1 = la plus pertinente pour le trading)
- `past_days_offset=2` (prévision J-2 = optionnel, pour étudier la convergence dans le temps)

**Variable :** `temperature_2m_max` (température maximale quotidienne à 2m)

**Important :**
- L'API est gratuite pour usage non-commercial, pas de clé API nécessaire
- Rate limit : respecter 10 requêtes/seconde max, ajouter un sleep entre les calls
- Les données sont en °C par défaut
- Documenter dans le README : https://open-meteo.com/en/docs/previous-runs-api

### 2. Température réelle — Open-Meteo Historical Forecast API

Pour obtenir la température réelle (ou très proche) enregistrée, on utilise le Historical Forecast API avec `past_days_offset=0` (= données quasi-observées, initialisées avec les mesures réelles).

**API endpoint :** `https://historical-forecast-api.open-meteo.com/v1/forecast`

**Paramètres :**
```
latitude=51.5054
longitude=0.0553
start_date=2025-03-14
end_date=2026-03-14
daily=temperature_2m_max
timezone=Europe/London
models=best_match
```

**Note importante sur la résolution Polymarket :**
- Polymarket Londres résout sur **Weather Underground, station London City Airport (EGLC)**
- URL de résolution : https://www.wunderground.com/history/daily/gb/london/EGLC
- La température est arrondie au **degré Celsius entier** (depuis ~2026 ; avant c'était en °F)
- Le backtest doit comparer avec cette granularité : `floor()` ou `round()` selon convention WU

**Alternative de validation :**
Si on veut valider la donnée Open-Meteo vs WU, on peut scraper WU pour quelques dates de référence. Mais pour le backtest principal, Open-Meteo Historical suffira car la différence avec la station EGLC est minime (même coordonnées).

### 3. (Optionnel) Prix historiques Polymarket — CLOB API

Pour une V2, on pourra récupérer les prix historiques des marchés température Londres.

**API endpoint :** `https://clob.polymarket.com/prices-history`

**Paramètres :**
```
market=<token_id>
interval=max
fidelity=720          # 12h en minutes (minimum pour marchés résolus)
```

**Découverte des marchés :** via Gamma API
```
GET https://gamma-api.polymarket.com/events?slug=highest-temperature-in-london-on-march-8-2026
```

**Limitations connues :**
- Les marchés résolus ne retournent des données qu'à granularité 12h minimum
- Il faut découvrir les token_id de chaque marché journalier (1 event par jour = 365 events à crawler)
- Le slug suit le pattern : `highest-temperature-in-london-on-{month}-{day}` ou `highest-temperature-in-london-on-{month}-{day}-{year}`
- Ces données sont un nice-to-have, pas un bloquant pour le backtest V1

**Pour la V1 du backtest, on n'a PAS besoin des données Polymarket.** On simule les prix du marché en utilisant une distribution uniforme naïve (chaque tranche à prix égal) ou une distribution basée sur la climatologie.

---

## Logique de backtest

### Tranches de température (buckets)

Polymarket structure ses marchés Londres en tranches de 2°C (peut varier). Exemples réels observés :
- "8°C or below"
- "9-10°C"
- "11-12°C"
- "13-14°C"
- "15°C or above"

Le nombre et la taille des tranches varient selon la saison et la date. Pour le backtest, implémenter un système de buckets configurable :

```python
# config.py
# Tranches par défaut (ajustables par saison)
DEFAULT_BUCKET_SIZE = 2  # degrés Celsius
DEFAULT_NUM_BUCKETS = 7  # nombre de tranches typique

def generate_buckets(center_temp: float, bucket_size: int = 2, num_buckets: int = 7) -> list[tuple]:
    """
    Génère les tranches centrées autour de la température attendue.
    Retourne une liste de tuples (min_inclusive, max_exclusive, label).
    La première tranche est "X or below", la dernière "Y or above".
    """
    ...
```

**Approche recommandée pour le backtest :**
Pour chaque jour, générer les buckets en se basant sur la climatologie historique de Londres pour ce jour de l'année (température moyenne ± écart type). Cela simule les buckets que PM aurait proposés.

### Stratégies à backtester

Implémenter au minimum ces 3 stratégies :

#### Stratégie 1 : "Naive Model Follow"
- Prendre la prévision J-1 du modèle GFS
- Identifier dans quelle tranche elle tombe
- Acheter YES sur cette tranche à un prix fixe simulé (ex: 30¢ = hypothèse marché efficient)
- Si la température réelle tombe dans la même tranche → WIN ($1), sinon → LOSS (-mise)

#### Stratégie 2 : "Best Model Follow"
- Même chose mais en utilisant le modèle avec la meilleure précision historique glissante (trailing 30 jours)
- Le modèle utilisé peut switcher dynamiquement entre GFS et ECMWF

#### Stratégie 3 : "Convergence Bet"
- Si GFS et ECMWF prédisent la même tranche → BET (les modèles convergent = signal fort)
- Si les modèles divergent → SKIP (pas de trade ce jour-là)
- Variante : quand ils convergent, acheter à un prix plus élevé (plus confiant) ; quand ils divergent, ne rien faire ou acheter les deux tranches à bas prix

### Simulation des prix

En l'absence de données PM historiques (V1), simuler les prix d'achat :

```python
# Hypothèses de prix simulés
PRICE_SCENARIOS = {
    "efficient_market": {
        # La tranche correspondant au consensus modèle est pricée à ~40-50¢
        # Les tranches adjacentes à ~15-20¢
        # Les tranches éloignées à ~2-5¢
        "consensus_bucket": 0.45,
        "adjacent_bucket": 0.18,
        "far_bucket": 0.03,
    },
    "inefficient_market": {
        # Simule un marché lent à s'ajuster (là où il y a edge)
        # La tranche correcte est sous-pricée
        "consensus_bucket": 0.30,
        "adjacent_bucket": 0.20,
        "far_bucket": 0.05,
    }
}
```

### Paramètres du backtest

```python
# config.py
BACKTEST_CONFIG = {
    "city": "london",
    "station_name": "London City Airport (EGLC)",
    "latitude": 51.5054,
    "longitude": 0.0553,
    "timezone": "Europe/London",
    "start_date": "2025-03-14",
    "end_date": "2026-03-14",
    "models": ["ncep_gfs025", "ecmwf_ifs025"],
    "lead_times": [1, 2],           # jours avant la date cible
    "stake_per_trade": 1.00,        # $1 par trade pour simplifier
    "bucket_size_celsius": 2,
    "temperature_variable": "temperature_2m_max",
}
```

---

## Métriques à calculer

### Accuracy modèles
- **MAE** (Mean Absolute Error) par modèle, globale et par mois
- **Bucket Accuracy** : % du temps où la prévision tombe dans la bonne tranche PM (c'est la métrique la plus importante)
- **Bias** : le modèle surestime-t-il ou sous-estime-t-il systématiquement ?
- **Error distribution** : histogramme des erreurs en °C

### Performance trading
- **Win rate** global et par stratégie
- **P/L cumulé** sur la période
- **P/L mensuel** (courbe de progression)
- **Win rate quand convergence** vs **win rate quand divergence**
- **Meilleur/pire mois** et raison (pattern météo)
- **Max drawdown**
- **Sharpe ratio** (si assez de données)

### Analyse saisonnière
- Accuracy et P/L par saison (hiver/printemps/été/automne)
- Identifier les périodes où le modèle est le plus/moins fiable
- Identifier les types de météo (fronts, anticyclones, etc.) qui causent les plus grosses erreurs

---

## Visualisations à générer (matplotlib)

Tous les graphiques doivent être sauvegardés en PNG dans `output/plots/`.

1. **Forecast vs Actual** : scatter plot GFS prediction vs actual, ECMWF prediction vs actual, avec ligne diagonale parfaite
2. **Error time series** : erreur quotidienne GFS et ECMWF sur 1 an, avec moving average 7j
3. **Monthly MAE bars** : barplot MAE par mois, GFS vs ECMWF côte à côte
4. **Bucket accuracy by month** : % de trades gagnants par mois pour chaque stratégie
5. **Cumulative P/L** : courbe P/L cumulé pour chaque stratégie sur 1 an
6. **Convergence analysis** : pie chart ou barplot montrant win rate quand convergence vs divergence
7. **Error distribution** : histogram des erreurs (°C) pour chaque modèle
8. **Seasonal heatmap** : heatmap jour_de_la_semaine × mois montrant la bucket accuracy
9. **Model bias** : bar chart de la bias moyenne par mois (over/under-prediction)

Style des graphiques :
- `plt.style.use('seaborn-v0_8-whitegrid')`
- Palette de couleurs cohérente : GFS en bleu (#2196F3), ECMWF en rouge (#F44336), Actual en vert (#4CAF50)
- Titres en français
- Tous les graphiques avec `tight_layout()` et `dpi=150`

---

## Output final

### Rapport Markdown auto-généré

Le script `main.py` doit produire un fichier `output/reports/backtest_report.md` qui résume :

```markdown
# Backtest Report — London Weather Trading
## Période : {start_date} → {end_date}
## Résumé exécutif
- Meilleur modèle : {best_model} (MAE: {mae}°C, Bucket Accuracy: {acc}%)
- Win rate stratégie convergence : {wr}%
- P/L simulé sur 1 an : ${pnl}
## Détail par modèle
...
## Détail par stratégie
...
## Analyse saisonnière
...
## Recommandations
...
```

### CSV de données intermédiaires

Sauvegarder les dataframes pandas dans `data/processed/` :
- `forecasts_daily.csv` : date, gfs_forecast, ecmwf_forecast, actual_temp, gfs_error, ecmwf_error
- `trades_log.csv` : date, strategy, bucket_bet, price, actual_bucket, result, pnl
- `monthly_summary.csv` : month, gfs_mae, ecmwf_mae, gfs_bucket_acc, ecmwf_bucket_acc, pnl_strategy1, pnl_strategy2, pnl_strategy3

---

## Contraintes techniques

- **Python 3.10+**
- **Dépendances** : pandas, numpy, matplotlib, requests, python-dateutil
- Pas de frameworks lourds (pas de Django, Flask, etc.)
- Le code doit tourner en une seule commande : `python main.py`
- Arguments CLI optionnels via argparse :
  - `--start-date` / `--end-date` pour modifier la période
  - `--models` pour choisir les modèles
  - `--strategy` pour lancer une seule stratégie
  - `--no-plots` pour skip la génération de graphiques
- Le fetch de données doit être idempotent : si les données sont déjà en cache dans `data/raw/`, ne pas re-télécharger (sauf flag `--refresh`)
- Logging avec le module `logging` standard, niveau INFO par défaut
- Tout le code en anglais (noms de variables, fonctions, docstrings)
- Les commentaires explicatifs peuvent être en français si ça aide à la compréhension
- Type hints partout
- Docstrings sur toutes les fonctions publiques

---

## Instructions pour Claude Code

### Ordre d'implémentation recommandé

1. **config.py** — Constantes, coordonnées EGLC, paramètres backtest
2. **fetch_forecasts.py** — Appels Open-Meteo Previous Runs API pour GFS + ECMWF
3. **fetch_actuals.py** — Appels Open-Meteo Historical API pour temp réelle
4. **buckets.py** — Génération des tranches PM et mapping température → tranche
5. **engine.py** — Moteur de backtest qui combine tout
6. **strategy.py** — Les 3 stratégies décrites ci-dessus
7. **model_accuracy.py** — Calculs MAE, bucket accuracy, bias
8. **pnl.py** — Calculs P/L et métriques trading
9. **plots.py** — Les 9 visualisations
10. **main.py** — CLI et orchestration

### Points d'attention

- **L'API Open-Meteo Previous Runs a des subtilités** : le paramètre `past_days_offset` n'est pas dans l'URL en tant que tel. Il faut utiliser le paramètre `&previous_day=1` pour obtenir les prévisions faites la veille. Consulter la doc : https://open-meteo.com/en/docs/previous-runs-api
- **Vérifier la disponibilité des données** : ECMWF IFS peut ne pas avoir de données pour toute la période demandée. Gérer les valeurs manquantes (NaN) proprement.
- **La température WU pour EGLC est arrondie au degré entier.** Le backtest doit reproduire cet arrondi sur la température réelle avant de déterminer la tranche gagnante : `actual_bucket_temp = round(actual_temp)`.
- **Attention au changement °F → °C** : Polymarket Londres est passé de °F (avant ~fin 2025) à °C (début 2026). Pour le backtest, on travaille en °C partout et on ne se préoccupe pas de cette transition.
- **Avant juin 2025**, les marchés météo Londres n'existaient peut-être pas sur Polymarket. Le backtest simule quand même le trading pour toute la période afin d'évaluer la performance des modèles, même si le marché réel n'existait pas.

### Ce qu'il ne faut PAS faire

- Ne PAS scraper Weather Underground (rate limited, ToS restrictifs). Utiliser Open-Meteo.
- Ne PAS implémenter de connexion Polymarket pour trading réel. C'est un backtest uniquement.
- Ne PAS stocker de clés API ou credentials dans le code.
- Ne PAS utiliser de base de données. Des CSV/parquet suffisent.
- Ne PAS sur-ingéniérer : pas de classes abstraites, pas de factory pattern, pas d'ORM. Du code procédural/fonctionnel simple et lisible.