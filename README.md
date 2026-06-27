# 🚛 Geospatial Fleet Tracking & Optimization Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-Q2UnOJGdfQ8lYBFVx0iWeYIf3tlI0YE?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end automated Python pipeline that transforms raw GPS telematics data from an 86-vehicle delivery fleet into actionable financial and operational intelligence. The pipeline monitors driver behavior, flags safety violations, quantifies fuel waste, and segments drivers into behavioral profiles using unsupervised machine learning.

> **Note:** All data in this repository is synthetically generated to replicate real-world fleet telematics patterns. No proprietary or confidential data is used.

---

## 📋 Table of Contents

- [Project Background](#-project-background)
- [Key Engineering Challenges](#-key-engineering-challenges)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology](#-methodology)
- [Machine Learning](#-machine-learning-k-means-clustering)
- [Outputs](#-outputs)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 📌 Project Background

Raw GPS telematics data is notoriously unreliable. Satellite interference causes coordinates to "teleport," multiple agents sharing a single vehicle create conflicting data streams, and signal gaps during rural deliveries leave holes in the positional record. Standard fleet tracking tools either ignore these issues or produce silently corrupted results.

This pipeline was engineered to solve those problems at the source — applying physics-based validation, sensor election logic, and multi-layer signal filtering before any metric is calculated. The result is a defensible, auditable dataset that quantifies exactly how much money a fleet loses to unauthorized detours and excessive idling, while simultaneously surfacing safety liabilities.

---

## 🔧 Key Engineering Challenges

### 1. GPS "Ping-Pong" Drift — Primary Sensor Election
When multiple delivery agents ride in the same truck, each agent's mobile device broadcasts its own GPS pings. This produces conflicting, overlapping coordinate streams for the same vehicle, causing severe distance inflation if naïvely summed.

**Solution:** For every trip, the pipeline counts pings per agent and elects the agent with the highest ping density as the sole primary data source for all kinematic calculations. Secondary agents are retained only for entity-level attribution (company, site, batch).

---

### 2. GPS Teleportation Glitches — 3-Layer Speeding Mitigation
Calculating speed from raw GPS (`Δdistance / Δtime`) is highly susceptible to satellite interference. A single corrupted coordinate can produce a calculated speed of 500+ km/h, making naïve speeding detection unusable.

The pipeline applies a **3-Layer Mitigation** system before logging any speeding event:

| Layer | Mechanism | Rationale |
|---|---|---|
| **Rolling Average** | Speed is smoothed over 3 consecutive pings | A GPS glitch lasts a single ping; sustained high speed over multiple readings is real |
| **Physics Limiter** | Acceleration must be ≤ 3.0 m/s² | A fully loaded delivery truck cannot accelerate like a sports car; faster = glitch |
| **Operational Time Filter** | Events only logged between 08:00–17:00 | Eliminates off-hours noise from parked or repositioning vehicles |

A speeding event is only logged when **all three layers** are satisfied simultaneously.

---

### 3. Signal Gap Distance Errors — Smart Gap Detection
When a truck enters a dead zone (tunnel, rural area, signal shadow), GPS pings may be missing for 10+ minutes. A naïve straight-line Haversine calculation between the last and next available ping significantly under-reports actual distance traveled.

**Solution:** The `calculate_smart_distance` function detects gaps exceeding 10 minutes with displacement greater than 500 m, and escalates to an OSRM road-network query for that segment — returning the realistic road distance rather than the straight-line approximation.

---

### 4. Harsh Braking Detection — Physics-Bounded Thresholds
Harsh braking is a leading indicator of aggressive driving, tailgating, and accelerated vehicle wear. It is calculated by measuring deceleration between consecutive pings.

**Thresholds:**
- **Flagged:** Deceleration between **-3.0 m/s²** and **-11.0 m/s²**
- **Below -11.0 m/s²:** Physically impossible for a loaded logistics truck without a collision — attributed to GPS signal bounce and ignored

---

### 5. Idle Analysis — Geofenced Red-Flag Detection
Not all stationary time is waste. A truck parked at a customer drop-off is expected; a truck sitting idle for 45+ minutes in an unrecognized location is a financial and operational concern.

**Logic:**
- Stationary pings within **50 m of a planned drop-off coordinate** → classified as legitimate turnaround time
- Stationary pings outside drop-off zones → tracked as potential idle events
- Idle events **≥ 45 minutes** during operational hours (08:00–17:00) → flagged as red-flag idle
- A **200 m geofence** around all registered warehouse coordinates exempts depot idle time from financial waste calculations

---

### 6. Four-Stage Data Quality Firewall
Rather than producing silently incorrect outputs, the pipeline explicitly drops corrupted trips and logs the reason for every rejection:

| Drop Condition | Trigger | Reason |
|---|---|---|
| **High Glitch Ratio** | > 5% of pings exceed 45 m/s | GPS signal too corrupted to trust |
| **OSRM Impossible** | Trip duration < 20% of OSRM minimum drive time | Massive teleportation — not a real trip |
| **Physically Impossible** | Avg speed > 80 km/h or distance > 1,000 km | Residual data corruption |
| **Ghost / Aborted Shift** | Zero route deviation AND zero turnaround time | No real driving activity detected |

All dropped trips are written to a separate diagnostic sheet with their drop reason, ensuring full auditability.

---

## 🏗️ Pipeline Architecture

```
Raw GPS Pings (SQL Server)
        │
        ▼
┌─────────────────────────┐
│  Day-by-Day Extraction  │  ← Prevents SQL timeout on large telematics tables
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Data Cleaning &       │  ← Type coercion, null removal, zero-coordinate filter
│   Type Coercion         │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Primary Agent Election │  ← Resolves multi-agent ping conflicts per trip
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Kinematic Calculation  │  ← Haversine + Smart Gap (OSRM escalation)
│  Speed & Acceleration   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  4-Stage Quality        │  ← Drop corrupted trips, log with diagnosis
│  Firewall               │
└─────────────────────────┘
        │
        ▼
┌───────────────┬──────────────────┬──────────────────┐
│  3-Layer      │  Harsh Braking   │  Geofenced Idle  │
│  Speeding     │  Detection       │  Analysis        │
│  Mitigation   │                  │                  │
└───────────────┴──────────────────┴──────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Route Deviation &      │  ← OSRM optimal route vs. actual distance
│  Financial Calculation  │  ← Detour waste + idle burn cost in PHP
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  K-Means Clustering     │  ← 4 behavioral profiles, empirical thresholds
│  (Scikit-Learn)         │
└─────────────────────────┘
        │
        ▼
   Excel Output (5 Sheets) → Power BI Dashboards
```

---

## 📐 Methodology

### Route Deviation Percentage
The core efficiency metric — quantifies how much extra distance a driver traveled relative to the mathematically optimal route:

```
Route Deviation % = ((Actual Distance − Optimal Distance) / Optimal Distance) × 100
```

A **zero-clamp fail-safe** is applied: when a driver uses an unmapped shortcut and actual distance is marginally shorter than the OSRM optimal, the result is clamped to 0.0% rather than producing a negative deviation.

### Financial Waste Calculations

```
Detour Waste (PHP) = (Extra Distance km / Vehicle Km·L⁻¹) × Fuel Price PHP·L⁻¹

Idle Waste (PHP)   = (Red Flag Idle Hours × Idle Burn Rate L·hr⁻¹) × Fuel Price PHP·L⁻¹

Total Fuel Cost    = (Actual Distance km / Vehicle Km·L⁻¹) × Fuel Price PHP·L⁻¹ + Idle Waste
```

### Empirical Threshold Anchoring
Rather than setting arbitrary management targets, all alert thresholds are anchored to the empirical averages of the fleet's dominant behavioral cluster. This ensures every alert means: *"this vehicle is performing worse than the majority of its own peers"* — a statistically defensible and operationally meaningful standard.

---

## 🤖 Machine Learning: K-Means Clustering

The pipeline uses **unsupervised K-Means clustering** (K=4) to segment the fleet into distinct behavioral profiles. K=4 was selected through elbow method and silhouette score analysis across K=2–8.

**Feature Set:**

| Feature | Operational Meaning |
|---|---|
| `Route_Deviation_Pct` | Routing efficiency |
| `Avg_Turnaround_Per_Drop_Mins` | Delivery handoff speed |
| `Harsh_Brake_Count` | Driving aggression proxy |
| `Speeding_Event_Count` | Speed compliance |

All features are standardized using `StandardScaler` before clustering to prevent high-magnitude features from dominating the distance calculations.

---

## 📤 Outputs

The pipeline produces a single Excel workbook with five sheets:

| Sheet | Contents |
|---|---|
| `Trip_Summaries` | Main KPI output — one row per trip per company/site entity |
| `Dropped_Trips` | Diagnostic log for data quality failures with drop reason |
| `Red_Flag_Locations` | Geostamped suspicious idle events for map overlays |
| `Harsh_Brake_Locations` | Geostamped braking events for incident heatmaps |
| `Speeding_Locations` | Geostamped speed violations for route corridor analysis |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and pipeline orchestration |
| `numpy` | Vectorized kinematic calculations |
| `scikit-learn` | K-Means clustering, StandardScaler |
| `SQLAlchemy` + `pyodbc` | SQL Server connection and day-by-day extraction |
| `OSRM` *(self-hosted)* | Road-network routing and optimal route calculation |
| `requests` | OSRM and reverse geocoding API calls |
| `openpyxl` | Multi-sheet Excel export |
| `tqdm` | Progress tracking for long-running trip loops |
| `matplotlib` | Cluster visualization and elbow/silhouette charts |

---

## ▶️ How to Run

**In Google Colab (recommended):**

Click the **Open in Colab** badge at the top of this README. All cells are annotated — run them top to bottom using **Runtime → Run all**.

**Locally:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install pandas numpy scikit-learn requests tqdm openpyxl matplotlib
jupyter notebook geospatial_fleet_tracking.ipynb
```

> The notebook uses synthetically generated data and requires no external database or API connections to run in demo mode.

---

*Built with Python 3 · pandas · NumPy · scikit-learn · Haversine · OSRM · openpyxl · matplotlib*
