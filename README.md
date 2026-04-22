# 😴 Sleep Disorder Prediction System

A machine learning system that predicts potential sleep disorders — **None**, **Insomnia**, or **Sleep Apnea** — based on lifestyle and health data. Built with Python, LightGBM, and served via a FastAPI backend integrated with a Next.js frontend.

---

## 📁 Dataset

- **Source:** [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/) — Kaggle
- **Size:** 374 entries, 13 columns
- **Target Variable:** `Sleep Disorder` (None, Sleep Apnea, Insomnia)
- **Class Distribution:**
  - None: 219 samples (58.6%)
  - Sleep Apnea: 78 samples (20.9%)
  - Insomnia: 77 samples (20.6%)

---

## 🧠 Machine Learning Pipeline

### 1. Initial Data Understanding
- Inspected dataset structure: 374 rows × 13 columns
- Identified data types: 7 integer, 1 float, 5 object columns
- No missing values detected across all columns
- Descriptive statistics computed for all numeric features

### 2. Data Cleaning
Key cleaning steps performed:
- **Column Renaming** — Renamed all columns to descriptive Indonesian names (e.g., `Sleep Duration` → `Durasi_Tidur`)
- **Irrelevant Column Removal** — Dropped `Person ID` as it has no predictive value
- **Missing Value Handling** — Confirmed zero missing values; median/mode imputation strategy prepared
- **Data Consistency** — Standardized `Pekerjaan` (Occupation) text to Title Case
- **Label Encoding** — Encoded `Gangguan_Tidur` (Sleep Disorder): `None=0`, `Sleep Apnea=1`, `Insomnia=2`
- **Blood Pressure Splitting** — Parsed `Tekanan_Darah` (e.g., `120/80`) into separate `Sistolik` and `Diastolik` numeric features

### 3. Exploratory Data Analysis (EDA)

#### Univariate Analysis
- Sleep duration distributes bimodally — concentrated around 6–7h and 7.5–8.5h
- Stress level and heart rate show distinct patterns across disorder classes

#### Bivariate Analysis
- Sleep Apnea patients tend to have **higher heart rate** and **shorter sleep duration**
- Insomnia patients show the **lowest sleep quality** scores
- Nurses and Sales Representatives have the highest disorder prevalence

#### Multivariate Analysis — Correlation Heatmap

| Feature Pair | Correlation |
|---|---|
| Kualitas_Tidur ↔ Tingkat_Stres | **-0.90** (strong negative) |
| Durasi_Tidur ↔ Tingkat_Stres | -0.81 |
| Tingkat_Aktivitas_Fisik ↔ Langkah_Harian | **0.77** (strong positive) |
| Denyut_Jantung ↔ Tingkat_Stres | 0.67 |

### 4. Data Preparation for Modeling

- **Feature/Target Split:** X = 12 features, y = `Gangguan_Tidur`
- **Train-Test Split:** 80% train (299 samples) / 20% test (75 samples), stratified
- **Preprocessing Pipeline (ColumnTransformer):**
  - Numeric features → `StandardScaler`
  - Categorical features (`Jenis_Kelamin`, `Pekerjaan`, `Kategori_BMI`) → `OneHotEncoder`
  - Final processed shape: (299, 25) train / (75, 25) test

### 5. Model Training & Evaluation

Five models were trained and compared on the same preprocessing pipeline:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Baseline (Majority Class) | 0.5867 | — | — | — | — |
| Logistic Regression | 0.8933 | 0.8993 | 0.8933 | 0.8947 | 0.9045 |
| Logistic Regression (Balanced) | 0.8800 | 0.8893 | 0.8800 | 0.8824 | 0.9193 |
| Decision Tree | 0.8667 | 0.8724 | 0.8667 | 0.8581 | 0.8538 |
| Random Forest | 0.8667 | 0.8724 | 0.8667 | 0.8581 | 0.9172 |
| **LightGBM** ⭐ | **0.9200** | **0.9205** | **0.9200** | **0.9186** | **0.9212** |

> ⚠️ **Note on Random Forest:** Despite decent overall accuracy, Random Forest showed critically low recall (0.56) for Sleep Apnea — meaning it missed nearly half of actual Sleep Apnea cases. This makes it unsuitable for health-related use cases where false negatives are costly.

### 6. Model Interpretation

#### Top 15 Feature Importance (LightGBM)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Durasi_Tidur (Sleep Duration) | 982 |
| 2 | Usia (Age) | 799 |
| 3 | Langkah_Harian (Daily Steps) | 197 |
| 4 | Sistolik (Systolic BP) | 194 |
| 5 | Denyut_Jantung (Heart Rate) | 193 |
| 6 | Tingkat_Aktivitas_Fisik (Physical Activity) | 152 |
| 7 | Kualitas_Tidur (Sleep Quality) | 103 |
| 8 | Diastolik (Diastolic BP) | 103 |
| 9 | Kategori_BMI_Normal | 58 |
| 10 | Tingkat_Stres (Stress Level) | 47 |

#### Key Findings
- **Sleep duration** is the single most important predictor — significantly lower in Insomnia patients
- **Age** is the second strongest signal — Sleep Apnea risk increases with age
- **Physiological indicators** (blood pressure, heart rate) are more predictive than occupation or gender

### 7. Final Model

- **Chosen Model:** LightGBM (via scikit-learn Pipeline)
- **Why LightGBM:** Best F1 Score (0.9186), best ROC AUC (0.9212), and most balanced performance across all three disorder classes
- **Model Serialization:** Saved as `lightgbm_sleep_pipeline.pkl` using `joblib` — includes full preprocessing pipeline, ready for direct inference

---

## 🛠️ Technologies Used

### Machine Learning
- **Python 3.11** — Core language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Preprocessing, pipeline, model evaluation
- **LightGBM** — Gradient boosting classifier
- **Matplotlib & Seaborn** — EDA visualizations
- **Joblib** — Model serialization

### Backend
- **FastAPI** — High-performance REST API
- **Pydantic** — Data validation & schema management
- **python-dotenv** — Environment management

### Frontend
- **Next.js (React)** — Frontend framework
- **React Three Fiber** — 3D visual rendering
- **shadcn/ui** — Prebuilt UI components
- **TailwindCSS** — Utility-first CSS styling
- **Framer Motion** — Animations
- **Zod + React Hook Form** — Frontend validation

---

## 🚀 Setup & Installation

### Backend (FastAPI)

```bash
git clone <repository_url>
cd backend
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs on: `http://127.0.0.1:8000`

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_KEY=apikeystress
```

App runs on: `http://localhost:3000`

### Testing

```bash
cd backend
pytest
```

### Docker (Backend)

```bash
cd backend
docker build -t project-ml-backend .
docker run -d -p 8000:8000 --name project-ml-backend project-ml-backend
```

---

## 📊 API Prediction Input

The model accepts the following features as input:

| Feature | Type | Description |
|---|---|---|
| Jenis_Kelamin | categorical | Gender (Male / Female) |
| Usia | numeric | Age (years) |
| Pekerjaan | categorical | Occupation |
| Durasi_Tidur | numeric | Sleep duration (hours) |
| Kualitas_Tidur | numeric | Sleep quality score (1–9) |
| Tingkat_Aktivitas_Fisik | numeric | Physical activity level (30–90) |
| Tingkat_Stres | numeric | Stress level (1–8) |
| Kategori_BMI | categorical | BMI category |
| Tekanan_Darah | string | Blood pressure (e.g., `120/80`) |
| Denyut_Jantung | numeric | Resting heart rate (bpm) |
| Langkah_Harian | numeric | Daily steps count |

**Output:** Predicted class (`None`, `Sleep Apnea`, or `Insomnia`) + probability scores per class

---

## 📄 License

Licensed under the MIT License. See `LICENSE` for details.

---

*Model developed by Bayu Putra Pamungkas as part of the Sleep Health Prediction System project.*
