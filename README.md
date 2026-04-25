# 📊 ECLEARNIX Data Intelligence Dashboard

## 📖 Overview
ECLEARNIX Dashboard is an interactive, visually appealing data intelligence application built with Streamlit. It provides comprehensive insights, rich data visualizations, and predictive analytics regarding user behaviors, course completions, and engagement metrics within the ECLEARNIX platform.

## ✨ Features
- **Dynamic Data Filtering:** Easily drill down data by specific Regions and Departments.
- **Raw Data Preview:** Expandable section to inspect the underlying dataset on demand.
- **Interactive Visualizations:**
  - Course Completion Distributions
  - Event Registration Analytics by Region
  - Feedback Rating Analysis
  - Average Time Spent across various Departments
  - App Installation & Newsletter Subscription Rates
- **Advanced Analytics:** 
  - Visualizes distinct User Segments (Clustering).
  - Displays Machine Learning-based Predicted Course Completions.

## 🛠️ Tech Stack
- **Frontend UI / Data App Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Plotly Express, Seaborn, Matplotlib
- **Machine Learning Packages:** Scikit-Learn, XGBoost

## 📁 Project Structure
```text
eclearnix-dashboard/
│
├── data/
│   ├── notebooks/               # Jupyter Notebooks for EDA, clustering, and modeling
│   └── ECLEARNIX.csv            # Core dataset (automatically attempts to load clustered/predicted variants)
│
├── streamlit_app.py             # Main Streamlit dashboard application code
├── requirements.txt             # Python dependencies required to run the project
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed on your local machine.

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/sri-vathsun/eclearnix-dashboard.git
   cd eclearnix-dashboard
   ```

2. **Install the required dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
Launch the Streamlit application by executing the following command in your terminal:
```bash
streamlit run streamlit_app.py
```
The dashboard will automatically open in your default web browser (typically accessible at `http://localhost:8501`).

## 🔮 Future Enhancements
- Integration with real-time database endpoints instead of static CSVs.
- Extended predictive models targeting user churn and course recommendations.
- Interactive exporting of custom analytical reports directly from the UI.
