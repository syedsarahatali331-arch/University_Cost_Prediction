# University Cost Prediction — AI/ML Portfolio Project

Uses the **user-supplied International Education Costs dataset** with 907 rows.

## Target
`Total_Estimated_Cost_USD` is derived from the supplied cost fields:
**Tuition + monthly rent × 12 × duration + visa fee + annual insurance × duration**.

## ML workflow
Pandas → preprocessing → one-hot encoding → train/test split → Random Forest Regressor → MAE/R² → Joblib → Streamlit.

## Baseline result
MAE: **$1,862**  
R²: **0.9946**

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
