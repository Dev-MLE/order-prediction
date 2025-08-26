# Feature Pipeline

This pipeline is responsible for **data ingestion, preprocessing, and feature engineering** before training models. It provides flexibility in how data is loaded and where processed features are stored.

---

## 🔹 Data Sources

You can load data into the pipeline from two main options:

1. **Local Files (Excel/CSV, etc.)**  
   - Place your raw files inside:  
     ```
     feature_pipeline/data/raw/
     ```
   - Example: load data from Excel using `load_data`.

2. **Databases (Any DB connection)**  
   - Configure a database connection and fetch data directly into the pipeline.  
   - This allows seamless integration with production databases.

> ✅ You can **switch between DB connection and local files** for data ingestion without changing preprocessing or feature engineering steps.

---

## 🔹 Pipeline Workflow

1. **Load Data**  
   - From local files (Excel, CSV, Parquet, etc.)  
   - Or directly from a database connection

2. **Preprocessing**  
   - Cleaning  
   - Handling missing values  
   - Aggregation  

3. **Feature Engineering**  
   - Creating domain-specific features  
   - Transformations for downstream models  

---

## 🔹 Storage Options

Processed data can be stored in one of two ways:

1. **Hopsworks Feature Store (Optional)**  
   - Store processed data as a `feature_group` in Hopsworks for better management and reuse.

2. **Local Storage**  
   - Save processed files into:  
     ```
     feature_pipeline/data/transformed/
     ```
   - These files can then be used directly in the **training_pipeline**.

---

## 🔹 Summary

- Flexible data ingestion (**local files or DB connection**)  
- Modular preprocessing and feature engineering  
- Output options:  
  - **Hopsworks Feature Store** (optional)  
  - **Local Transformed Data** → used in training  

This design allows you to easily plug in new data sources and reuse the same transformation steps across pipelines.

---
