# Machine Learning Explorer & Reference Drug Explorer

## Overview
This Streamlit project contains two linked apps:
1. **ML Explorer** – Fetches IC50 data from ChEMBL, extracts molecular features, trains ML models, and ranks candidate molecules.
2. **Reference Drug Explorer** – Provides curated information on common reference drugs, including SMILES strings, therapeutic uses, and links to ChEMBL/PubChem.

Both apps are organized as a multi‑page Streamlit app. The main entry point is `ml_app.py`, and the Reference Drug Explorer is located in the `pages/` folder.

## Features
- Fetch IC50 datasets for selected protein targets via ChEMBL REST API
- Generate molecular descriptors or fingerprints using RDKit
- Train multiple ML models (Random Forest, Gradient Boosting, AdaBoost, KNN, Linear Regression)
- Compare model performance with R² scores
- Rank candidate molecules by predicted affinity
- Visualize top molecules in 2D
- Apply Lipinski drug‑likeness filter
- Download results as CSV
- Explore reference drugs with structures and external links

## Project Structure
