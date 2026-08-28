import streamlit as st
import pandas as pd
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# --- Step 1: Choose how many molecules to import ---
num_import = st.slider(
    "Select number of molecules to import",
    min_value=50,
    max_value=1000,
    value=200,
    step=50
)

st.title("🧪 ML Explorer")
st.write("This page lets you select protein targets, fetch IC50 data, train ML models, and rank candidate molecules.")
# --- Sidebar info ---
st.sidebar.title("ℹ️ About the App")
st.sidebar.markdown("""
This app demonstrates cheminformatics workflows using ChEMBL data.
It lets you:
- Select target proteins
- Train ML models on IC50 data
- Visualize 2D structures
- Apply drug-likeness filters
""")

st.sidebar.subheader("⚠️ Disclaimer")
st.sidebar.markdown("This app is for **educational purposes only**. Do not use for medical decisions.")
st.sidebar.markdown("[Go to Reference Drug Explorer](http://localhost:8502)")

# --- Protein targets ---
protein_options = {
    "EGFR": "CHEMBL203",
    "VEGFR2": "CHEMBL4025",
    "CDK2": "CHEMBL301",
    "HIV Protease": "CHEMBL236",
    "Dopamine D2 receptor": "CHEMBL217",
    "Histamine H1 receptor": "CHEMBL231",
    "Histamine H2 receptor": "CHEMBL251",
    "Histamine H3 receptor": "CHEMBL264",
    "Histamine H4 receptor": "CHEMBL2034"
}

selected_protein = st.sidebar.selectbox("Select a protein target:", list(protein_options.keys()))
protein_id = protein_options[selected_protein]

feature_choice = st.sidebar.radio("Choose feature set", ["Fingerprints","Descriptors"])
selected_model_name = st.sidebar.selectbox("Select ML model:", 
    ["Random Forest","Gradient Boosting","AdaBoost","K-Nearest Neighbors","Linear Regression"])

# --- Fetch IC50 data via REST ---
def fetch_ic50(protein_id, limit=500):
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={protein_id}&standard_type=IC50&limit={limit}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()["activities"]
        df = pd.DataFrame(data)
        df = df[["molecule_chembl_id","canonical_smiles","standard_value"]].dropna()
        df.rename(columns={"canonical_smiles":"SMILES","standard_value":"Affinity"}, inplace=True)
        return df
    else:
        return pd.DataFrame(columns=["SMILES","Affinity"])

# --- Fetch IC50 data ---
df = fetch_ic50(protein_id)
st.subheader("IC50 Data Summary")
st.write(f"Total rows: {len(df)}")
st.write(df.describe())
# --- Step 3: Define slider BEFORE slicing ---
num_rows = st.slider(
    "Select number of rows to import",
    min_value=10,
    max_value=len(df),
    value=50,
    step=10
)
df1 = df.head(num_rows)   # safer than df[0:num_rows]

st.subheader(f"Showing first {num_rows} rows")
st.dataframe(df1)
# --- Candidate DataFrame (from your model or shortlist) ---
candidates_df = pd.DataFrame([
    {"chembl_id": "CHEMBL1001", "SMILES": "N#CC(C#N)Cc1ccc(O)cc1", "PredictedAffinity": 6.32, "Lipinski": True},
    {"chembl_id": "CHEMBL1002", "SMILES": "O=C(O)/C=C/c1ccc(O)cc1", "PredictedAffinity": 6.02, "Lipinski": True},
    {"chembl_id": "CHEMBL1003", "SMILES": "N#CC(C#N)=Cc1ccc(F)cc1", "PredictedAffinity": 6.03, "Lipinski": True},
])

st.subheader("Candidate Data Summary")
st.write(f"Total rows: {len(candidates_df)}")
st.write(candidates_df.describe())

df = fetch_ic50(protein_id)

if df.empty:
    st.warning("No data found for this target.")
    st.stop()
st.write("Dataset preview:", df.head())

# --- Feature extraction helpers ---
def featurize_single(smiles, feature_choice):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if feature_choice == "Descriptors":
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ]
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

@st.cache_data
def featurize_dataset(df, feature_choice):
    features, affinities = [], []
    for smiles, affinity in zip(df["SMILES"], df["Affinity"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if feature_choice == "Descriptors":
            feat = [
                Descriptors.MolWt(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol)
            ]
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            feat = arr
        features.append(feat)
        affinities.append(affinity)
    return features, np.log10(pd.Series(affinities).astype(float))

# --- Build dataset ---
with st.spinner("Extracting features..."):
    X, y = featurize_dataset(df, feature_choice)

if len(X) < 50:
    st.error("Too few valid molecules for training.")
    st.stop()

# --- Compare R² across models ---
st.subheader("📊 Model Performance Comparison")
model_options = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Linear Regression": LinearRegression()
}
results_models = []
for name, mdl in model_options.items():
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        r2 = r2_score(y_test, preds)
        results_models.append((name, r2))
    except Exception:
        results_models.append((name, None))
perf_df = pd.DataFrame(results_models, columns=["Model","R²"]).set_index("Model")
st.bar_chart(perf_df)
st.write(perf_df)

# --- Train selected model ---
model = model_options[selected_model_name]
model.fit(X,y)
df["PredictedAffinity"] = model.predict(X)

ranked = df.sort_values("PredictedAffinity", ascending=False)
st.subheader("Top candidate molecules")
ranked["ChEMBL_Link"] = ranked["molecule_chembl_id"].apply(
    lambda x: f"https://www.ebi.ac.uk/chembl/compound_report_card/{x}/" if pd.notnull(x) else "N/A"
)
st.write(ranked.head(10)[["molecule_chembl_id","SMILES","PredictedAffinity","ChEMBL_Link"]])

# --- 2D structures ---
st.subheader("🧪 2D Structures of Top Candidates")
top_mols = [Chem.MolFromSmiles(s) for s in ranked.head(10)["SMILES"]]
img = Draw.MolsToGridImage(top_mols, molsPerRow=5, subImgSize=(200,200))
st.image(img, caption="Top candidate molecules (2D structures)")
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Example candidate DataFrame
candidates_df = pd.DataFrame([
    {"chembl_id": "CHEMBL25", "smiles": "CN(C)C(=O)c1ccc(O)cc1", "score": 0.92},
    {"chembl_id": "CHEMBL120", "smiles": "CCOC(=O)c1ccc(N)cc1", "score": 0.89},
])

st.title("Candidate Molecules")

# Show table
st.dataframe(candidates_df)
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Example: your real results table
candidates_df = pd.DataFrame([
    {"chembl_id": "CHEMBL1001", "smiles": "N#CC(C#N)Cc1ccc(O)cc1", "PredictedAffinity": 6.32, "Lipinski": True},
    {"chembl_id": "CHEMBL1002", "smiles": "O=C(O)/C=C/c1ccc(O)cc1", "PredictedAffinity": 6.02, "Lipinski": True},
    {"chembl_id": "CHEMBL1003", "smiles": "N#CC(C#N)=Cc1ccc(F)cc1", "PredictedAffinity": 6.03, "Lipinski": True},
])

st.title("Top Candidate Molecules")

# Show table
st.dataframe(candidates_df)

# Function to render molecule image
def mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(200,200))
print(candidates_df.columns)

st.subheader("Structures with ChEMBL Labels")
for _, row in candidates_df.iterrows():
    img = mol_image(row["smiles"])
    st.image(
    img,
    caption=f"{row['chembl_id']} | Affinity: {row['PredictedAffinity']:.2f} | Lipinski: {row['Lipinski']}"
)

    #st.image(img, caption=f"{row['chembl_id']} | Affinity: {row['PredictedAffinity']:.2f} | Lipinski: {row['Lipinski']}")

# Function to render molecule image
def mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(200,200))

st.subheader("Structures with Labels")
for _, row in candidates_df.iterrows():
    img = mol_image(row["smiles"])
    st.image(img, caption=f"{row['chembl_id']}")

# --- Lipinski filter ---
def lipinski(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    return (mw < 500 and logp < 5 and h_donors <= 5 and h_acceptors <= 10)

ranked["Lipinski"] = [lipinski(Chem.MolFromSmiles(s)) for s in ranked["SMILES"]]
st.subheader("💊 Drug-likeness (Lipinski filter)")
st.write(ranked.head(20)[["SMILES","PredictedAffinity","Lipinski"]])

# --- Scatter plot ---
st.subheader("📈 Predicted vs Actual Affinities")
fig, ax = plt.subplots()
ax.scatter(y, model.predict(X), alpha=0.6)
ax.set_xlabel("Actual log(IC50)")
ax.set_ylabel("Predicted log(IC50)")
ax.set_title("Model performance")
st.pyplot(fig)

# --- CSV download ---
st.subheader("📥 Download Results")
lipinski_csv = ranked[["SMILES","PredictedAffinity","Lipinski"]].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Lipinski-filtered candidates (CSV)",
    data=lipinski_csv,
    file_name="lipinski_candidates.csv",
    mime="text/csv"
)
st.write("developed by subramanian ramajaym with copilot'assistance")
