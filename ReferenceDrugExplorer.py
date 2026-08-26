import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

st.sidebar.title("🔗 Navigation")
st.sidebar.markdown("[Go to ML Explorer](http://localhost:8501)")

st.title("💊 Reference Drug Explorer")

# Static dictionary of reference drugs
reference_drugs = {
    "Diphenhydramine (H1)": {
        "smiles": "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",
        "chembl": "CHEMBL941",
        "pubchem": "3100",
        "indication": "Antihistamine for allergies, anaphylaxis"
    },
    "Cetirizine (H1)": {
        "smiles": "O=C(O)C(CN1CCN(CC1)C2=CC=CC=C2)C3=CC=CC=C3Cl",
        "chembl": "CHEMBL1201580",
        "pubchem": "2678",
        "indication": "Second‑generation antihistamine for allergic rhinitis, urticaria"
    },
    "Ranitidine (H2)": {
        "smiles": "CN(C)C=C(NC=C)SCc1nc2c(nc1N)ccc(c2)OCCN(C)C",
        "chembl": "CHEMBL109",
        "pubchem": "6579",
        "indication": "H2 blocker for peptic ulcers, GERD"
    },
    "Gefitinib (EGFR)": {
        "smiles": "COC1=CC=CC=C1OC2=NC3=C(C=C(C=C3)F)N=C2NC4=CC=CC=C4",
        "chembl": "CHEMBL941",
        "pubchem": "123631",
        "indication": "EGFR inhibitor for non‑small cell lung cancer"
    },
    "Erlotinib (EGFR)": {
        "smiles": "COC1=CC=CC=C1OC2=NC3=C(C=C(C=C3)C)N=C2NC4=CC=CC=C4",
        "chembl": "CHEMBL941",
        "pubchem": "123631",
        "indication": "EGFR inhibitor for non‑small cell lung cancer"
    }
}

# Dropdown for drug selection
selected_drug = st.selectbox("Select a reference drug:", list(reference_drugs.keys()))

drug_info = reference_drugs[selected_drug]

st.subheader(f"📌 {selected_drug}")
st.write(f"**Therapeutic use:** {drug_info['indication']}")
st.write(f"**SMILES:** {drug_info['smiles']}")

# Links
st.markdown(f"🔗 [ChEMBL page](https://www.ebi.ac.uk/chembl/compound_report_card/{drug_info['chembl']}/)")
st.markdown(f"🔗 [PubChem page](https://pubchem.ncbi.nlm.nih.gov/compound/{drug_info['pubchem']})")

# 2D structure
mol = Chem.MolFromSmiles(drug_info["smiles"])
if mol:
    img = Draw.MolToImage(mol, size=(300,300))
    st.image(img, caption=f"Structure of {selected_drug}")
