
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)

from predict import predict_solubility, validate_smiles


# Page configuration
st.set_page_config(
    page_title="MolecularAI",
    page_icon="🧬",
    layout="wide",
)


# Header
st.title("🧬 MolecularAI")
st.subheader("AI-powered Molecular Property Prediction")

st.write(
    "Predict aqueous solubility, visualize molecular structures, "
    "and compare the chemical properties of molecules."
)

st.divider()


# Choose application mode
mode = st.radio(
    "Choose a mode",
    ["Single Molecule", "Compare Molecules"],
    horizontal=True,
)


def analyze_molecule(smiles):
    """Validate and analyze one molecular structure."""

    if not validate_smiles(smiles):
        raise ValueError(
            f"Invalid SMILES representation: {smiles}"
        )

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Could not construct molecule: {smiles}"
        )

    prediction = predict_solubility(smiles)

    descriptors = {
        "Molecular weight": Descriptors.MolWt(molecule),
        "H-bond donors": Lipinski.NumHDonors(molecule),
        "H-bond acceptors": Lipinski.NumHAcceptors(molecule),
        "Rotatable bonds": Lipinski.NumRotatableBonds(molecule),
        "TPSA": rdMolDescriptors.CalcTPSA(molecule),
        "LogP": Descriptors.MolLogP(molecule),
    }

    image = Draw.MolToImage(
        molecule,
        size=(450, 350),
    )

    return prediction, descriptors, image


def display_molecule(smiles, heading):
    """Display a molecule's prediction, structure, and descriptors."""

    prediction, descriptors, image = analyze_molecule(smiles)

    st.subheader(heading)
    st.caption(f"SMILES: {smiles}")

    st.metric(
        "Predicted log solubility",
        f"{prediction:.3f}",
    )

    st.caption(
        "Model estimate, not an experimental measurement. "
        "Accuracy depends on the model and its training data."
    )

    st.image(image, width="stretch")

    st.subheader("Molecular Descriptors")

    st.table(
        [
            {
                "Descriptor": name,
                "Value": round(float(value), 3),
            }
            for name, value in descriptors.items()
        ]
    )

    return prediction, descriptors


# Single-molecule mode
if mode == "Single Molecule":
    smiles = st.text_input(
        "Enter SMILES",
        value="CCO",
        help=(
            "Examples: CCO (ethanol), "
            "CC(=O)O (acetic acid), O=C=O (carbon dioxide)"
        ),
    ).strip()

    if st.button("Predict Solubility", type="primary"):
        try:
            display_molecule(smiles, "Molecule Analysis")
            st.success("Analysis completed!")

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# Two-molecule comparison mode
else:
    col1, col2 = st.columns(2)

    with col1:
        smiles_a = st.text_input(
            "First molecule — SMILES",
            value="CCO",
            key="molecule_a",
        ).strip()

    with col2:
        smiles_b = st.text_input(
            "Second molecule — SMILES",
            value="CC(=O)O",
            key="molecule_b",
        ).strip()

    if st.button("Compare Molecules", type="primary"):
        try:
            # Validate both before displaying results
            if not validate_smiles(smiles_a):
                raise ValueError(
                    "The first molecule has invalid SMILES."
                )

            if not validate_smiles(smiles_b):
                raise ValueError(
                    "The second molecule has invalid SMILES."
                )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                prediction_a, descriptors_a = display_molecule(
                    smiles_a, "Molecule A"
                )

            with result_col2:
                prediction_b, descriptors_b = display_molecule(
                    smiles_b, "Molecule B"
                )

            st.divider()
            st.subheader("Comparison Summary")

            comparison_rows = [
                {
                    "Property": "Predicted log solubility",
                    "Molecule A": round(float(prediction_a), 3),
                    "Molecule B": round(float(prediction_b), 3),
                }
            ]

            for name in descriptors_a:
                comparison_rows.append(
                    {
                        "Property": name,
                        "Molecule A": round(
                            float(descriptors_a[name]), 3
                        ),
                        "Molecule B": round(
                            float(descriptors_b[name]), 3
                        ),
                    }
                )

            st.table(comparison_rows)

            st.divider()
            st.subheader("Visual Property Comparison")

            # Compare predicted solubility
            st.markdown("**Predicted log solubility**")

            solubility_chart = pd.DataFrame(
                {
                    "Molecule": ["Molecule A", "Molecule B"],
                    "Value": [
                        float(prediction_a),
                        float(prediction_b),
                    ],
                }
            )

            st.bar_chart(
                solubility_chart,
                x="Molecule",
                y="Value",
            )

            # Compare descriptors individually because they have
            # different units and scales.
            for name in descriptors_a:
                st.markdown(f"**{name}**")

                chart_data = pd.DataFrame(
                    {
                        "Molecule": ["Molecule A", "Molecule B"],
                        "Value": [
                            float(descriptors_a[name]),
                            float(descriptors_b[name]),
                        ],
                    }
                )

                st.bar_chart(
                    chart_data,
                    x="Molecule",
                    y="Value",
                )

            st.info(
                "Compare the predicted values as model estimates. "
                "A difference between predictions does not establish "
                "which molecule is experimentally more soluble."
            )

        except Exception as error:
            st.error(f"Comparison failed: {error}")


# Footer
st.divider()

st.caption(
    "MolecularAI | Experimental molecular property prediction "
    "using RDKit and Random Forest."
)
