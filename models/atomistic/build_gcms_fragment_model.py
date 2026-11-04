"""
Build the GC/MS fragment models: one fragment pool per GC/MS ion a model can carry.

E. coli core
------------
A GC/MS ion retains a known subset of its metabolite's carbons -- Ala232 is alanine C2-C3,
Ala260 is C1-C3 -- so each row of data/GCMS_knowns.csv whose parent the core model holds
and whose carbon set is unambiguous becomes a fragment pool (see
:func:`sigmaflux.core.measurement_pools.add_fragment_pools`). Rows are skipped, with the
reason printed, when the parent is absent from the core, is a mixture the knowns do not
resolve (F6P_G6P, CitIcit, P5P_dp), or the carbon set is ambiguous ("1,2 OR 3,4", "not
stated", multi-ion rows).

Also writes E_coli_core_fragment_mapping.csv, mapping the Long & Antoniewicz (2019)
workbook's fragment names onto the fragment ids with the formula and backbone carbon
count that ``read_xlsx_mid_data`` needs to correct the raw MIDs for natural isotopes.

Run this whenever its inputs or the annotated models change:

    python models/atomistic/build_gcms_fragment_model.py
"""

import re
from pathlib import Path

import cobra
import pandas as pd

from sigmaflux.core.measurement_pools import add_fragment_pools, check_mapping_csv

ROOT = Path(__file__).resolve().parents[2]
KNOWNS = ROOT / "data" / "GCMS_knowns.csv"

#: Knowns ``Abbrev`` -> core metabolite. Glucose is glycogen-bound and ribose RNA-bound,
#: so they report the G6P and R5P they were polymerised from.
PARENTS = {
    "Ala": "Ala", "Asn": "Asn", "Asp": "Asp", "Gln": "Gln", "Glu": "Glu", "Gly": "Gly",
    "His": "His", "Ile": "Ile", "Leu": "Leu", "Lys": "Lys", "Met": "Met", "Phe": "Phe",
    "Pro": "Pro", "Ser": "Ser", "Thr": "Thr", "Trp": "Trp", "Tyr": "Tyr", "Val": "Val",
    "aKG": "AKG", "Cit": "Cit", "Mal": "Mal", "Fum": "Fum", "Suc": "Suc", "Pyr": "Pyr",
    "6PG": "6PG", "FBP": "FBP", "G6P": "G6P", "G6P_dp": "G6P", "F6P_dp": "F6P",
    "S7P_dp": "S7P", "X5P_dp": "X5P", "R5P": "R5P", "Glc": "G6P", "Rib": "R5P",
}

#: Long & Antoniewicz workbook name -> knowns fragment id, where the two differ.
WORKBOOK_NAMES = {"G6P_173": "Glc173", "G6P_370": "Glc370",
                  "R5P_173": "Rib173", "R5P_284": "Rib284"}
WORKBOOK = ROOT / "data/E_coli_GC_MS/EColi_B.xlsx"


def fragment_id(raw):
    """Model id for a knowns ``Fragment ID``, or None if it names no single ion."""
    raw = raw.strip()
    return raw if re.fullmatch(r"[A-Za-z0-9_]+\d[A-Za-z0-9_]*", raw) else None


def carbon_set(raw):
    """1-based carbon positions, or None when the knowns do not pin them down."""
    raw = str(raw).replace(" ", "")
    return [int(c) for c in raw.split(",")] if re.fullmatch(r"\d+(,\d+)*", raw) else None


def plan_fragments(knowns, model):
    """``({fragment_id: (parent, carbons)}, [(knowns id, reason)])``."""
    fragments, skipped = {}, []
    for _, row in knowns.iterrows():
        raw_id = row["Fragment ID"]
        parent = PARENTS.get(row["Abbrev"])
        frag, carbons = fragment_id(raw_id), carbon_set(row["Carbons retained"])
        if parent is None or parent not in model.metabolites:
            reason = f"no core parent for {row['Abbrev']!r}"
        elif frag is None:
            reason = "id names no single ion"
        elif carbons is None:
            reason = f"ambiguous carbons {row['Carbons retained']!r}"
        else:
            member = model.metabolites.get_by_id(parent)
            n = len(member.annotation.get("atom_pattern", ""))
            if carbons[-1] > n:
                reason = f"carbons {carbons} exceed {parent}'s {n}"
            elif member.annotation.get("symmetries") and len(carbons) < n:
                reason = f"partial fragment of symmetric {parent}"
            else:
                fragments[frag] = (parent, carbons)
                continue
        skipped.append((raw_id, reason))
    return fragments, skipped


def workbook_mapping(knowns, fragments):
    """Mapping rows for every workbook fragment the model carries."""
    names = pd.read_excel(WORKBOOK, sheet_name=None)
    compounds = sorted({c for sheet in names.values() for c in sheet["Compound"].dropna()})
    by_id = knowns.set_index(knowns["Fragment ID"].map(fragment_id))
    rows = []
    for compound in compounds:
        frag = WORKBOOK_NAMES.get(compound, compound)
        if frag not in fragments:
            print(f"  workbook {compound}: not a model fragment, left unmapped")
            continue
        rows.append({"experimental_id": compound, "model_id": frag,
                     "formula": by_id.loc[frag, "Fragment formula"],
                     "n_backbone": len(fragments[frag][1])})
    return pd.DataFrame(rows)


def plan_e_coli_core(model):
    """Fragments from the GC/MS knowns, mapped onto the Long & Antoniewicz workbook."""
    knowns = pd.read_csv(KNOWNS)
    fragments, skipped = plan_fragments(knowns, model)
    print(f"Skipped {len(skipped)} of {len(knowns)} knowns:")
    for raw_id, reason in skipped:
        print(f"  {raw_id}: {reason}")
    return fragments, lambda: workbook_mapping(knowns, fragments)


CONFIGS = [
    dict(name="E_coli_core", plan=plan_e_coli_core,
         model_in="models/atomistic/E_coli_core/e_coli_core_annotated.json",
         model_out="models/atomistic/E_coli_core/e_coli_core_gcms_annotated.json",
         mapping_out="models/atomistic/E_coli_core/E_coli_core_fragment_mapping.csv"),
]


def build(name, plan, model_in, model_out, mapping_out):
    """Add one organism's fragment pools and write its model and mapping CSV."""
    print(f"\n== {name}")
    model = cobra.io.load_json_model(str(ROOT / model_in))
    fragments, mapping_of = plan(model)

    add_fragment_pools(model, fragments)
    cobra.io.save_json_model(model, str(ROOT / model_out))
    print(f"\nAdded {len(fragments)} fragment pools -> {model_out}")

    mapping = mapping_of()
    problems = check_mapping_csv(model, zip(mapping["experimental_id"], mapping["model_id"]))
    if problems or mapping["formula"].isna().any():
        raise ValueError(f"mapping does not match the model: {problems}\n{mapping}")
    mapping.to_csv(ROOT / mapping_out, index=False)
    print(f"Mapped {len(mapping)} workbook fragments -> {mapping_out}")


def main():
    for cfg in CONFIGS:
        build(**cfg)


if __name__ == "__main__":
    main()
