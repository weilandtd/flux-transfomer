"""
Build metabolite_pathways.csv for each organism.

For each metabolite in the COBRA JSON model, assigns the most-frequent subsystem
among all reactions that metabolite participates in. Subsystems come from the GEM
Excel file (Reactions sheet, 'subsystem' column), which the JSON models lack.

Usage:
    python models/build_pathway_annotations.py
"""
import collections
import json
import pathlib
import warnings

import pandas as pd

SUBSYSTEM_TO_COARSE = {
    # Central carbon metabolism
    "Glycolysis / Gluconeogenesis":                      "Central carbon metabolism",
    "Pentose phosphate pathway":                         "Central carbon metabolism",
    "Citric acid cycle":                                 "Central carbon metabolism",
    "Pyruvate metabolism":                               "Central carbon metabolism",
    "Acetyl-CoA synthesis":                              "Central carbon metabolism",
    "Glyoxylate metabolism":                             "Central carbon metabolism",
    "Methylglyoxal metabolism":                          "Central carbon metabolism",
    "Oxidative phosphorylation":                         "Central carbon metabolism",
    "Glyc3P-DHAP shuttle":                               "Central carbon metabolism",
    # Amino acid metabolism
    "Alanine, aspartate and glutamate metabolism":       "Amino acid metabolism",
    "Arginine and proline metabolism":                   "Amino acid metabolism",
    "Cysteine and methionine metabolism":                "Amino acid metabolism",
    "Glycine, serine and threonine metabolism":          "Amino acid metabolism",
    "Histidine metabolism":                              "Amino acid metabolism",
    "Lysine metabolism":                                 "Amino acid metabolism",
    "Phenylalanine, tyrosine and tryptophan metabolism": "Amino acid metabolism",
    "Tryptophan metabolism":                             "Amino acid metabolism",
    "Valine, leucine and isoleucine metabolism":         "Amino acid metabolism",
    # Nucleotide metabolism
    "Purine metabolism":                                 "Nucleotide metabolism",
    "Pyrimidine metabolism":                             "Nucleotide metabolism",
    # Carbohydrate metabolism
    "Carbohydrate biosynthesis":                         "Carbohydrate metabolism",
    "Amino sugar and nucleotide sugar metabolism":       "Carbohydrate metabolism",
    # Lipid metabolism
    "Fatty acid biosynthesis":                           "Lipid metabolism",
    "Fatty acid degradation":                            "Lipid metabolism",
    "Fatty acid metabolism":                             "Lipid metabolism",
    "Lipid biosynthesis":                                "Lipid metabolism",
    "Inositol phosphate metabolism":                     "Lipid metabolism",
    # Sterol & isoprenoid metabolism
    "Ergosterol biosynthesis":                           "Sterol & isoprenoid metabolism",
    "Terpenoid backbone biosynthesis":                   "Sterol & isoprenoid metabolism",
    # Cofactor & sulfur metabolism
    "Folate metabolism":                                 "Cofactor & sulfur metabolism",
    "Sulfur metabolism":                                 "Cofactor & sulfur metabolism",
    # Transport
    "Transport":                                         "Transport",
    "Exchange":                                          "Transport",
    "Glycerol secretion":                                "Transport",
    # Other
    "Other":                                             "Other",
    "Detoxification":                                    "Other",
    "Pseudoreaction":                                    "Other",
}

_FALLBACK_COARSE = "Other"

CONFIGS = [
    dict(
        model_json='models/I_orientalis/imIsor_annotated_hierarchical.json',
        gem_xlsx='models/I_orientalis/imIsor_GEM.xlsx',
        out_csv='models/I_orientalis/metabolite_pathways.csv',
    ),
    dict(
        model_json='models/S_cerevisiae/sc_annotated_hierarchical.json',
        gem_xlsx='models/S_cerevisiae/scGEM.xlsx',
        out_csv='models/S_cerevisiae/metabolite_pathways.csv',
    ),
]

_REPO_ROOT = pathlib.Path(__file__).parent.parent


def build_annotations(model_json, gem_xlsx, out_csv):
    model_json = _REPO_ROOT / model_json
    gem_xlsx   = _REPO_ROOT / gem_xlsx
    out_csv    = _REPO_ROOT / out_csv

    # Subsystem lookup from GEM Excel
    rxn_df = pd.read_excel(gem_xlsx, sheet_name='Reactions', usecols=['Rxn ID', 'subsystem'])
    rxn_subsystem = {}
    for _, row in rxn_df.iterrows():
        rxn_id = str(row['Rxn ID']).strip()
        sub    = str(row['subsystem']).strip() if pd.notna(row['subsystem']) else 'Other'
        rxn_subsystem[rxn_id] = sub or 'Other'

    # COBRA model
    with open(model_json) as f:
        model = json.load(f)

    met_name = {m['id']: m.get('name', m['id']) for m in model['metabolites']}

    # metabolite → list of subsystems from all participating reactions
    met_subsystems: dict[str, list[str]] = collections.defaultdict(list)
    for rxn in model['reactions']:
        sub = rxn_subsystem.get(rxn['id'], 'Other')
        for met_id in rxn.get('metabolites', {}):
            met_subsystems[met_id].append(sub)

    rows = []
    for met_id, name in met_name.items():
        subs = met_subsystems.get(met_id, [])
        if subs:
            counter  = collections.Counter(subs)
            max_freq = max(counter.values())
            # majority vote; ties broken alphabetically
            winner = min(s for s, c in counter.items() if c == max_freq)
        else:
            winner = 'Other'
        coarse = SUBSYSTEM_TO_COARSE.get(winner)
        if coarse is None:
            warnings.warn(f"Unknown subsystem {winner!r} — mapping to {_FALLBACK_COARSE!r}")
            coarse = _FALLBACK_COARSE
        rows.append({'metabolite_id': met_id, 'metabolite_name': name,
                     'subsystem': winner, 'coarse_subsystem': coarse})

    out_df = pd.DataFrame(rows).sort_values('metabolite_id')
    out_df.to_csv(out_csv, index=False)

    n_sub    = out_df['subsystem'].nunique()
    n_coarse = out_df['coarse_subsystem'].nunique()
    print(f"  {out_csv.name}: {len(out_df)} metabolites, {n_sub} subsystems → {n_coarse} coarse groups")
    print("  Subsystems:", sorted(out_df['subsystem'].unique()))


if __name__ == '__main__':
    for cfg in CONFIGS:
        print(cfg['out_csv'])
        build_annotations(**cfg)
