"""
BiGG-id -> real compound name fallback, for organisms whose GEM xlsx carries no
metabolite names at all.

iAF1260 (E_coli) and its core subset (E_coli_core) ship a `Metabolites` sheet whose
`metname` column is entirely empty, so `sigmaflux.utils.lcms_knowns.match_lcms_knowns`
has nothing to match against `data/HILIC_LCMS_knowns.csv` by name. These two tables
fill that gap for the metabolites this repo's models actually carry, each value
spelled exactly as it appears in `data/HILIC_LCMS_knowns.csv`'s `compound` column so
`normalize_compound` still resolves it. They are separate because the two organisms
use different id vocabularies: iAF1260 follows BiGG abbreviations (`accoa`, `glc-D`,
keyed here with the compartment suffix already stripped), while E_coli_core uses its
own short-form ids (`AcCoA`, `Gluc_e`) that carry no compartment suffix and are
looked up verbatim.

A metabolite id absent from its table simply has no fallback name and continues to
go unmatched, same as before this module existed -- entries were added only after
confirming both that the id is real (present in the organism's GEM xlsx) and that
the name is real (present verbatim in data/HILIC_LCMS_knowns.csv); nothing here is
guessed.
"""
import re

E_COLI_BIGG_NAMES: dict[str, str] = {
    "3pg": "3-Phosphoglyceric acid",
    "adp": "ADP",
    "aicar": "AICAR",
    "accoa": "Acetyl-CoA",
    "cdp": "CDP",
    "cit": "Citric acid",
    "citr-L": "Citrulline",
    "coa": "Coenzyme A",
    "e4p": "D-Erythrose 4-phosphate",
    "glc-D": "D-Glucose",
    "g3p": "D-Glyceraldehyde 3-phosphate",
    "r5p": "D-Ribose 5-phosphate",
    "dhap": "Dihydroxyacetone phosphate",
    "fdp": "Fructose 1,6-bisphosphate",
    "f6p": "Fructose 6-phosphate",
    "fum": "Fumaric acid",
    "g6p": "Glucose 6-phosphate",
    "glyc3p": "Glycerol 3-phosphate",
    "gly": "Glycine",
    "ala-L": "L-Alanine",
    "arg-L": "L-Arginine",
    "asn-L": "L-Asparagine",
    "asp-L": "L-Aspartic acid",
    "cys-L": "L-Cysteine",
    "glu-L": "L-Glutamic acid",
    "gln-L": "L-Glutamine",
    "his-L": "L-Histidine",
    "ile-L": "L-Isoleucine",
    "leu-L": "L-Leucine",
    "lys-L": "L-Lysine",
    "met-L": "L-Methionine",
    "phe-L": "L-Phenylalanine",
    "pro-L": "L-Proline",
    "ser-L": "L-Serine",
    "thr-L": "L-Threonine",
    "trp-L": "L-Tryptophan",
    "tyr-L": "L-Tyrosine",
    "val-L": "L-Valine",
    "nad": "NAD",
    "nadh": "NADH",
    "nadp": "NADP",
    "nadph": "NADPH",
    "orn": "Ornithine",
    "orot": "Orotic acid",
    "pep": "Phosphoenolpyruvic acid",
    "pyr": "Pyruvic acid",
    "succ": "Succinic acid",
    "dcdp": "dCDP",
    "dctp": "dCTP",
    "dgdp": "dGDP",
    "dtdp": "dTDP",
    "dump": "dUMP",
    # Verified additions beyond the hand-curated seed: id confirmed present in
    # iAF1260_GEM.xlsx's metid, name confirmed verbatim in HILIC_LCMS_knowns.csv.
    "acon-C": "cis-Aconitic acid",
    "actp": "Acetylphosphate",
    "ade": "Adenine",
    "ahcys": "S-Adenosylhomocysteine",
    "amet": "S-Adenosylmethionine",
    "amp": "Adenosine monophosphate",
    "atp": "Adenosine triphosphate",
    "cmp": "Cytidine monophosphate",
    "ctp": "Cytidine triphosphate",
    "cytd": "Cytidine",
    "dad-2": "Deoxyadenosine",
    "dhor-S": "4,5-Dihydroorotic acid",
    "din": "Deoxyinosine",
    "dpcoa": "Dephospho-CoA",
    "fmn": "Flavin Mononucleotide",
    "gam6p": "Glucosamine 6-phosphate",
    "gdp": "Guanosine diphosphate",
    "glx": "Glyoxylic acid",
    "glyc": "Glycerol",
    "glyclt": "Glycolic acid",
    "gmp": "Guanosine monophosphate",
    "gthrd": "Glutathione",
    "gtp": "Guanosine triphosphate",
    "gua": "Guanine",
    "hcys-L": "Homocysteine",
    "hxan": "Hypoxanthine",
    "imp": "Inosinic acid",
    "ins": "Inosine",
    "ipdp": "Isopentenyl pyrophosphate",
    "malcoa": "Malonyl-CoA",
    "pnto-R": "Pantothenic acid",
    "ppcoa": "Propionyl-CoA",
    "pydx5p": "Pyridoxal 5'-phosphate",
    "ribflv": "Riboflavin",
    "skm": "Shikimic acid",
    "spmd": "Spermine",
    "thmmp": "Thiamine monophosphate",
    "thmpp": "Thiamine pyrophosphate",
    "uacgam": "Uridine diphosphate-N-acetylglucosamine",
    "udp": "Uridine 5'-diphosphate",
    "udpg": "Uridine diphosphate glucose",
    "ump": "Uridine 5'-monophosphate",
    "ura": "Uracil",
    "uri": "Uridine",
    "utp": "Uridine triphosphate",
    "xan": "Xanthine",
}

E_COLI_CORE_NAMES: dict[str, str] = {
    "3PG": "3-Phosphoglyceric acid",
    "Cit": "Citric acid",
    "E4P": "D-Erythrose 4-phosphate",
    "GAP": "D-Glyceraldehyde 3-phosphate",
    "R5P": "D-Ribose 5-phosphate",
    "DHAP": "Dihydroxyacetone phosphate",
    "FBP": "Fructose 1,6-bisphosphate",
    "F6P": "Fructose 6-phosphate",
    "Fum": "Fumaric acid",
    "G6P": "Glucose 6-phosphate",
    "Gly": "Glycine",
    "Ala": "L-Alanine",
    "Arg": "L-Arginine",
    "Asp": "L-Aspartic acid",
    "Glu": "L-Glutamic acid",
    "Gln": "L-Glutamine",
    "His": "L-Histidine",
    "Ile": "L-Isoleucine",
    "Leu": "L-Leucine",
    "Lys": "L-Lysine",
    "Met": "L-Methionine",
    "Phe": "L-Phenylalanine",
    "Pro": "L-Proline",
    "Ser": "L-Serine",
    "Thr": "L-Threonine",
    "Trp": "L-Tryptophan",
    "Tyr": "L-Tyrosine",
    "Val": "L-Valine",
    "PEP": "Phosphoenolpyruvic acid",
    "Pyr": "Pyruvic acid",
    "Suc": "Succinic acid",
    # Verified additions beyond the hand-curated seed.
    "6PG": "6-Phosphogluconic acid",
    "AKG": "Oxoglutaric acid",
    "ATP": "Adenosine triphosphate",
    "AcCoA": "Acetyl-CoA",
    "Asn": "L-Asparagine",
    "Cys": "L-Cysteine",
    "Glx": "Glyoxylic acid",
    "Gluc_e": "D-Glucose",
    "Gluc_pre": "D-Glucose",
    "KDPG": "2-Keto-3-deoxy-D-gluconic acid",
    "LLDAP": "Diaminopimelic acid",
    "NADH": "NADH",
    "NADPH": "NADPH",
    "Mal": "L-Malic acid",
    "S7P": "Sedoheptulose 7-phosphate",
}

_COMPARTMENT_SUFFIX = re.compile(r'_[a-z]$')


def fill_missing_names(metabolites_df, id_col, name_col, table, strip_compartment):
    """Fill empty *name_col* cells of *metabolites_df* from *table*, keyed by *id_col*.

    Rows whose id has no entry in *table* are left untouched (still empty), which
    leaves them unmatched downstream exactly as before this fallback existed.

    Args:
        metabolites_df: DataFrame with *id_col*/*name_col* columns to patch.
        id_col: column holding the model metabolite id.
        name_col: column to fill in where empty.
        table: model id -> compound name.
        strip_compartment: strip a trailing `_x` compartment suffix off *id_col*
            before the lookup (True for BiGG-style ids, False for ids that already
            carry no compartment suffix, e.g. E_coli_core's).
    """
    def _lookup(met_id):
        key = _COMPARTMENT_SUFFIX.sub('', met_id) if strip_compartment else met_id
        return table.get(key)

    # name_col starts all-NaN (float64) in these GEMs; cast to object before writing
    # strings into it, or pandas warns it will refuse the dtype change outright.
    metabolites_df[name_col] = metabolites_df[name_col].astype(object)
    missing = metabolites_df[name_col].isna()
    metabolites_df.loc[missing, name_col] = metabolites_df.loc[missing, id_col].map(_lookup)
    return metabolites_df
