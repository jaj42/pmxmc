from functools import partial

import numpy as np
import pandas as pd

pdtonum = partial(pd.to_numeric, errors="coerce")


def _extract_rate_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a piecewise-constant infusion rate schedule from NONMEM dose records.

    For each dose record (EVID == 1 or 4), creates two entries:
      - rate-ON  at TIME            with the infusion RATE
      - rate-OFF at TIME + AMT/RATE with RATE = 0

    Parameters
    ----------
    df : DataFrame
        Must contain columns ID, TIME, AMT, RATE, EVID.  ID should already
        be the composite occasion-level identifier.

    Returns
    -------
    DataFrame indexed by (ID, TIME) with a single RATE column, sorted.

    Notes
    -----
    Assumes infusions within the same occasion do not overlap temporally.
    If they did, rates would need to be summed rather than set.
    """
    doses = df[df["EVID"] != 0].copy()
    doses["TINF"] = doses.eval("AMT / RATE")
    doses["TEND"] = doses.eval("TIME + TINF")

    rows = []
    for _, row in doses.iterrows():
        subjectid = int(row["ID"])
        # Infusion turns on
        rows.append({"ID": subjectid, "TIME": row["TIME"], "RATE": row["RATE"]})
        # Infusion turns off
        rows.append({"ID": subjectid, "TIME": row["TEND"], "RATE": 0.0})

    result = pd.DataFrame(rows)
    result = result.sort_values(["ID", "TIME"]).reset_index(drop=True)

    # If an end-time coincides exactly with the next start-time, keep the
    # start (last entry at that time) so the new infusion takes effect.
    result = result.drop_duplicates(subset=["ID", "TIME"], keep="last")

    return result.set_index(["ID", "TIME"])


def read_dataset(
    filepath: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Read a NONMEM-style CSV and return data structures for Bayesian fitting.

    Occasion handling
    -----------------
    Each EVID=4 record marks a compartment reset **and** a new dose — i.e.
    the start of a new occasion.  Occasions are split into independent units
    with a composite ID = SUBJID * 100 + OCC (OCC = 1, 2, …).

    Returns
    -------
    rate : DataFrame
        Piecewise-constant infusion schedule indexed by (ID, TIME).
    dv : Series
        Observed concentrations (EVID==0 only) indexed by (ID, TIME).
    covar : DataFrame
        Per-biological-subject covariates indexed by SUBJID.
    bio_map : Series
        Mapping from composite occasion-ID → biological subject ID.
    """
    # ---- read & coerce ----
    df = pd.read_csv(filepath)
    df = df.rename(columns={"@ID": "ID"})
    df = df.apply(pdtonum)

    # ---- assign occasion numbers ----
    # EVID=4 (reset + dose) marks the beginning of each occasion.
    # cumsum of the indicator gives 1 for the first occasion, 2 for the
    # second, etc.
    df["OCC"] = df.groupby("ID")["EVID"].transform(lambda s: s.eq(4).cumsum())

    # Composite ID that uniquely identifies each (subject, occasion)
    df["SUBJID"] = df["ID"].astype(int)
    df["ID"] = df["SUBJID"] * 100 + df["OCC"].astype(int)

    # Sort within each occasion — times should now be monotonic
    df = df.sort_values(["ID", "TIME"]).reset_index(drop=True)

    # ---- sanity check: monotonic times within each occasion ----
    for cid, grp in df.groupby("ID"):
        times = grp["TIME"].values
        if not np.all(np.diff(times) >= 0):
            raise ValueError(
                f"Non-monotonic TIME within composite ID {cid}. "
                "Check occasion splitting logic."
            )

    # ---- rate schedule ----
    rate = _extract_rate_schedule(df)

    # ---- observations (EVID == 0 only) ----
    obs = df[df["EVID"] == 0].copy()
    # The concentration column may be called "DV" or "CP" depending on
    # the dataset version.
    if "CP" in obs.columns:
        dv_col = "CP"
    elif "DV" in obs.columns:
        dv_col = "DV"
    else:
        raise KeyError("Dataset has neither a 'DV' nor a 'CP' column.")
    dv = obs.set_index(["ID", "TIME"])[dv_col]

    # ---- covariates (one row per biological subject) ----
    covar = (
        df.drop_duplicates("SUBJID")
        .set_index("SUBJID")[["AGE", "WT", "HT", "M1F2"]]
        .sort_index()
    )

    # ---- mapping: composite occasion-ID → biological subject ID ----
    bio_map = df.drop_duplicates("ID").set_index("ID")["SUBJID"].sort_index()
    # print(bio_map)
    # quit()

    return rate, dv, covar, bio_map
