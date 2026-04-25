from functools import partial

import numpy as np
import pandas as pd

pdtonum = partial(pd.to_numeric, errors="coerce")


def _extract_rate_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a piecewise-constant infusion rate schedule from NONMEM dose records.

    For each infusion record (EVID != 0, RATE > 0), creates two entries:
      - rate-ON  at TIME            with the infusion RATE
      - rate-OFF at TIME + AMT/RATE with RATE = 0

    Bolus records (RATE == 0) are excluded; they are returned separately by
    _extract_bolus_schedule.

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
    doses = df[(df["EVID"] != 0) & (df["RATE"] > 0)].copy()
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


def _extract_bolus_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract instantaneous bolus doses (EVID != 0, RATE == 0, AMT > 0).

    Returns
    -------
    DataFrame indexed by (ID, TIME) with a single AMT column, sorted.
    Empty DataFrame with the correct structure if there are no bolus records.
    """
    boluses = df[(df["EVID"] != 0) & (df["RATE"] == 0) & (df["AMT"] > 0)].copy()
    if boluses.empty:
        empty = pd.DataFrame(columns=["AMT"])
        empty.index = pd.MultiIndex.from_tuples([], names=["ID", "TIME"])
        return empty
    result = boluses[["ID", "TIME", "AMT"]].copy()
    result["ID"] = result["ID"].astype(int)
    return result.set_index(["ID", "TIME"]).sort_index()


def read_nonmem_dataset(
    filepath: str,
    covariates: list[str] | None = None,
    sep: str = r"\s+",
    dv_col: str = "DV",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
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
    bolus : DataFrame
        Instantaneous bolus doses indexed by (ID, TIME) with column AMT.
        Empty if the dataset contains no bolus records.
    """
    # ---- read & coerce ----
    df = pd.read_csv(filepath, sep=sep)
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

    # ---- rate schedule & bolus schedule ----
    rate = _extract_rate_schedule(df)
    bolus = _extract_bolus_schedule(df)

    # ---- observations (EVID == 0 only) ----
    obs = df[df["EVID"] == 0].copy()
    # The concentration column may be called "DV" or "CP" depending on
    # the dataset version.
    dv = obs.set_index(["ID", "TIME"])[dv_col]
    dv = dv[~dv.index.duplicated(keep="first")]

    # ---- covariates (one row per biological subject) ----
    if covariates:
        covar = df.drop_duplicates("SUBJID").set_index("SUBJID")[covariates].sort_index()
    else:
        covar = pd.DataFrame([])

    # ---- mapping: composite occasion-ID → biological subject ID ----
    bio_map = df.drop_duplicates("ID").set_index("ID")["SUBJID"].sort_index()

    return rate, dv, covar, bio_map, bolus
