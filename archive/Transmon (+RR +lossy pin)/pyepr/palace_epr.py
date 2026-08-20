"""
palace_epr.py
=============
Run Palace eigenmode results through pyEPR's real ``QuantumAnalysis`` --
the identical Hamiltonian code path your Ansys notebook uses.

Usage (notebook)
----------------
    from palace_epr import load_palace, analyze, print_fancy_results

    eig, Pmj = load_palace('postpro/iteration3')
    res, epra = analyze(eig, Pmj, Ljs=[10e-9], Cjs=[3e-15],
                        modes=[0, 1, 2, 3])
    print_fancy_results(res, names=['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4'])

Usage (CLI)
-----------
    python palace_epr.py --postpro postpro/iteration3 --Lj 10e-9 --Cj 3e-15 \
                         --modes 0 1 2 3

Design notes
------------
pyEPR splits into three classes:

    ProjectInfo         -- Ansys connection            (replaced by Palace)
    DistributedAnalysis -- extract EPRs from fields    (replaced by Palace)
    QuantumAnalysis     -- build/diagonalize H         (KEPT, unmodified)

Palace already writes everything QuantumAnalysis needs, so only the first
two stages are substituted. Every formula (chi_O1, f_1, chi_ND, f_ND, ZPF)
is computed by pyEPR itself -- nothing is re-implemented here.

Two pyEPR 0.9.0 issues are worked around:

  1. The documented ``HamiltonianResultsContainer.save()`` ->
     ``QuantumAnalysis`` route is broken: ``save()`` writes ``.npz`` while
     ``QuantumAnalysis.__init__`` reads ``pickle``. We write the pickle
     directly instead.

  2. The participation renormalization needs Ansys field-energy integrals
     (U_tot_cap, U_tot_ind, U_H, U_E) that Palace does not produce. It must
     be disabled -- which is also correct, since Palace's participations are
     already normalized. Disabling it exposes a latent bug where
     ``Pm_norm`` is returned as a bare int and then indexed; patched here.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

from pyEPR import config
from pyEPR.core_quantum_analysis import QuantumAnalysis

__all__ = [
    "load_palace", "read_palace_eig", "read_palace_epr",
    "validate_junction_ports",
    "analyze", "get_g", "print_fancy_results", "compare_with_ansys",
    "hilbert_size", "identify_modes",
]


# ==========================================================================
# 1. Reading Palace output
# ==========================================================================

def read_palace_eig(path: str) -> pd.DataFrame:
    """Palace ``eig.csv`` -> DataFrame with columns f_GHz, fim_GHz, Q."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    col_re = [c for c in df.columns if "Re{f}" in c][0]
    col_im = [c for c in df.columns if "Im{f}" in c][0]
    col_q = [c for c in df.columns if c.strip() == "Q"][0]
    return pd.DataFrame({
        "f_GHz":   df[col_re].astype(float),
        "fim_GHz": df[col_im].astype(float),
        "Q":       df[col_q].astype(float),
    })


def read_palace_epr(path: str) -> np.ndarray:
    """Palace ``port-EPR.csv`` -> (n_modes, n_junctions) SIGNED participations."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    pcols = sorted((c for c in df.columns if c.startswith("p[")),
                   key=lambda c: int(c.split("[")[1].split("]")[0]))
    if not pcols:
        raise ValueError(f"No p[...] columns found in {path}")
    return df[pcols].to_numpy(dtype=float)


def load_palace(postpro_dir: str):
    """
    Load both Palace CSVs from a postpro directory.

    With AMR there is one subdirectory per refinement pass
    (``iteration1``, ``iteration2``, ...). Point at the highest-numbered
    one -- that is the most converged mesh.

    Returns (eig DataFrame, Pmj array), truncated to a common mode count.
    """
    eig = read_palace_eig(os.path.join(postpro_dir, "eig.csv"))
    Pmj = read_palace_epr(os.path.join(postpro_dir, "port-EPR.csv"))
    n = min(len(eig), Pmj.shape[0])
    if len(eig) != Pmj.shape[0]:
        print(f"note: eig.csv has {len(eig)} modes, port-EPR.csv has "
              f"{Pmj.shape[0]}; using first {n}")
    return eig.iloc[:n].reset_index(drop=True), Pmj[:n]


def validate_junction_ports(postpro_dir: str, Ljs, Cjs=None,
                            palace_config: str | None = None) -> pd.DataFrame:
    """Validate that Palace lumped JJ ports agree with the PyEPR inputs.

    Palace does not infer a Josephson junction from its metal geometry. Every
    JJ must be an inductive ``Boundaries.LumpedPort`` spanning the JJ gap.
    """
    _, pmj = load_palace(postpro_dir)
    Ljs = np.atleast_1d(np.asarray(Ljs, dtype=float))
    Cjs = (np.zeros(len(Ljs), dtype=float) if Cjs is None
           else np.atleast_1d(np.asarray(Cjs, dtype=float)))

    if pmj.shape[1] != len(Ljs):
        raise ValueError(
            f"port-EPR.csv has {pmj.shape[1]} p[j] column(s), but {len(Ljs)} "
            "PyEPR junction inductance value(s) were supplied.")
    if len(Cjs) != len(Ljs):
        raise ValueError("Cjs must have one value per junction inductance.")
    if not np.all(np.isfinite(Ljs)) or np.any(Ljs <= 0):
        raise ValueError("Every Palace/PyEPR junction inductance must be finite and > 0.")

    peak_modes = np.abs(pmj).argmax(axis=0)
    peak_p = np.abs(pmj[peak_modes, np.arange(pmj.shape[1])])
    if np.any(peak_p == 0):
        missing = (np.flatnonzero(peak_p == 0) + 1).tolist()
        raise ValueError(
            f"Junction port(s) {missing} have zero participation in every mode. "
            "Check that each LumpedPort surface bridges the JJ gap and is inductive.")

    report = pd.DataFrame({
        "Palace p[j]": np.arange(1, len(Ljs) + 1),
        "PyEPR Lj (H)": Ljs,
        "PyEPR Cj (F)": Cjs,
        "peak mode (0-based)": peak_modes,
        "max |p|": peak_p,
    })

    if palace_config is not None:
        with open(palace_config, encoding="utf-8") as fh:
            cfg = json.load(fh)
        ports = cfg.get("Boundaries", {}).get("LumpedPort", [])
        inductive = [p for p in ports if float(p.get("L", 0.0)) > 0.0]
        if len(inductive) != len(Ljs):
            raise ValueError(
                f"{palace_config} has {len(inductive)} inductive LumpedPort(s), "
                f"but port-EPR/PyEPR expects {len(Ljs)}.")
        cfg_L = np.asarray([p["L"] for p in inductive], dtype=float)
        cfg_C = np.asarray([p.get("C", 0.0) for p in inductive], dtype=float)
        if not np.allclose(cfg_L, Ljs, rtol=1e-10, atol=0.0):
            raise ValueError(
                f"LumpedPort L values {cfg_L.tolist()} do not match PyEPR Lj "
                f"values {Ljs.tolist()}.")
        if not np.allclose(cfg_C, Cjs, rtol=1e-10, atol=0.0):
            raise ValueError(
                f"LumpedPort C values {cfg_C.tolist()} do not match PyEPR Cj "
                f"values {Cjs.tolist()}.")
        report["mesh attribute"] = [
            p.get("Elements", [{}])[0].get("Attributes", [None])[0]
            for p in inductive
        ]
        report["direction"] = [
            p.get("Elements", [{}])[0].get("Direction", None)
            for p in inductive
        ]
        print(f"JJ port configuration agrees with {palace_config}")

    print("Palace JJ-port / PyEPR mapping:")
    print(report.to_string(index=False))
    return report

# ==========================================================================
# 2. Mode identification and cost estimation
# ==========================================================================

def hilbert_size(n_modes: int, fock_trunc: int):
    """Hilbert dimension and dense Hamiltonian memory (GB) for numerical diag."""
    dim = fock_trunc ** n_modes
    return dim, (dim ** 2) * 16 / 1024 ** 3


def identify_modes(eig: pd.DataFrame, Pmj: np.ndarray,
                   fock_trunc: int = 7, junction: int = 0) -> pd.DataFrame:
    """
    Rank modes by junction participation and print the Hilbert-space cost.

    The qubit mode is the one with |p| ~ 1: nearly all of its inductive
    energy sits in the junction. If |p| is split across two nearly
    degenerate modes, they are hybridised and per-mode quantities become
    order-sensitive -- compare sums over the pair instead.
    """
    p = np.abs(Pmj[:, junction])
    tbl = pd.DataFrame({
        "f_GHz": eig["f_GHz"].to_numpy(),
        "Q":     eig["Q"].to_numpy(),
        "|p|":   p,
    })
    tbl.index.name = "mode"

    order = np.argsort(p)[::-1]
    print("modes ranked by junction participation:")
    for i in order:
        print(f"  index {i}:  f = {tbl['f_GHz'][i]:10.5f} GHz"
              f"   |p| = {p[i]:.6f}   Q = {tbl['Q'][i]:.4e}")
    print(f"\n-> qubit is index {order[0]}  (|p| = {p[order[0]]:.4f})")

    if len(order) > 1 and p[order[1]] > 0.05 * p[order[0]]:
        print(f"   ** index {order[1]} also carries |p| = {p[order[1]]:.4f}"
              f" -- modes may be hybridised; compare sums, not per-mode values")

    print("\nnumerical diagonalization cost (fock_trunc="
          f"{fock_trunc}):")
    for k in range(2, min(len(p), 6) + 1):
        dim, gb = hilbert_size(k, fock_trunc)
        note = "  <-- fine" if gb < 0.5 else ("  <-- heavy" if gb < 2 else
                                              "  <-- will crash")
        print(f"  {k} modes: dim {dim:>10,}   {gb:>10.3f} GB{note}")

    return tbl


# ==========================================================================
# 3. pyEPR 0.9.0 workaround: Pm_norm returned as a scalar
# ==========================================================================

_PATCHED = False


def _patch_pm_norm():
    """
    With renorm_pj=False, pyEPR returns ``Pm_norm = 1`` (a bare int), but
    ``analyze_variation`` then does ``_temp['Pm_norm'][modes]``, raising
    TypeError. Wrap the scalars in Series so indexing works.
    """
    global _PATCHED
    if _PATCHED:
        return
    orig = QuantumAnalysis._get_participation_normalized

    def wrapped(self, variation, _renorm_pj=None, print_=False):
        res = orig(self, variation, _renorm_pj=_renorm_pj, print_=print_)
        idx = res["PJ"].index
        for key in ("Pm_norm", "Pm_cap_norm"):
            if np.isscalar(res[key]):
                res[key] = pd.Series(np.full(len(idx), float(res[key])),
                                     index=idx)
        return res

    QuantumAnalysis._get_participation_normalized = wrapped
    _PATCHED = True


# ==========================================================================
# 4. Build the pickle QuantumAnalysis.__init__ expects
# ==========================================================================

def _build_pickle(path, f_GHz, Qs, Pmj_signed, Ljs, Cjs,
                  mode_names=None, junction_names=None, variation="0"):
    """
    Write ``{'results': {variation: {...}}}`` matching the schema
    ``QuantumAnalysis.__init__`` unpacks.

    Physics-carrying entries come from Palace. The rest (mesh stats,
    convergence history, peak I/V, sols) are inert placeholders: stored on
    the object, never read by ``analyze_variation``.
    """
    f_GHz = np.asarray(f_GHz, dtype=float)
    Qs = np.asarray(Qs, dtype=float)
    Pmj_signed = np.atleast_2d(np.asarray(Pmj_signed, dtype=float))
    Ljs = np.atleast_1d(np.asarray(Ljs, dtype=float))
    M, J = Pmj_signed.shape

    # Cj only feeds Ec -> n_zpf, which analyze_variation computes but never
    # uses for chi. Must be non-zero to avoid divide-by-zero in Ec.
    Cjs = (np.full(J, 1e-15) if Cjs is None
           else np.atleast_1d(np.asarray(Cjs, dtype=float)))
    Cjs = np.where(Cjs <= 0, 1e-15, Cjs)

    if len(Ljs) != J:
        raise ValueError(f"{J} junction column(s) in port-EPR.csv but "
                         f"{len(Ljs)} Lj value(s) given")

    m_lbl = mode_names or [f"mode_{i}" for i in range(M)]
    j_lbl = junction_names or [f"j{j + 1}" for j in range(J)]

    # pyEPR keeps magnitudes in PM and signs separately in SM
    PM = pd.DataFrame(np.abs(Pmj_signed), index=m_lbl, columns=j_lbl)
    SM = pd.DataFrame(np.sign(Pmj_signed), index=m_lbl, columns=j_lbl)
    SM[SM == 0] = 1.0
    zeros = pd.DataFrame(np.zeros((M, J)), index=m_lbl, columns=j_lbl)

    var = {
        # ---- real physics from Palace ----
        "freqs_hfss_GHz": pd.Series(f_GHz, index=m_lbl),
        "Qs":             pd.Series(Qs, index=m_lbl),
        "Om":             pd.Series(f_GHz, index=m_lbl),   # GHz, hbar = 1
        "Pm":             PM,
        "Sm":             SM,
        "Pm_cap":         zeros.copy(),
        "Ljs":            pd.Series(Ljs, index=j_lbl),
        "Cjs":            pd.Series(Cjs, index=j_lbl),
        "modes":          list(range(M)),
        # ---- inert placeholders ----
        "hfss_variables":     pd.Series({"_source": "palace"}),
        "Qm_coupling":        zeros.copy(),
        "I_peak":             zeros.copy(),
        "V_peak":             zeros.copy(),
        "sols":               pd.DataFrame(index=m_lbl),
        "mesh":               pd.DataFrame(),
        "convergence":        pd.DataFrame(),
        "convergence_f_pass": pd.DataFrame(),
        "ansys_energies":     {},
    }

    with open(path, "wb") as fh:
        pickle.dump({"results": {variation: var}}, fh)
    return path


# ==========================================================================
# 5. Main entry point
# ==========================================================================

def analyze(eig, Pmj, Ljs, Cjs=None, modes=None,
            cos_trunc=6, fock_trunc=7,
            mode_names=None, junction_names=None,
            renorm_pj=False, max_gb=2.0, keep_pickle=None):
    """
    Run pyEPR's ``QuantumAnalysis.analyze_variation`` on Palace results.

    Parameters
    ----------
    eig       : DataFrame from ``load_palace`` (or read_palace_eig)
    Pmj       : (M, J) signed participations from ``load_palace``
    Ljs       : junction inductance(s), Henries, e.g. [10e-9]
    Cjs       : junction capacitance(s), Farads, e.g. [3e-15].
                Only feeds Ec/n_zpf, which do not affect chi.
    modes     : 0-based indices to include, e.g. [0, 1, 2, 3].
                REQUIRED for more than ~4 modes: the numerical
                diagonalization scales as fock_trunc**n_modes.
                pyEPR sorts this list ascending internally.
    renorm_pj : leave False for Palace (see module docstring).

    Returns
    -------
    (res, epra)
        res  : dict exactly as analyze_variation returns -- f_0, f_1, f_ND
               (MHz), chi_O1, chi_ND (MHz), ZPF, Pm_normed, Qs, ...
               Indices are normalized to 0..M-1 so downstream pandas
               operations align correctly.
        epra : the live QuantumAnalysis object.
    """
    if isinstance(eig, pd.DataFrame):
        f_GHz, Qs = eig["f_GHz"].to_numpy(), eig["Q"].to_numpy()
    else:                                     # allow (f_GHz, Qs) tuple
        f_GHz, Qs = map(np.asarray, eig)

    Pmj = np.atleast_2d(np.asarray(Pmj, dtype=float))
    n_sel = len(modes) if modes is not None else len(f_GHz)

    dim, gb = hilbert_size(n_sel, fock_trunc)
    if gb > max_gb:
        raise MemoryError(
            f"Numerical diagonalization needs ~{gb:.1f} GB (Hilbert dim "
            f"{dim:,}, {n_sel} modes, fock_trunc={fock_trunc}).\n"
            f"Pass modes=[...] to select fewer modes, or lower fock_trunc.\n"
            f"At fock_trunc=7: 3 modes ~0.002 GB, 4 ~0.09 GB, 5 ~4.5 GB.")

    _patch_pm_norm()
    config.epr.renorm_pj = renorm_pj

    path = keep_pickle or tempfile.mktemp(suffix=".pkl")
    _build_pickle(path, f_GHz, Qs, Pmj, Ljs, Cjs,
                  mode_names=mode_names, junction_names=junction_names)

    epra = QuantumAnalysis(path, do_print_info=False)
    res = epra.analyze_variation(variation="0", cos_trunc=cos_trunc,
                                 fock_trunc=fock_trunc, print_result=False,
                                 modes=modes)

    if keep_pickle is None:
        try:
            os.unlink(path)
        except OSError:
            pass

    # --- verify the Ansys-only renormalization was skipped ---------------
    if not renorm_pj:
        pm_norm = np.asarray(res["_Pm_norm"], dtype=float)
        if not np.allclose(pm_norm, 1.0):
            raise RuntimeError(
                f"_Pm_norm != 1 ({pm_norm}) -- participations were rescaled "
                f"using Ansys energy integrals Palace does not provide.")

    # --- normalize indices -----------------------------------------------
    # pyEPR returns f_0/Qs label-indexed but f_1/f_ND integer-indexed, which
    # breaks pd.concat (union -> 2M rows) and silently NaNs any arithmetic
    # between them.
    M = len(np.asarray(res["f_0"]).ravel())
    for k in ("f_0", "f_1", "f_ND", "Qs"):
        res[k] = pd.Series(np.asarray(res[k], dtype=float).ravel()[:M])
    for k in ("chi_O1", "chi_ND"):
        res[k] = pd.DataFrame(np.asarray(res[k], dtype=float))
    res["ZPF"] = np.asarray(res["ZPF"], dtype=float)

    return res, epra


# ==========================================================================
# 6. Derived quantities and display
# ==========================================================================

def get_g(f_MHz, ZPF, qubit_mode: int) -> np.ndarray:
    """Coupling g from ``qubit_mode`` to every other mode, MHz."""
    Z = np.asarray(ZPF, dtype=float)
    Z = Z[:, 0] if Z.ndim > 1 else Z
    f = np.asarray(f_MHz, dtype=float)
    qf, qz = f[qubit_mode], Z[qubit_mode]
    of = np.delete(f, qubit_mode)
    oz = np.delete(Z, qubit_mode)
    return np.abs((of - qf) * qz * oz / (qz ** 2 + oz ** 2))


def print_fancy_results(res, names=None, qubit_ind=None, display_fn=None):
    """
    Mirror of ``EPR_master.print_fancy_results`` for Palace results.

    In a notebook pass ``display_fn=display`` for styled tables; otherwise
    plain text is printed.
    """
    M = len(res["f_0"])
    names = names or [f"Mode {i + 1}" for i in range(M)]
    if len(names) != M:
        raise ValueError(f"{len(names)} names given for {M} modes")

    if qubit_ind is None:
        qubit_ind = int(np.argmax(np.abs(np.diag(res["chi_ND"].to_numpy()))))

    def show(df, caption, fmt=None):
        if display_fn is not None:
            st = df.style.set_caption(caption)
            if fmt:
                st = st.format(fmt)
            try:
                st = st.background_gradient(axis=None)
            except Exception:
                pass
            display_fn(st)
        else:
            print(f"\n=== {caption} ===")
            print(df.to_string())

    cols = ["Palace (linear)", "Dressed", "Numerical"]

    freqs = pd.DataFrame(
        np.column_stack([res["f_0"].to_numpy(), res["f_1"].to_numpy(),
                         res["f_ND"].to_numpy()]) / 1000.0,
        index=names, columns=cols)
    show(freqs, "Mode frequencies, GHz", "{:.6f}")

    g = get_g(res["f_0"].to_numpy(), res["ZPF"], qubit_ind)
    gnames = [f"{names[qubit_ind]} - {n}"
              for i, n in enumerate(names) if i != qubit_ind]
    show(pd.DataFrame({"g, MHz": g}, index=gnames),
         f"Couplings from {names[qubit_ind]}", "{:.4f}")

    Qs = res["Qs"].to_numpy()
    lossless = bool(np.all(~np.isfinite(Qs)) or np.all(Qs <= 0))
    if lossless:
        print("\nNo lossy elements -> Q, T_1, kappa not calculated")
    else:
        kap = pd.DataFrame(
            np.column_stack([res["f_0"].to_numpy() / Qs,
                             res["f_1"].to_numpy() / Qs,
                             res["f_ND"].to_numpy() / Qs]),
            index=names, columns=cols)
        show(kap * 1e3, "kappa/2pi, kHz", "{:.4f}")
        show(1.0 / (2 * np.pi * kap), "T_1, us", "{:.4f}")

    for key, cap in (("chi_O1", "chi analytical (O1), MHz"),
                     ("chi_ND", "chi numerical (ND), MHz")):
        c = pd.DataFrame(res[key].to_numpy(), index=names, columns=names)
        show(c, cap + "   [diag = anharmonicity, off-diag = cross-Kerr]",
             "{:.4f}")

    if not lossless:
        show(pd.DataFrame({"Q": Qs}, index=names), "Q factors", "{:.4e}")

    p = np.abs(np.asarray(res["Pm_normed"], dtype=float))
    p = p[:, 0] if p.ndim > 1 else p
    show(pd.DataFrame({"|p|": p}, index=names),
         "Junction participation", "{:.6f}")

    return qubit_ind


# ==========================================================================
# 7. Ansys comparison
# ==========================================================================

def compare_with_ansys(res, ansys_f_GHz=None, ansys_Q=None,
                       ansys_chi_O1=None, ansys_chi_ND=None,
                       ansys_p=None, names=None, pair=(0, 1),
                       display_fn=None):
    """
    Compare Palace results against Ansys/pyEPR values.

    Compare like with like: chi_O1 against chi_O1, chi_ND against chi_ND.
    They are different quantities and diverge for hybridised modes.

    ``pair`` selects the (possibly hybridised) modes for the sum-rule test.
    If the qubit is split across two nearly degenerate modes, per-mode
    values are order-sensitive but sums over the pair are not.
    """
    M = len(res["f_0"])
    names = names or [f"Mode {i + 1}" for i in range(M)]

    def show(df, caption, fmt=None):
        if display_fn is not None:
            st = df.style.set_caption(caption)
            if fmt:
                st = st.format(fmt)
            display_fn(st)
        else:
            print(f"\n=== {caption} ===")
            print(df.to_string())

    def diffcol(df, a, b):
        df["diff (%)"] = 100 * (df[a] - df[b]) / df[b]
        return df

    if ansys_f_GHz is not None:
        a = np.asarray(ansys_f_GHz, dtype=float)
        df = pd.DataFrame({"Palace (GHz)": res["f_0"].to_numpy() / 1000.0,
                           "Ansys (GHz)": a}, index=names)
        df["diff (MHz)"] = (df["Palace (GHz)"] - df["Ansys (GHz)"]) * 1e3
        show(diffcol(df, "Palace (GHz)", "Ansys (GHz)"), "Mode frequencies",
             {"Palace (GHz)": "{:.5f}", "Ansys (GHz)": "{:.5f}",
              "diff (MHz)": "{:+.2f}", "diff (%)": "{:+.3f}"})

    if ansys_Q is not None:
        df = pd.DataFrame({"Palace Q": res["Qs"].to_numpy(),
                           "Ansys Q": np.asarray(ansys_Q, dtype=float)},
                          index=names)
        df["ratio (P/A)"] = df["Palace Q"] / df["Ansys Q"]
        show(df, "Q factors", {"Palace Q": "{:.4e}", "Ansys Q": "{:.4e}",
                               "ratio (P/A)": "{:.3f}"})

    for key, ans, lbl in (("chi_O1", ansys_chi_O1, "analytic (O1)"),
                          ("chi_ND", ansys_chi_ND, "numerical (ND)")):
        if ans is None:
            continue
        ans = np.asarray(ans, dtype=float)
        pal = res[key].to_numpy()
        df = pd.DataFrame({"Palace (MHz)": np.diag(pal),
                           "Ansys (MHz)": np.diag(ans)}, index=names)
        show(diffcol(df, "Palace (MHz)", "Ansys (MHz)"),
             f"Anharmonicity, {lbl}",
             {"Palace (MHz)": "{:.4f}", "Ansys (MHz)": "{:.4f}",
              "diff (%)": "{:+.2f}"})
        show(pd.DataFrame(pal - ans, index=names, columns=names),
             f"chi difference {lbl}  (Palace - Ansys), MHz", "{:+.4f}")

    if ansys_p is not None:
        ap = np.asarray(ansys_p, dtype=float).ravel()
        pp = np.abs(np.asarray(res["Pm_normed"], dtype=float))
        pp = (pp[:, 0] if pp.ndim > 1 else pp).ravel()
        df = pd.DataFrame({"Palace |p|": pp, "Ansys |p|": ap}, index=names)
        show(diffcol(df, "Palace |p|", "Ansys |p|"),
             "Junction participation",
             {"Palace |p|": "{:.6f}", "Ansys |p|": "{:.6f}",
              "diff (%)": "{:+.2f}"})

        print("\n" + "=" * 64)
        print("SUM RULES  (invariant to how the qubit splits across modes)")
        print("=" * 64)
        sl_pair = list(pair)
        for lbl, sl in (("hybridised pair " + str(tuple(pair)), sl_pair),
                        ("all modes", slice(None))):
            sa, sp = ap[sl].sum(), pp[sl].sum()
            qa, qp = (ap[sl] ** 2).sum(), (pp[sl] ** 2).sum()
            print(f"{lbl}:")
            print(f"   sum p    Ansys {sa:.6f}  Palace {sp:.6f}"
                  f"   diff {100 * (sp - sa) / sa:+7.3f} %")
            print(f"   sum p^2  Ansys {qa:.6f}  Palace {qp:.6f}"
                  f"   ratio {qp / qa:7.4f}")
        print("\nsum p agreeing but sum p^2 differing => same total junction")
        print("coupling, distributed differently across the pair. Since")
        print("anharmonicity ~ p^2, that redistribution is amplified.")


# ==========================================================================
# 8. CLI
# ==========================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run pyEPR's QuantumAnalysis on Palace eigenmode output.")
    ap.add_argument("--postpro", default="postpro",
                    help="dir with eig.csv and port-EPR.csv "
                         "(use the highest iterationN for AMR runs)")
    ap.add_argument("--Lj", type=float, nargs="+", required=True,
                    help="junction inductance(s), Henries, e.g. 10e-9")
    ap.add_argument("--Cj", type=float, nargs="+", default=None,
                    help="junction capacitance(s), Farads, e.g. 3e-15")
    ap.add_argument("--modes", type=int, nargs="+", default=None,
                    help="0-based mode indices to analyze, e.g. 0 1 2 3")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--cos-trunc", type=int, default=6)
    ap.add_argument("--fock-trunc", type=int, default=7)
    args = ap.parse_args()

    eig, Pmj = load_palace(args.postpro)
    print(f"Loaded {len(eig)} modes from {args.postpro}\n")

    identify_modes(eig, Pmj, fock_trunc=args.fock_trunc)

    res, epra = analyze(eig, Pmj, Ljs=args.Lj, Cjs=args.Cj,
                        modes=args.modes, cos_trunc=args.cos_trunc,
                        fock_trunc=args.fock_trunc)

    sel = args.modes if args.modes is not None else list(range(len(eig)))
    names = args.names or [f"Mode {i + 1}" for i in sorted(sel)]
    print()
    print_fancy_results(res, names=names)