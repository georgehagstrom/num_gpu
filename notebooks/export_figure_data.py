"""Reduce a spin-up output (scripts/spinup_global.jl) to tidy CSV files for notebooks/run_figures.qmd.

Usage: python notebooks/export_figure_data.py <run.mat> <grid name, e.g. MITgcm_2.8deg> <outdir>

Writes <outdir>/cells.csv (one row per horizontal grid cell: cell edges, ocean flag, final-year NPP,
NPP of the reference year, pico/nano/micro fractions and size-spectrum exponent of the 0-120 m
community), timeseries.csv (per model year: global NPP, total N, biomass per group, wall clock),
spectrum.csv (area-weighted global 0-120 m community spectrum, split into unicellulars and
copepods) and meta.json. Same definitions as notebooks/make_century_notebook.py.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

run, gridname, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]


def loadmat_any(path):
    # scipy for older .mat files, h5py for Matlab v7.3 (HDF5; arrays transposed back to Matlab order)
    try:
        return loadmat(path)
    except NotImplementedError:
        with h5py.File(path, "r") as h:
            return {k: np.array(h[k]).T for k in h.keys() if isinstance(h[k], h5py.Dataset)}


with h5py.File(run, "r") as h:
    r = {k: np.array(h[k]) for k in h.keys() if isinstance(h[k], h5py.Dataset) and k != "U_month"}
    gnames = ["".join(chr(c) for c in col).strip() for col in np.array(h["group_names"]).T]
Umean = r["U_mean"].T
NPPbox = r["NPP_boxyear"].T                       # boxes x years, mgC/m3/yr
ix, iy, iz = (r[k].ravel().astype(int) - 1 for k in ("ixBox", "iyBox", "izBox"))
vol, years = r["vol"].ravel(), int(np.array(r["years"]).ravel()[0])
grid = loadmat_any(ROOT / "NUMmodel/TMs" / gridname / "grid.mat")
x, y, dz = grid["x"].ravel(), grid["y"].ravel(), grid["dznom"].ravel()
nx, ny = len(x), len(y)
classes = json.loads((ROOT / "reference/num_classes.json").read_text())
m = np.array([c["m"] for c in classes]); mLower = np.array([c["mLower"] for c in classes])
mDelta = np.array([c["mDelta"] for c in classes]); ctype = np.array([c["type"] for c in classes])
cgroup = np.array([c["group"] for c in classes])
nN = 3
B = Umean[:, nN:]

ocean = np.zeros((nx, ny), bool); ocean[ix[iz == 0], iy[iz == 0]] = True


def integ(v, sel=None):                            # depth integral per column
    sel = np.ones(len(ix), bool) if sel is None else sel
    G = np.zeros((nx, ny)); np.add.at(G, (ix[sel], iy[sel]), (v * dz[iz])[sel])
    return np.where(ocean, G, np.nan)


# NPP maps (g C m-2 yr-1): final year and reference year (10, or 1 for short runs)
yref = 10 if years > 10 else 1
npp_final = integ(NPPbox[:, -1]) / 1000
npp_ref = integ(NPPbox[:, yref - 1]) / 1000

# size structure of the 0-120 m unicellular community and the community spectrum exponent
up = iz < 2
prot = np.isin(ctype, ["Generalists", "Diatoms"]); cope = np.isin(ctype, ["Passive copepods", "Active copepods"])
esd = 1e4 * 1.5 * (m * 1e-6) ** (1 / 3)
cls = {"pico": prot & (esd <= 2), "nano": prot & (esd > 2) & (esd <= 20), "micro": prot & (esd > 20)}
parts = np.stack([integ(B[:, s].sum(1), up) for s in cls.values()], axis=-1)
frac = parts / parts.sum(-1, keepdims=True)
Bup = np.zeros((nx, ny, B.shape[1])); np.add.at(Bup, (ix[up], iy[up]), B[up] * dz[iz[up], None]); Bup /= dz[:2].sum()
mc = np.logspace(np.log10(m.min()), np.log10(m[ctype != "POM"].max()), 100)


def community_spectrum(Bcols):
    Bc = np.where(Bcols > 0, Bcols, 1e-100)
    BSh = np.zeros(Bc.shape[:-1] + (len(mc),))
    for gi in np.unique(cgroup):
        sel = cgroup == gi
        if ctype[sel][0] == "POM":
            continue
        logk = np.log(Bc[..., sel] / np.log((mLower[sel] + mDelta[sel]) / mLower[sel]))
        lm = np.log(m[sel]); inside = (np.log(mc) >= lm[0]) & (np.log(mc) <= lm[-1])
        BSh[..., inside] += np.exp(np.apply_along_axis(lambda v: np.interp(np.log(mc[inside]), lm, v), -1, logk))
    dff = np.diff(np.log(mc))
    mU = np.exp(np.log(mc) + 0.5 * np.append(dff, dff[-1])); mL = np.exp(np.log(mc) - 0.5 * np.insert(dff, 0, dff[0]))
    return BSh / (mU - mL) * (mc[1] / mc[0])


def fit_exponent(sp):
    ok = sp > 1e-12 * sp.max()
    return np.polyfit(np.log(mc[ok]), np.log(sp[ok]), 1)[0] if ok.sum() > 10 else np.nan


lam = np.full((nx, ny), np.nan)
lam[ocean] = [fit_exponent(sp) for sp in community_spectrum(Bup[ocean])]

# cell edges (regular grids): midpoints between centres, ends extrapolated, latitude clamped
def edges(c):
    e = np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[:-1] + c[1:]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
    return e


xe, ye = edges(x), np.clip(edges(y), -90, 90)
I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
rows = np.column_stack([x[I].ravel(), y[J].ravel(), xe[I].ravel(), xe[I + 1].ravel(), ye[J].ravel(), ye[J + 1].ravel(),
                        ocean.ravel().astype(int), npp_final.ravel(), npp_ref.ravel(),
                        frac[..., 0].ravel(), frac[..., 1].ravel(), frac[..., 2].ravel(), lam.ravel()])
hdr = "lon,lat,lon_w,lon_e,lat_s,lat_n,ocean,npp_final,npp_ref,f_pico,f_nano,f_micro,lambda"
np.savetxt(outdir / "cells.csv", rows, delimiter=",", header=hdr, comments="", fmt="%.6g")

# time series per model year
NPPg = r["NPP_global"].ravel(); N_year = r["N_year"].ravel(); B_year = r["B_year"].T; wall = r["wall_year"].ravel()
ts = np.column_stack([np.arange(1, years + 1), NPPg, N_year / 1e15, wall, B_year / 1e15])
np.savetxt(outdir / "timeseries.csv", ts, delimiter=",", comments="", fmt="%.8g",
           header=",".join(["year", "npp_pgc", "n_total_pgn", "wall_s"] + [f"b_{g}" for g in gnames]))

# global mean 0-120 m spectrum (area weighted)
area = np.zeros((nx, ny)); s0 = iz == 0; area[ix[s0], iy[s0]] = vol[s0] / dz[0]
Bglob = (Bup[ocean] * area[ocean][:, None]).sum(0) / area[ocean].sum()
spec = np.column_stack([mc, community_spectrum(np.where(prot, Bglob, 0)[None, :])[0],
                        community_spectrum(np.where(cope, Bglob, 0)[None, :])[0], community_spectrum(Bglob[None, :])[0]])
np.savetxt(outdir / "spectrum.csv", spec, delimiter=",", comments="", fmt="%.6g",
           header="mass_ugc,unicellulars,copepods,community")

meta = {"run": run.name, "grid": gridname, "years": years, "yref": yref, "boxes": int(Umean.shape[0]),
        "groups": gnames, "setup_s": float(np.array(r["setup_s"]).ravel()[0]),
        "lambda_global": float(fit_exponent(spec[:, 3])), "dx": float(xe[1] - xe[0])}
(outdir / "meta.json").write_text(json.dumps(meta, indent=1))
print(outdir, meta)
