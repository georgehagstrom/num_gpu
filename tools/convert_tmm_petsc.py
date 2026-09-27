"""Convert a PETSc-format TMM example (e.g. samarkhatiwala/tmm examples/ironmops, MITgcm_2.8deg)
into the .mat layout that NUMmodel's parametersGlobal/simulateGlobal expect.

The PETSc files are already discretized for dt = 43200 s and stored in profile order:
    Ae = I + dt*Aexp,   Ai = Aimp^(dt/deltaT)
We write Aexp = (Ae - I)/dt (NUM rebuilds Ae exactly) and Aimp = Ai, and store deltaT = dt in
grid.mat so NUM's Aimp^(dtTransport*86400/deltaT) becomes Ai^1. No change to NUM code is needed.

Usage: python convert_tmm_petsc.py <ironmops dir> <NUMmodel/TMs/MITgcm_2.8deg>
"""
import os
import sys

import h5py
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

DT = 43200.0  # time step the PETSc matrices were built with (TMM make_input_files: dt=43200)


def read_petsc_mat(f):
    with open(f, 'rb') as fh:
        cid, M, N, nnz = np.fromfile(fh, '>i4', 4)
        assert cid == 1211216, f'{f}: not a PETSc Mat'
        rn = np.fromfile(fh, '>i4', M)
        ci = np.fromfile(fh, '>i4', nnz)
        v = np.fromfile(fh, '>f8', nnz)
    ip = np.concatenate([[0], np.cumsum(rn)])
    return sp.csr_matrix((v.astype(float), ci, ip), shape=(M, N))


def read_petsc_vec(f):
    with open(f, 'rb') as fh:
        cid, n = np.fromfile(fh, '>i4', 2)
        assert cid == 1211214, f'{f}: not a PETSc Vec'
        return np.fromfile(fh, '>f8', n).astype(float)


def h5_to_dict(f):
    out = {}
    with h5py.File(f) as h:
        for k, v in h.items():
            x = v[()]
            if isinstance(x, bytes):
                x = x.decode()
            out[k] = x
    return out


def profile_permutation(boxes, src):
    """perm[p] = natural (boxes.mat) index of the box at profile-order position p."""
    ix, iy, iz = boxes['ixBox'], boxes['iyBox'], boxes['izBox']
    cols = {}
    for i in range(len(ix)):
        cols.setdefault((ix[i], iy[i]), []).append(i)
    surf = np.where(iz == 1)[0]
    perm = np.concatenate([sorted(cols[(ix[s], iy[s])], key=lambda j: iz[j]) for s in surf])
    # Must reproduce the profile lengths the TMM driver used
    gs = np.fromfile(os.path.join(src, 'gStartIndices.bin'), '>i4')[1:]
    ge = np.fromfile(os.path.join(src, 'gEndIndices.bin'), '>i4')[1:]
    lens = np.array([len(cols[(ix[s], iy[s])]) for s in surf])
    assert np.array_equal(lens, ge - gs + 1), 'profile structure does not match boxes.h5'
    assert np.array_equal(np.sort(perm), np.arange(len(ix)))
    return perm


def to_natural_mat(A, perm):
    # A_nat[perm[p], perm[q]] = A_prof[p, q]
    A = A.tocoo()
    n = A.shape[0]
    return sp.csc_matrix((A.data, (perm[A.row], perm[A.col])), shape=(n, n))


def main(src, dst):
    grid = h5_to_dict(os.path.join(src, 'grid.h5'))
    boxes = h5_to_dict(os.path.join(src, 'boxes.h5'))
    nb = int(boxes['nb'])
    perm = profile_permutation(boxes, src)
    I = sp.identity(nb, format='csc')

    os.makedirs(os.path.join(dst, 'Matrix5', 'TMs'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Matrix5', 'Data'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'BiogeochemData'), exist_ok=True)

    # Index arrays stay 1-based, as in the original Matlab files
    sio.savemat(os.path.join(dst, 'Matrix5', 'Data', 'boxes.mat'),
                {k: (float(v) if k == 'nb' else np.asarray(v, float).reshape(-1, 1))
                 for k, v in boxes.items()}, do_compression=True)

    # Matlab code expects 1-D grid arrays (x, y, z, dznom, ...) as column vectors
    grid_out = {k: (v.reshape(-1, 1) if isinstance(v, np.ndarray) and v.ndim == 1 else v)
                for k, v in grid.items()}
    grid_out['deltaT_gcm'] = grid['deltaT']
    grid_out['deltaT'] = DT  # makes NUM's Aimp^(43200/deltaT) = Ai^1; see README
    sio.savemat(os.path.join(dst, 'grid.mat'), grid_out, do_compression=True)

    # parametersGlobal only checks that config_data.mat exists
    sio.savemat(os.path.join(dst, 'config_data.mat'),
                {'note': 'placeholder; converted from PETSc TMM example, see README_converted.md'})

    nx, ny, nz = int(grid['nx']), int(grid['ny']), int(grid['nz'])
    ix, iy, iz = (boxes[k].astype(int) - 1 for k in ('ixBox', 'iyBox', 'izBox'))
    Tbc = np.full((nx, ny, nz, 12), np.nan)
    for m in range(12):
        T = np.empty(nb)
        T[perm] = read_petsc_vec(os.path.join(src, f'Ts_{m:02d}'))
        Tbc[ix, iy, iz, m] = T
    sio.savemat(os.path.join(dst, 'BiogeochemData', 'Theta_bc.mat'), {'Tbc': Tbc}, do_compression=True)

    for m in range(12):
        Ae = to_natural_mat(read_petsc_mat(os.path.join(src, f'Ae_{m:02d}')), perm)
        Ai = to_natural_mat(read_petsc_mat(os.path.join(src, f'Ai_{m:02d}')), perm)
        Aexp = ((Ae - I) / DT).tocsc()
        Aexp.eliminate_zeros()
        err = abs((I + DT * Aexp) - Ae).max()
        sio.savemat(os.path.join(dst, 'Matrix5', 'TMs', f'matrix_nocorrection_{m + 1:02d}.mat'),
                    {'Aexp': Aexp, 'Aimp': Ai}, do_compression=True)
        print(f'month {m + 1:02d}: nnz Aexp {Aexp.nnz}, Aimp {Ai.nnz}, '
              f'max|I+dt*Aexp - Ae| = {err:.1e}, min offdiag Aimp = {(Ai - sp.diags(Ai.diagonal())).min():.1e}')

    with open(os.path.join(dst, 'README_converted.md'), 'w') as fh:
        fh.write(f"""# Converted TMM input (not the official MITgcm_2.8deg download)

Source: samarkhatiwala/tmm examples/ironmops (PETSc format), converted by
tools/convert_tmm_petsc.py in the num_model_gpu repo.

- Matrices were pre-discretized with dt = {DT:.0f} s and stored in profile order; they are
  permuted back to boxes.mat order.
- Aexp = (Ae - I)/dt, so NUM's `Ix + dt*Aexp` reproduces Ae to round-off.
- Aimp = Ai is ALREADY Aimp_raw^{int(DT / grid['deltaT'])}. grid.mat therefore stores
  deltaT = {DT:.0f} (the real GCM step, {grid['deltaT']:.0f} s, is in deltaT_gcm) so NUM's
  `Aimp^(43200/deltaT)` is Ai^1. Do not use this grid.mat for anything else that needs deltaT.
- NUM applies function_convert_TM_positive before discretizing. For Aexp this works as upstream
  (the recovered Aexp has ~1.4e6 negative off-diagonals, which NUM moves); for Aimp it is applied after the power (Ai has no negative off-diagonals, so it is a no-op).
- Theta_bc.mat holds the GCM temperature (Theta_gcm, from Ts_00..Ts_11), not the original
  Theta_bc. Months: Ts_00/Ae_00 -> January.
- Whether the matrices were built from matrix_nocorrection_* or corrected matrices is not recorded.
""")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
