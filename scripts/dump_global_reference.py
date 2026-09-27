"""Dump upstream simulateGlobal internals for julia/NUMGPU/test/global_steps.jl.

Writes a copy of NUMmodel/matlab/simulateGlobal.m with save() calls added (after the
BCvalue setup and after each of the first 3 transport steps; nothing else changed), then
runs it serially in Matlab on MITgcm_2.8deg (~1 min).

Usage: python scripts/dump_global_reference.py <outdir>   ->  <outdir>/matlab_dump.mat
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATLAB_DIR = os.path.join(ROOT, "NUMmodel", "matlab")


def main(outdir):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    dump = os.path.join(outdir, "matlab_dump.mat")
    s = open(os.path.join(MATLAB_DIR, "simulateGlobal.m")).read()
    s = s.replace("function sim = simulateGlobal(p, sim, options)",
                  "function sim = simulateGlobalDump(p, sim, options)")
    after_bc = "        BCvalue(:,i) = u(ixBottom,i)';\n    end\nend\n"
    after_tm = "    if p.bTransport\n        u =  Aimp*(Aexp*u);\n    end\n"
    assert after_bc in s and after_tm in s, "simulateGlobal.m changed; update the patch points"
    s = s.replace(after_bc, after_bc +
                  f"U_init = u; save('{dump}','U_init','L0','Tmat','ixBottom','dzBottom',"
                  "'BCvalue','Asink','idxSinking','-v7.3');\n")
    # simulateGlobal has a nested function (static workspace), so save through a struct
    s = s.replace(after_tm, after_tm + f"""    DS = struct(); DS.(sprintf('U_step%d', i)) = u;
    if i == 1, DS.Aexp_m1 = Aexp; DS.Aimp_m1 = Aimp; DS.T_m1 = T; end
    save('{dump}', '-struct', 'DS', '-append');
    if i == 3, sim = []; return; end
""")
    open(os.path.join(outdir, "simulateGlobalDump.m"), "w").write(s)
    open(os.path.join(outdir, "run_dump.m"), "w").write(f"""addpath('{outdir}');
addpath('{MATLAB_DIR}/Transport matrix');
cd('{MATLAB_DIR}');
p = setupNUMmodel(bParallel=false);
p = parametersGlobal(p);
velocity = calllib(loadNUMmodelLibrary(), 'f_getsinking', 0*p.m); u0 = p.u0;
simulateGlobalDump(p);
save('{dump}','velocity','u0','-append');
""")
    subprocess.run(["matlab", "-batch", "run('run_dump.m')"], cwd=outdir, check=True)
    print(dump)


if __name__ == "__main__":
    main(sys.argv[1])
