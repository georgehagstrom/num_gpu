% Upstream Matlab + Fortran reference: NUM_YEARS (default 5) years of the full NUM setup on
% MITgcm_2.8deg with a closed bottom boundary for POM (p.BC_POMclosed = true). Saves the state
% at the end of every year (tSave = 365) and the final year's annual averages (ProdNetAnnual,
% Bpico/nano/microAnnualMean), for comparison with the Julia run
% (NUM_POM_CLOSED=1 scripts/spinup_global.jl).
here = fileparts(mfilename('fullpath'));
cd(fullfile(here, '..', 'NUMmodel', 'matlab'));

nWorkers = str2double(getenv('NUM_WORKERS'));
if isnan(nWorkers), nWorkers = 5; end
nYears = str2double(getenv('NUM_YEARS'));
if isnan(nYears), nYears = 5; end
delete(gcp('nocreate'));
pool = parpool('Processes', nWorkers);
p = setupNUMmodel(bParallel=true);
p = parametersGlobal(p);          % MITgcm_2.8deg, dt = 0.1 d, dtTransport = 0.5 d
p.tEnd = 365*nYears;
p.tSave = 365;
p.BC_POMclosed = true;

t0 = tic;
sim = simulateGlobal(p, bCalcAnnualAverages=true);
wall = toc(t0);

info.wall_s = wall;
info.nYears = nYears;
info.nWorkers = pool.NumWorkers;
info.matlab = version;
info.date = char(datetime('now'));
fprintf('\nWALL CLOCK %d model years: %.1f s (%.1f min per year) on %d workers\n', ...
    nYears, wall, wall/60/nYears, info.nWorkers);
checkConservation(sim);
save(fullfile(here, '..', 'reference', sprintf('matlab_closed_%dyr.mat', nYears)), 'sim', 'info', '-v7.3');
