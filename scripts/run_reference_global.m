% Baseline: one model year of the full NUM setup on MITgcm_2.8deg with the
% unmodified upstream code (Matlab parfor + per-box Fortran). Saves timing and
% the monthly sim struct as the reference for the vectorized/GPU port.
here = fileparts(mfilename('fullpath'));
cd(fullfile(here, '..', 'NUMmodel', 'matlab'));

% Start the pool first: setupNUMmodel(bParallel=true) loads the library on its workers.
nWorkers = str2double(getenv('NUM_WORKERS'));
if isnan(nWorkers), nWorkers = 6; end
delete(gcp('nocreate'));
pool = parpool('Processes', nWorkers);
p = setupNUMmodel(bParallel=true);
p = parametersGlobal(p);          % MITgcm_2.8deg, dt = 0.1 d, dtTransport = 0.5 d
p.tEnd = 365;

t0 = tic;
sim = simulateGlobal(p);
wall = toc(t0);

info.wall_s = wall;
info.nWorkers = pool.NumWorkers;
info.nCores = feature('numcores');
info.matlab = version;
info.date = char(datetime('now'));
info.numCommit = strtrim(evalc('!git -C .. describe --tags --always'));
fprintf('\nWALL CLOCK one model year: %.1f s (%.1f min) on %d workers\n', wall, wall/60, info.nWorkers);
checkConservation(sim);
save(fullfile(here, '..', 'reference', 'global_2p8_1yr_setupNUMmodel.mat'), 'sim', 'info', '-v7.3');
