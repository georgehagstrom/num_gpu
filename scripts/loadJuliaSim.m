% Load a global run written by NUMGPU.save_sim_mat and complete it like simulateGlobal:
% attaches p (setupNUMmodel + parametersGlobal, MITgcm_2.8deg) and Ntot, so the upstream
% plot scripts (plotGlobal etc.) can be used on it.
%   sim = loadJuliaSim('reference/julia_global_2p8_1yr_sim.mat');
function sim = loadJuliaSim(file)
S = load(file);
sim = S.sim;
here = fileparts(mfilename('fullpath'));
old = cd(fullfile(here, '..', 'NUMmodel', 'matlab'));
cleanup = onCleanup(@() cd(old));
p = setupNUMmodel();
p = parametersGlobal(p);
p.tEnd = sim.t(end);
sim.p = p;
sim.Ntot = calcGlobalN(sim);
end
