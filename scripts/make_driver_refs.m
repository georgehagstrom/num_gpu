% Reference runs of the upstream Matlab drivers for julia/NUMGPU/test/driver_refs.jl.
% Usage: matlab -batch "outdir='/some/dir'; run('scripts/make_driver_refs.m')"
here = fileparts(mfilename('fullpath'));
cd(fullfile(here, '..', 'NUMmodel', 'matlab'));
addpath('Transport matrix');
out = fullfile(outdir, 'driver_refs.mat');
R = struct();
% --- chemostat (ode23s)
p = setupGeneralistsOnly(10); p = parametersChemostat(p); s = simulateChemostat(p, 100, 10);
R.chem1_t = s.t; R.chem1_u = s.u; R.chem1_Cb = s.Cbalance; R.chem1_Nb = s.Nbalance;
p = setupNUMmodel(); p = parametersChemostat(p, seasonalAmplitude=0.5); s = simulateChemostat(p, 100, 10);
R.chem2_t = s.t; R.chem2_u = s.u; R.chem2_jTot = s.rates.jTot; R.chem2_mortpred = s.rates.mortpred;
p = setupGeneralistsDiatoms(10); p = parametersChemostat(p); s = simulateChemostat(p, 60, 15, bUnicellularloss=false);
R.chem3_t = s.t; R.chem3_u = s.u; R.chem3_Nprod = s.Nprod; R.chem3_Nloss = s.Nloss; R.chem3_NlossHTL = s.NlossHTL;
% --- chemostat Euler (Fortran simulateChemostatEuler + getFunctions of the final state)
p = setupNUMmodel(); p = parametersChemostat(p); p.tEnd = 50; s = simulateChemostatEuler(p, 100, 10);
R.cheE_u = s.u; R.cheE_jTot = s.rates.jTot; R.cheE_Cb = s.Cbalance; R.cheE_Nb = s.Nbalance; R.cheE_Sib = s.Sibalance;
R.cheE_ProdGross = s.ProdGross; R.cheE_ProdNet = s.ProdNet; R.cheE_ProdHTL = s.ProdHTL;
% --- insolation
Y = linspace(-89, 89, 50)'; days = [0.5 1 80 172 355 364.5];
R.insol_Y = Y; R.insol_days = days; R.insol = zeros(50, 6);
for j = 1:6, R.insol(:, j) = daily_insolation(0, Y, days(j), 1); end
% --- water columns (2.8 deg)
p = setupNUMmodel(); p = parametersWatercolumn(p, 1); p.tEnd = 60;
s = simulateWatercolumn(p, 50, -30, [], bExtractcolumn=true);
R.wc1_N = s.N; R.wc1_DOC = s.DOC; R.wc1_Si = s.Si; R.wc1_B = s.B; R.wc1_L = s.L; R.wc1_T = s.T; R.wc1_t = s.t;
R.wc1_Ntot = s.Ntot; R.wc1_Nprod = s.Nprod; R.wc1_Nloss = s.Nloss; R.wc1_z = s.z;
p = setupGeneralistsOnly(10); p = parametersWatercolumn(p, 1); p.tEnd = 30; p.bUse_parday_light = false;
s = simulateWatercolumn(p, -20, -120, [], bExtractcolumn=true);
R.wc2_N = s.N; R.wc2_DOC = s.DOC; R.wc2_B = s.B; R.wc2_L = s.L; R.wc2_T = s.T; R.wc2_Nloss = s.Nloss; R.wc2_NlossHTL = s.NlossHTL; R.wc2_Nprod = s.Nprod;
save(out, '-struct', 'R', '-v7.3');
delete('Watercolumn_MITgcm_2.8deg_lat050_lon-30.mat'); delete('Watercolumn_MITgcm_2.8deg_lat-20_lon-120.mat');
disp('chemostat + watercolumn refs done');
% --- global, 30 days, annual averages
delete(gcp('nocreate')); parpool('Processes', 4);
p = setupNUMmodel(bParallel=true); p = parametersGlobal(p); p.tEnd = 30;
s = simulateGlobal(p, bCalcAnnualAverages=true);
G.N = s.N; G.B = s.B; G.ProdNet = s.ProdNet; G.ProdHTL = s.ProdHTL; G.mHTL = s.mHTL;
G.ProdGrossAnnual = s.ProdGrossAnnual; G.ProdNetAnnual = s.ProdNetAnnual; G.ProdHTLAnnual = s.ProdHTLAnnual;
G.BpicoAnnualMean = s.BpicoAnnualMean; G.BnanoAnnualMean = s.BnanoAnnualMean; G.BmicroAnnualMean = s.BmicroAnnualMean;
G.mHTLAnnualMean = s.mHTLAnnualMean; G.BHTL = s.BHTL; G.t = s.t;
G.Ntot = s.Ntot; G.Nloss = s.Nloss; G.NlossHTL = s.NlossHTL; G.Nprod = s.Nprod;
save(fullfile(outdir, 'global_annual_ref.mat'), '-struct', 'G', '-v7.3');
disp('global annual ref done');
