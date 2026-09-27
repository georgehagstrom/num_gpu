"""
Port of the NUM model (NUMmodel, upstream branch Develop at 5644bf5: Fortran core + Matlab
drivers) that runs on CPUs and GPUs.

State layout: `U` is `nb × nGrid` (boxes × state variables: nutrients, then biomass classes).
The biology of each box is computed by one kernel thread (src/kernel.jl) for any of the
upstream setups; the drivers (global transport matrix, water column, chemostat) mirror the
Matlab simulateXXX functions.
"""
module NUMGPU

using LinearAlgebra
using StaticArrays
using SpecialFunctions: erf

export read_input_file, NUMSetup, GroupType, initial_state,
       setup_num_model, setup_num_model_simple, setup_gen_diat_cope, setup_generic,
       setup_generalists_only, setup_generalists_simple_only, setup_generalists_pom,
       setup_generalists_simple_pom, setup_diatoms_only, setup_diatoms_simple_only,
       setup_generalists_diatoms, setup_generalists_diatoms_simple, setup_generalists_simple_copepod,
       set_htl!, set_mort_htl!, set_sinking!, set_sinking_pom!,
       kernel_setup, KSetup, calc_derivatives, simulate_euler!, net_primary_production!,
       get_functions, get_rates, get_balance, simulate_euler_functions!,
       GlobalParams, GlobalModel, load_global_model, simulate_global, matrix_to_grid, save_sim_mat,
       tm_paths, state_from_sim, daily_insolation, save_mat,
       WatercolumnParams, watercolumn_model, simulate_watercolumn,
       ChemostatParams, parameters_chemostat, simulate_chemostat, simulate_chemostat_euler, ode23s

include("input.jl")
include("setup.jl")
include("kernel.jl")
include("diagnostics.jl")
include("insolation.jl")
include("transport.jl")
include("watercolumn.jl")
include("chemostat.jl")
include("device.jl")
include("output.jl")

end
