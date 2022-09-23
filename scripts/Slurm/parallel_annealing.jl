#!/usr/bin/bash -l
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10g
#SBATCH --mail-type=all
#SBATCH --mail-user=meese022@umn.edu
#SBATCH --array=1-5
#SBATCH --job-name=L-128_%A_%a.out
#SBATCH -o %x-%j.out
#=
    pwd
    echo $SLURM_NPROCS
    echo $SLURM_CPUS_PER_TASK
    echo
    srun julia --threads=$SLURM_CPUS_PER_TASK parallel_annealing.jl
    exit
=#
using Pkg
using DrWatson
@quickactivate :RFIMSimulatedAnnealing
# Pkg.instantiate()

using Random
Random.seed!(42)

const slurm_arr_length::Int = parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
@show const my_index::Int = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

@show const Lvalue = 16
@show const Jex = 1.0
@show const Deltah = 0.95 * Jex 
bias_ratios = Float64[0, 0.0001, 0.001, 0.01, 0.1]
@show hext_values = Deltah .* bias_ratios

include("../SA_RFIM_tests.jl")

ising_model = RandomFieldIsingModel(Lvalue, Lvalue, Jex, hext_values[my_index], Deltah)
Tregimen = Float64[8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2.375, 2.325, 2.275, 2.25, 2.20, 2.15, 2.125, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.1]
# Tregimen = Float64[3]
# sa_params = SimulatedAnnealingParameters(2^10, 1, Float64[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5])
sa_params = SimulatedAnnealingParameters(10, 2^11, 0.01, Tregimen)
@time anneal!(ising_model, sa_params, metropolis_sweep!, true, datadir())