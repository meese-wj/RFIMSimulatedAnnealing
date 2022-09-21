
import ..Hamiltonians: energy, write_state, num_DoF
using Printf
using Statistics

export anneal!, SA_datapath, SimulatedAnnealingParameters

const SA_EQUILIBRATION_PARTITIONS::Int = 2

struct SimulatedAnnealingParameters{T <: Float64} <: AbstractMonteCarloParameters
    therm_sweeps::Int        # Sweeps to thermalize before recording the energy at a given temperature (probably 1-10)
    measure_sweeps::Int      # Total number of sweeps used to take measurements while equilibrating
    tolerance::Float64       # Variance tolerance for the adaptive SA algorithm (usually <= 0.05)
    temperatures::Vector{T}

    SimulatedAnnealingParameters(args...) = SimulatedAnnealingParameters{Float64}(args...)
    SimulatedAnnealingParameters{T}(args...) where T = new{T}(args...)
end
thermalization_sweeps(params::SimulatedAnnealingParameters) = params.therm_sweeps
sampling_sweeps(params::SimulatedAnnealingParameters) = params.measure_sweeps
total_measurements(params::SimulatedAnnealingParameters) = SA_EQUILIBRATION_PARTITIONS * sampling_sweeps(params)
sweeps_per_export(params::SimulatedAnnealingParameters) = thermalization_sweeps(params)
tolerance(params::SimulatedAnnealingParameters) = params.tolerance
temperatures(params::SimulatedAnnealingParameters) = params.temperatures

function SA_datapath(model::AbstractModel, temperature)
    return "NDoFs=$(num_DoF(model))_T=$temperature"
end

SA_datapath(pathprepend::String = "", args...) = pathprepend * SA_datapath(args...)

function SA_info(timer, equilis, model, mc_params, iteration)
    time = timer.time
    total_sweeps = equilis * total_measurements(mc_params) * thermalization_sweeps(mc_params)
    sweep_rate = total_sweeps / time
    update_rate = num_DoF(Hamiltonian(model)) * sweep_rate
    @info "Adaptive Iteration $iteration --> $(round(100 * iteration / length(temperatures(mc_params)); digits = 3))% complete."
    println( @sprintf "    Equilibrations: %d" equilis )
    println( @sprintf "    Total sweeps:   %.3e" total_sweeps )
    println( @sprintf "    Total time:     %.3e seconds" time )
    println( @sprintf "    Sweep rate:     %.3e sweeps per second" sweep_rate )
    println( @sprintf "    Update rate:    %.3e updates per second" update_rate )
    return nothing
end

function not_equilibrated(energy_obs, mc_params)
    partition_divider = length(energy_obs) ÷ SA_EQUILIBRATION_PARTITIONS
    part1 = @view energy_obs[1:partition_divider]
    part2 = @view energy_obs[(partition_divider + one(Int)):end]
    μ1, var1 = mean( part1 ), var( part1 )
    μ2, var2 = mean( part2 ), var( part2 )
    condition_1::Bool = μ1 == μ2 == zero(μ1)
    condition_2::Bool = (abs2( μ1 - μ2 ) <= var1) && ((abs(var1 - var2) / var1) <= tolerance(mc_params)) 
    return !(condition_1 || condition_2)
end

function equilibrate!(model, energy_obs, beta, mc_params, mc_sweep)
    Evalue = zero(eltype(Hamiltonian(model)))
    equili_counter = zero(Int)
    while equili_counter == zero(equili_counter) || not_equilibrated(energy_obs, mc_params)
        equili_counter += one(equili_counter)
        for idx ∈ (1:total_measurements(mc_params))
            thermalize!(model, beta, mc_params, mc_sweep)
            energy_obs[idx] = energy(Hamiltonian(model), Lattice(model))
        end
    end
    return equili_counter
end

function anneal!(model::AbstractModel, mc_params::AbstractMonteCarloParameters, mc_sweep::Function, write_out = true, pathprepend = "")
    temperature_regimen = temperatures(mc_params)
    spe = sweeps_per_export(mc_params)
    energy_obs = zeros(eltype(Hamiltonian(model)), total_measurements(mc_params) )
    @inbounds for (Tdx, temperature) ∈ enumerate(temperature_regimen)
        beta = 1 / temperature
        # # TODO: write in the adaptive part
        # # while not equilibrated
        # timer = @timed thermalize!(model, beta, mc_params, mc_sweep)
        timer = @timed equilibrate!(model, energy_obs, beta, mc_params, mc_sweep)
        SA_info(timer, timer.value, model, mc_params, Tdx)
        # end
        write_out ? write_state(Hamiltonian(model), SA_datapath(pathprepend, model, temperature)) : nothing
    end
end