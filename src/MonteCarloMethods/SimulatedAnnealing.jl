
import ..Hamiltonians: energy, write_state, num_DoF
using Printf

export anneal!, SA_datapath, SimulatedAnnealingParameters

struct SimulatedAnnealingParameters{T <: Float64} <: AbstractMonteCarloParameters
    therm_sweeps::Int        # Total number of sweeps while thermalizing at a given temperature
    # measure_sweeps::Int    # Total number of sweeps used to take measurements
    total_measurements::Int  # Total number of measurements
    temperatures::Vector{T}

    SimulatedAnnealingParameters(args...) = SimulatedAnnealingParameters{Float64}(args...)
    SimulatedAnnealingParameters{T}(args...) where T = new{T}(args...)
end
thermalization_sweeps(params::SimulatedAnnealingParameters) = params.therm_sweeps
sampling_sweeps(params::SimulatedAnnealingParameters) = zero(Int)
total_measurements(params::SimulatedAnnealingParameters) = params.total_measurements
sweeps_per_export(params::SimulatedAnnealingParameters) = thermalization_sweeps(params)
temperatures(params::SimulatedAnnealingParameters) = params.temperatures

function SA_datapath(model::AbstractModel, temperature)
    return "NDoFs=$(num_DoF(model))_T=$temperature"
end

SA_datapath(pathprepend::String = "", args...) = pathprepend * SA_datapath(args...)

function SA_info(timer, model, mc_params, iteration)
    time = timer.time
    total_sweeps = sweeps_per_export(mc_params) * thermalization_sweeps(mc_params)
    sweep_rate = total_sweeps / time
    update_rate = num_DoF(Hamiltonian(model)) * sweep_rate
    @info "Iteration $iteration --> $(round(100 * iteration / length(temperatures(mc_params)); digits = 3))% complete."
    println( @sprintf "    Total sweeps: %.3e" total_sweeps )
    println( @sprintf "    Total time:   %.3e seconds" time )
    println( @sprintf "    Sweep rate:   %.3e sweeps per second" sweep_rate )
    println( @sprintf "    Update rate:  %.3e updates per second" update_rate )
    return nothing
end

function anneal!(model::AbstractModel, mc_params::AbstractMonteCarloParameters, mc_sweep::Function, write_out = true, pathprepend = "")
    temperature_regimen = temperatures(mc_params)
    spe = sweeps_per_export(mc_params)
    @inbounds for (Tdx, temperature) ∈ enumerate(temperature_regimen)
        beta = 1 / temperature
        # TODO: write in the adaptive part
        # while not equilibrated
        timer = @timed @inbounds for sweep ∈ (1:spe)
            thermalize!(model, beta, mc_params, mc_sweep)
        end
        SA_info(timer, model, mc_params, Tdx)
        # end
        write_out ? write_state(Hamiltonian(model), SA_datapath(pathprepend, model, temperature)) : nothing
    end
end