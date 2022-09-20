
import ..Hamiltonians: energy, write_state
export anneal!, SA_datapath

function SA_datapath(model::AbstractModel, temperature)
    return "NDoFs=$(num_DoF(model))_T=$temperature"
end

SA_datapath(pathprepend::String = "", args...) = pathprepend * SA_datapath(args...)

function anneal!(model::AbstractModel, temperature_regimen, mc_params::AbstractMonteCarloParameters, mc_sweep::Function, write_out = true, pathprepend = "")
    spe = sweeps_per_export(mc_params)
    @inbounds for temperature ∈ temperature_regimen
        beta = 1 / temperature
        # TODO: write in the adaptive part
        # while not equilibrated
        @inbounds for sweep ∈ (1:spe)
            thermalize!(model, beta, mc_params, mc_sweep)
        end
        # end
        write_out ? write_state(Hamiltonian(model), SA_datapath(pathprepend, model)) : nothing
    end
end