
import ..Hamiltonians: energy, write_state
export anneal!

function anneal!(model::AbstractModel, temperature_regimen, mc_params::AbstractMonteCarloParameters, mc_sweep::Function, write_out = true, filepath = ".")
    spe = sweeps_per_export(mc_params)
    @inbounds for temperature ∈ temperature_regimen
        beta = 1 / temperature
        # TODO: write in the adaptive part
        # while not equilibrated
        @inbounds for sweep ∈ (1:spe)
            thermalize!(model, beta, mc_params, mc_sweep)
        end
        # end
        write_out ? write_state(Hamiltonian(model), filepath) : nothing
    end
end