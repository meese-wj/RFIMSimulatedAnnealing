# Main code for concrete Ising Hamiltonians

export BasicIsingHamiltonian, BasicIsingParameters

struct BasicIsingParameters{T <: AbstractFloat} <: AbstractIsingParameters
    Jex::T  # Ising exchanges. Jex > 0 is ferromagnetic
end
BasicIsingParameters{T}(; J::T = 1. ) where {T <: AbstractFloat} = BasicIsingParameters{T}( J )

mutable struct BasicIsingHamiltonian{T <: AbstractFloat} <: AbstractIsing
    params::BasicIsingParameters{T}
    spins::Vector{T}

    function BasicIsingHamiltonian(latt, params::BasicIsingParameters{T}) where T
        ndofs = num_sites(latt)
        return new{T}(params, rand([one(T), -one(T)], ndofs))
    end
end

@inline site_energy(ham::BasicIsingHamiltonian, latt, site, site_values) = _base_site_energy(ham, latt, site, site_values)

@inline energy(ham::BasicIsingHamiltonian, latt) = _base_energy(ham, latt)

@inline DoF_energy_change(ham::BasicIsingHamiltonian, latt, site) = _base_DoF_energy_change(ham, latt, site)

@inline accept_move!(condition::Bool, ham::BasicIsingHamiltonian, site) = _base_accept_move!(condition, ham, site)