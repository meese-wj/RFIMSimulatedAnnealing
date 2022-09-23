# Main code for concrete Ising Hamiltonians
using Random, Distributions

export BasicIsingHamiltonian, BasicIsingParameters, RandomFieldIsingHamiltonian, RandomFieldIsingParameters

struct BasicIsingParameters{T <: AbstractFloat} <: AbstractIsingParameters
    Jex::T  # Ising exchanges. Jex > 0 is ferromagnetic
end
BasicIsingParameters{T}(; J::T = 1. ) where {T <: AbstractFloat} = BasicIsingParameters{T}( J )

struct RandomFieldIsingParameters{T <: AbstractFloat} <: AbstractIsingParameters
    Jex::T   # Ising exchanges. Jex > 0 is ferromagnetic
    hext::T  # External field. Spins should be colinear with hext by Zeeman energy
    Δh::T    # Characteristic width of the random field distribution.
end

mutable struct BasicIsingHamiltonian{T <: AbstractFloat} <: AbstractIsing
    params::BasicIsingParameters{T}
    spins::Vector{T}

    function BasicIsingHamiltonian(latt, params::BasicIsingParameters{T}) where T
        ndofs = num_sites(latt)
        return new{T}(params, rand([one(T), -one(T)], ndofs))
    end
end

function random_field_generator(Δh, ndofs)
    Δh == zero(Δh) ? zeros(typeof(Δh), ndofs) : nothing
    nrng = Normal(zero(Δh), Δh)
    return rand(nrng, ndofs)
end

mutable struct RandomFieldIsingHamiltonian{T <: AbstractFloat} <: AbstractZeemanIsing
    params::RandomFieldIsingParameters{T}
    spins::Vector{T}
    random_fields::Vector{T}

    function RandomFieldIsingHamiltonian(latt, params::RandomFieldIsingParameters{T}) where T
        ndofs = num_sites(latt)
        return new{T}(params, rand([one(T), -one(T)], ndofs), random_field_generator(params.Δh, ndofs))
    end
end

@inline Zeeman_field(ham::RandomFieldIsingHamiltonian, site) = ham.params.hext + ham.random_fields[site]

@inline site_energy(ham::BasicIsingHamiltonian, latt, site, site_values) = _base_site_energy(ham, latt, site, site_values)
@inline site_energy(ham::RandomFieldIsingHamiltonian, latt, site, site_value) = _base_site_energy(ham, latt, site, site_value) + _Zeeman_site_energy(ham, site, site_value)

@inline energy(ham::BasicIsingHamiltonian, latt) = _base_energy(ham, latt)
function energy(ham::RandomFieldIsingHamiltonian, latt)
    en = _base_energy(ham, latt)
    @inbounds for (site, dof_location_val) ∈ enumerate(IterateBySite, ham)
        en += _Zeeman_site_energy(ham, dof_location_val[1], dof_location_val[2])
    end
    return en
end

@inline DoF_energy_change(ham::BasicIsingHamiltonian, latt, site) = _base_DoF_energy_change(ham, latt, site)
@inline DoF_energy_change(ham::RandomFieldIsingHamiltonian, latt, site) = _base_DoF_energy_change(ham, latt, site) + _Zeeman_DoF_energy_change(ham, site)

@inline accept_move!(condition::Bool, ham::BasicIsingHamiltonian, site) = _base_accept_move!(condition, ham, site)
@inline accept_move!(condition::Bool, ham::RandomFieldIsingHamiltonian, site) = _base_accept_move!(condition, ham, site)

@inline relevant_parameters(ham::RandomFieldIsingHamiltonian, starter = "_") = starter * "hext=$(ham.params.hext)_Δh=$(ham.params.Δh)"