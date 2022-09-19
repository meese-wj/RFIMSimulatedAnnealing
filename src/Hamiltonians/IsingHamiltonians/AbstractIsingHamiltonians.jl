
import StaticArrays: @SVector, @MVector
import Base: getindex, setindex!, to_index
import ..Lattices: num_sites  # Automatically submodulizes this code
import MuladdMacro: @muladd 
using Roots: find_zero

export 
# Base overloads
       getindex, setindex!, eltype, length, to_index,
# Abstract Ising exports
       critical_temperature, sigma_values

############################################################################
#             Abstract Ising Hamiltonian Interface                         #
############################################################################

abstract type AbstractIsing <: AbstractHamiltonian end
abstract type AbstractIsingParameters <: AbstractHamiltonianParameters end

@inline Base.getindex(ham::AbstractIsing, idx) = spins(ham)[idx]
@inline Base.setindex!(ham::AbstractIsing, value, idx) = spins(ham)[idx] = value
@inline Base.eltype(ham::AbstractIsing) = eltype(spins(ham))
@inline Base.length( ham::AbstractIsing ) = length(spins(ham))
@inline num_sites( ham::AbstractIsing ) = length(ham)
# Base.size( ham::AbstractIsing ) = ( num_sites(ham), num_sites(ham) )
@inline num_DoF( ham::AbstractIsing ) = length(ham)

"""
    iterate(::HamiltonianIterator{<: AbstractIsing, IterateByDefault}, [state])

Traverse a `<:`[`AbstractIsing`](@ref) as it is laid out in memory.
"""
function iterate(iter::HamiltonianIterator{<: AbstractIsing, IterateByDefault}, state = (one(Int),))
    ham = Hamiltonian(iter)
    spin_idx, = state
    next = (spin_idx + one(Int),)
    return spin_idx <= length(ham) ? ((spin_idx, ham[spin_idx], next) : nothing
end

"""
    iterate(::HamiltonianIterator{<: AbstractIsing, IterateBySite}, [state])

Traverse a `<:`[`AbstractIsing`](@ref) and return the spin value.
"""
function iterate(iter::HamiltonianIterator{<: AbstractIsing, IterateBySite}, state = (one(Int),))
    ham = Hamiltonian(iter)
    site, = state
    next = (site + one(Int), )
    return site <= num_sites(ham) ? ((site, ham[site]), next) : nothing
end

"""
    iterate(::HamiltonianIterator{<: AbstractIsing, IterateByDoFType}, [state])

Traverse a `<:`[`AbstractIsing`](@ref) by the DoF type which is just a single Ising spin.
"""
iterate(iter::HamiltonianIterator{T, IterateByDoFType}, state = (one(Int),)) where {T <: AbstractIsing} = iterate(HamiltonianIterator(Hamiltonian(iter), IterateByDefault))

"""
    sigma_values(::AbstractIsing)

Return a `view` into the σ degrees of freedom.
"""
@inline sigma_values(ham::AbstractIsing) = @view spins(ham)

"""
    neighbor_field(::AbstractIsing, ::AbstractIsingParameters, latt, site)

Return the fields at a given `site` generated by the neighbors in a `latt`ice.
"""
@inline function neighbor_field(ham::AbstractIsing, hamparams::AbstractIsingParameters, latt, site)
    near_neighbors = nearest_neighbors(latt, site)
    σ_field = zero(eltype(ham))
    @inbounds for nn ∈ near_neighbors
        σ_field += ham[nn]
    end
    return hamparams.Jex * σ_field
end

@doc raw"""
    _base_site_energy(::AbstractIsing, latt, site, site_values)

Calculate the energy for a single site. This 
_base_ energy is for a clean model defined by 

```math
H = -J \sum_{\langle ij \rangle} \sigma_i\sigma_j .
```
"""
@inline _base_site_energy( ham::AbstractIsing, latt, site, site_value ) = -site_value * neighbor_field(ham, ham.params, latt, site)    

@doc raw"""
    _base_energy(::AbstractIsing, latt)

Calculate the base energy for a pure Ashkin-Teller Hamiltonian given
on a given `latt`ice by the following Hamiltonian:

```math
H = -J \sum_{\langle ij \rangle} \sigma_i\sigma_j .
```
"""
@inline function _base_energy( ham::AbstractIsing, latt )
    en = zero(eltype(ham))
    @inbounds for (iter, dof_location_val) ∈ enumerate(IterateBySite, ham)
        en += _base_site_energy( ham, latt, dof_location_val...)
    end
    return 0.5 * en
end

"""
    _base_DoF_energy_change(::AbstractIsing, latt, site)

Calculate the change in energy for a base [`AbstractIsing`](@ref) model.
This function works by calling [`_base_site_energy`](@ref) where the 'site_value'
is Δσ = σ1 - σ0 = -2σ0.
"""
@inline _base_DoF_energy_change(ham::AbstractIsing, latt, site) = _base_site_energy( ham, latt, site, -2 * ham[site] )

"""
    _base_accept_move!(::Bool, ::AbstractIsing, site)

Defines a binary operation of how to accept a single-DoF spin flip move if the `Bool`ean is 
evaluated to `true`. This `Bool`ean may be, for example, the Metropolis acceptance condition.
"""
@inline _base_accept_move!( condition::Bool, ham::AbstractIsing, site ) = (ham[site] = condition ? -ham[site] : ham[site]; nothing)

@inline critical_temperature(::Type{<: AbstractIsing}, Jex) = 2 * Jex / asinh(one(Jex))