using DrWatson; @quickactivate :RFIMSimulatedAnnealing
using RFIMSimulatedAnnealing.MonteCarloMethods: AbstractModel

struct BaseIsingModel{T <: AbstractFloat} <: AbstractModel
    lattice::CubicLattice2D
    hamiltonian::BasicIsingHamiltonian{T}

    BaseIsingModel(args...) = BaseIsingModel{Float64}(args...)
    function BaseIsingModel{T}(Lx = 16, Ly = Lx, Jex = 1) where T
        latt = CubicLattice2D(Int(Lx), Int(Ly))
        ham = BasicIsingHamiltonian(latt, BasicIsingParameters{T}(T(Jex)))
        return new{T}(latt, ham)
    end
end

ising_model = BaseIsingModel(64)
Tregimen = Float64[8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2.375, 2.325, 2.275, 2.25, 2.20, 2.15, 2.125, 2.0, 1.5, 1.0, 0.5, 0.1]
# sa_params = SimulatedAnnealingParameters(2^10, 1, Float64[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5])
sa_params = SimulatedAnnealingParameters(10, 2^11, 0.01, Tregimen)
@time anneal!(ising_model, sa_params, metropolis_sweep!, true, datadir())