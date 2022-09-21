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
# sa_params = SimulatedAnnealingParameters(2^10, 1, Float64[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5])
sa_params = SimulatedAnnealingParameters(10, 2^10, 0.05, Float64[8])
@time anneal!(ising_model, sa_params, metropolis_sweep!, false)