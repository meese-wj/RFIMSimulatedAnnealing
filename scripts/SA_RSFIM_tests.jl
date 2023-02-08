using DrWatson; @quickactivate :RFIMSimulatedAnnealing
using RFIMSimulatedAnnealing.MonteCarloMethods: AbstractModel
using RandomStrainDistributions
using Random

struct RandomFieldIsingModel{T <: AbstractFloat} <: AbstractModel
    lattice::CubicLattice2D
    hamiltonian::RandomFieldIsingHamiltonian{T}

    function RandomFieldIsingModel{T}(Lx = 16, Ly = Lx, Jex = 1, hext = 0, Δh = 0) where T
        latt = CubicLattice2D(Int(Lx), Int(Ly))
        ham = RandomFieldIsingHamiltonian( latt, RandomFieldIsingParameters{T}(T(Jex), T(hext), T(Δh)) )
        return new{T}(latt, ham)
    end
    RandomFieldIsingModel(args...) = RandomFieldIsingModel{Float64}(args...)
end

Random.seed!(42)
Lx = 64
Ly = Lx
con = 1024 / (Lx * Ly)
rsd = RandomDislocationDistribution(; concentration = con,
									  Lx = Lx, Ly = Ly, )
dislocations = collect_dislocations(rsd)
@time strains = generate_disorder(Lx, Ly, dislocations; include_Δ = false, tolerance = 1e-3)
@show strains[:, :, 1] |> maximum

ising_model = RandomFieldIsingModel(Ly, Lx, 1, 0.0, 0.0)
ising_model.hamiltonian.random_fields = reshape(strains[:, :, 1], (Lx * Ly,))
@show ising_model.hamiltonian.random_fields |> maximum

Tregimen = Float64[8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2.375, 2.325, 2.275, 2.25, 2.20, 2.15, 2.125, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.1]
sa_params = SimulatedAnnealingParameters(10, 2^11, 0.01, Tregimen)
@time anneal!(ising_model, sa_params, metropolis_sweep!, false, datadir())