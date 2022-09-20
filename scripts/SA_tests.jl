using DrWatson; @quickactivate :RFIMSimulatedAnnealing

struct BaseIsingModel{T <: AbstractFloat} <: AbstractModel
    lattice::CubicLattice2D
    hamiltonian::BasicIsingHamiltonian{T}

    BaseIsingModel(args...) = BaseIsingModel{Float64}(args...)
    function BasicIsingModel{T}(; Lx = 16, Ly = Lx, Jex = 1) where T
        latt = CubicLattice2D(Int(Lx), Int(Ly))
        ham = BasicIsingHamiltonian{T}(latt, BasicIsingParameters{T}(T(Jex)))
        return new{T}(latt, ham)
    end
end

ising_model = BaseIsingModel()