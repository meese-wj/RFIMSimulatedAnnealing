using DrWatson; @quickactivate :RFIMSimulatedAnnealing
using Plots, JLD2

# include("SA_tests.jl")

states = Vector{Float64}[]
for (Tdx, Temp) ∈ enumerate(Tregimen)
    @show flpath = MonteCarloMethods.SA_datapath(datadir(), ising_model, Temp)
    push!(states, load_state(flpath))
end

anim = @animate for (Temp, σstate) ∈ zip(Tregimen, states)
    heatmap( reshape(σstate, size(Lattice(ising_model)));
                     clims = (-1, 1),
                     aspect_ratio = :equal,
                     title = "\$T = $(round(Temp; digits = 3))\\, J\$" )
end
gif(anim, plotsdir("clean_tests.gif"), fps = 3)



