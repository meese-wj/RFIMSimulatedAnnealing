### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 93e02752-4362-11ed-09ef-c9fb59bc64bc
using DrWatson

# ╔═╡ 401503f0-6eab-47e6-86e4-3ea1484bcc49
@quickactivate "RFIMSimulatedAnnealing"

# ╔═╡ a048a2b3-93ed-4867-960a-75b44dc65959
using PlutoUI

# ╔═╡ c8692ca0-c816-420f-bb45-bb67fe48c5e2
using RFIMSimulatedAnnealing

# ╔═╡ 74b24b3f-8704-4230-aa66-d31563658b1c
using .MonteCarloMethods: AbstractModel

# ╔═╡ 210e7853-928c-4ccf-b5c3-26f98dc7e48d
using CairoMakie

# ╔═╡ d1830211-01dd-4e08-abd1-6aba49ed4f3d
using Statistics

# ╔═╡ fd243a2f-d8ee-487b-8630-0d88e158f4d8
md"""
# Simulated Annealing Summary Plots

This notebook generates the plots summarizing the real-space and spin-decimated behavior of a 2D RFIM undergoing a simulated annealing regimen under external field.
"""

# ╔═╡ f42ce74a-c980-445e-ad7e-82148cf4538d
TableOfContents()

# ╔═╡ ee2e0214-8599-4a8c-8e6c-04f9fdc2c316
md"""
## Load the necessary packages
"""

# ╔═╡ fd8533c7-8bd2-4b5b-b7bc-1a61aa5b4fe7
md"""
## Data directory

Find the data in this location:
"""

# ╔═╡ 1e0245c8-47d6-4e82-b882-60ff54c9b932
datalocation() = datadir("SA_Summary")

# ╔═╡ aafa068c-6a09-4f64-88b7-062553550c24
md"""
## Define the simulation parameters of interest
"""

# ╔═╡ a899c224-eb20-466d-9eed-4202adad6f02
begin
	@show const Lvalue = 512
	@show const Jex = 1.0
	@show const Tc = 2 * Jex / asinh(1)
	@show const Deltah = 0.95 * Jex 
	# const bias_ratios = Float64[0.001, 0.01, 0.0375, 0.04192, 0.05, 0.1]
	const bias_ratios = reverse( Float64[0.001, 0.0375, 0.1] )
	@show hext_values = Deltah .* bias_ratios
	const Tregimen = logspace_temperatures(8, 0.1, 50)
	const Tregimen_indices = reverse([10, 15, 18, 20, 50])
	const Tregimen_chosen = [ Tregimen[idx] for idx ∈ Tregimen_indices ]
end

# ╔═╡ 1e85f40d-2aac-4daa-ba4a-ad4a46d7da2f
for T ∈ Tregimen_chosen
	@show T
end

# ╔═╡ 25a02cf6-addf-493b-84ac-ade04fbc3c48
md"""
## Create the RFIM type

Since I haven't exported this type yet, it needs to be copied for the time being...
"""

# ╔═╡ 2f3ea1d5-5c20-4da3-af0d-1f4628cb8c4c
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

# ╔═╡ 9b6c9aed-1cb0-4575-9937-1f6c4bc460cb
md"""
## Load the necessary states
"""

# ╔═╡ ba3478b4-13d5-44ff-9227-24301a59908e
function load_states(datapath, Tvalues = Tregimen)
	all_states = []
	for idx ∈ eachindex(bias_ratios)
		ising_model = RandomFieldIsingModel(Lvalue, Lvalue, Jex,
											hext_values[idx], Deltah)
		states = Vector{Float64}[]
		for (Tdx, Temp) ∈ enumerate(Tvalues)
		    flpath = MonteCarloMethods.SA_datapath(datapath, ising_model, Temp)
		    push!(states, load_state(flpath))
		end
		push!(all_states, states)
	end
	return all_states
end

# ╔═╡ c4850845-b24e-4143-8689-83e25f1b7c6e
all_states = load_states( datalocation(), Tregimen_chosen )

# ╔═╡ 76f862bb-c304-43dc-9351-70f710bc9923
md"""
## Plot the $h_{\rm ex}$-$T$ plane with images

In this plot, we will have a $N_h \times N_T$ grid of different `heatmap`s for the different RFIM results. Here $N_h$ is the number of external fields, and $N_T$ in the number of temperatures to be plotted.
"""

# ╔═╡ 162b836b-8952-4fa9-ad4a-e987b14160dc
begin

Nhext = length(bias_ratios)
NTemp = length(Tregimen_chosen)

gapsize = 10

HM_fig = Figure(; 
		 backgroundcolor = :white
)

figcmap = :plasma
figdigits = 3

GL_HMs = HM_fig[2:(Nhext + one(Nhext)), 2:(NTemp + one(NTemp)) ] = GridLayout()
GL_Tlabel = HM_fig[(Nhext + 2), 1:end] = GridLayout()
GL_Hlabel = HM_fig[2:(Nhext + one(Nhext)), 1] = GridLayout()
GL_Cbar = HM_fig[1, 2:(NTemp + one(NTemp))] = GridLayout()

figaxes = []
hm1 = nothing
for hdx ∈ eachindex(bias_ratios), Tdx ∈ eachindex(Tregimen_chosen)
	ax, hm = heatmap(GL_HMs[hdx, Tdx], reshape(all_states[hdx][Tdx], Lvalue, Lvalue);
					 clims = (-1, 1),
					 colormap = figcmap,
	)
	hidedecorations!(ax)
	push!(figaxes, ax)

	if hdx == one(hdx) && Tdx == one(Tdx)
		global hm1 = hm
	end
end

Colorbar(GL_Cbar[1, :], hm1; label = L"Spin $S_i^z$", vertical = false, 
		 alignmode = Mixed())

linkaxes!(figaxes...)

colgap!(GL_HMs, gapsize)
rowgap!(GL_HMs, gapsize)

for Tdx ∈ eachindex(Tregimen_chosen)
	Tcratio = round( Tregimen_chosen[Tdx] / Tc; digits = figdigits )
	Label(GL_HMs[Nhext, Tdx, Bottom()], L"$ T = %$Tcratio \, T_c^{(0)}$"; 
		  valign = :center, halign = :center, padding = (0, 0, 0, 3gapsize),
		  tellheight = false)
end

for ηdx ∈ eachindex(bias_ratios)
	η = round( bias_ratios[ηdx]; digits = figdigits )
	Label(GL_HMs[ηdx, 1, Left()], L"$ \eta = %$η $"; rotation = π/2, 
          valign = :center, padding = (0, 3gapsize, 0, 0),
		  tellwidth = false)
end

Label(GL_Tlabel[1, 1:NTemp],  L"Temperature $T$"; tellheight = true, 
	  padding = (0, 0, 0, gapsize))
Label(GL_Hlabel[:, 1], L"Bias $\eta = h_{\mathrm{ex}} / \Delta h$"; 
	  tellheight = true, rotation = π/2, padding = (0, gapsize, 0, 0))

for idx ∈ 1:NTemp
	colsize!(GL_HMs, idx, Aspect(1, 1.0))
end
	
# resize_to_layout!(HM_fig)

save( plotsdir("domain_plots.svg"), HM_fig, pt_per_unit = 1)

HM_fig
end

# ╔═╡ 05f1a2dd-5c64-475d-8853-4e1d8077916a
md"""
## Decimation and density plots
"""

# ╔═╡ c2cab92a-b0a7-4d3b-a15c-e731f740aa32
function decimate(img; bx::Int = 16, by::Int = bx)
	Lx, Ly = size(img)
	Lx % bx != zero(bx) ? throw(ArgumentError("bx = $bx must be an integer divisor of Lx = $Lx.")) : nothing
	Ly % by != zero(by) ? throw(ArgumentError("by = $by must be an integer divisor of Ly = $Ly.")) : nothing

	Nx, Ny = (Lx ÷ bx, Ly ÷ by)
	RG_img = similar(img, Nx, Ny)
	for ib ∈ 1:Nx, jb ∈ 1:Ny
		RG_img[ib, jb] = mean( img[ one(ib) + (ib - one(ib)) * bx : ib * bx, 
									one(jb) + (jb - one(jb)) * by : jb * by ] )
	end
	return RG_img
end

# ╔═╡ b9f49a16-bf4e-436e-b2f7-b7b2ba1fd3b1
begin
	
bvalue = 8
RG_fig = Figure(; 
		 backgroundcolor = :white
)

# RG_plot_type = density
RG_plot_type = hist

GL_RGs = RG_fig[1:(Nhext + one(Nhext)), 2:(NTemp + one(NTemp)) ] = GridLayout()
RG_Tlabel = RG_fig[(Nhext + 2), 1:end] = GridLayout()
RG_Hlabel = RG_fig[1:(Nhext + one(Nhext)), 1] = GridLayout()

RGfigaxes = []
for hdx ∈ eachindex(bias_ratios), Tdx ∈ eachindex(Tregimen_chosen)
	RG_img = decimate( reshape(all_states[hdx][Tdx], Lvalue, Lvalue); bx = bvalue )
	ax, RGplt = RG_plot_type(GL_RGs[hdx, Tdx], RG_img[:]; npoints = 200,
						strokecolor = :blue,
						# strokearound = true,
						strokewidth = 0.5,
						# boundary = (-1.01, 1.01),
						bins = 16,
						color = (:blue, 0.25),
						axis = (; xgridvisible = true, 
								  ygridvisible = false,
								  xlabel = L"$\mathrm{Dec}( S_i^z; \, %$ bvalue )$",
								  xticks = [-1, 0, 1],
						),
	)

	vlines!( GL_RGs[hdx, Tdx], mean(@view RG_img[:]); color = :orange,
			 linewidth = 3, linestyle = :dash)

	hlines!( GL_RGs[hdx, Tdx], zero(Float64); color = :gray80, linewidth = 1)
	
	hideydecorations!(ax)

	if hdx != length(bias_ratios)
		hidexdecorations!(ax)
		ax.xgridvisible = true
	end
	
	push!(RGfigaxes, ax)
end

linkxaxes!(RGfigaxes...)

for idx ∈ 1:NTemp
	colsize!(GL_HMs, idx, Aspect(1, 1.0))
end

for Tdx ∈ eachindex(Tregimen_chosen)
	Tcratio = round( Tregimen_chosen[Tdx] / Tc; digits = figdigits )
	Label(GL_RGs[Nhext, Tdx, Bottom()], L"$ T = %$Tcratio \, T_c^{(0)}$"; 
	  valign = :center, halign = :center, padding = (0, 0, 0, 8gapsize),
	  tellheight = false)
end

for ηdx ∈ eachindex(bias_ratios)
	η = round( bias_ratios[ηdx]; digits = figdigits )
	Label(GL_RGs[ηdx, 1, Left()], L"$ \eta = %$η $"; rotation = π/2, 
          valign = :center, padding = (0, 3gapsize, 0, 0),
		  tellwidth = false)
end

Label(RG_Tlabel[1, 1:NTemp], L"Temperature $T$"; tellheight = true, 
	  padding = (0, 0, 0, 2gapsize))
Label(RG_Hlabel[:, 1], L"Bias $\eta = h_{\mathrm{ex}} / \Delta h$"; 
	  tellheight = true, rotation = π/2, padding = (0, gapsize, 0, 0))

colgap!(GL_RGs, gapsize)
rowgap!(GL_RGs, gapsize)

save( plotsdir("$(RG_plot_type)_plots.svg"), RG_fig, pt_per_unit = 1)

RG_fig
	
end

# ╔═╡ 82ca6a4b-5dfe-4496-9e32-8256473bcbc3
begin
	cartoon = deepcopy(all_states[2][4])
	for (idx, spin) ∈ enumerate(cartoon)
		if spin < zero(spin)
			cartoon[idx] = rand() < 0.5 ? zero(spin) : spin
		end
	end
	cartoon
end

# ╔═╡ f9e187a4-e80a-42c4-8daa-727deb4e44d7
let
fig, ax, hm = heatmap(reshape(cartoon, Lvalue, Lvalue); colormap = [(:blue, 1), (:red, 0.5), (:green, 0.2)],
		figure = (; resolution = (Lvalue, Lvalue)),
)
hidedecorations!(ax)
fig
end

# ╔═╡ Cell order:
# ╟─fd243a2f-d8ee-487b-8630-0d88e158f4d8
# ╠═a048a2b3-93ed-4867-960a-75b44dc65959
# ╠═f42ce74a-c980-445e-ad7e-82148cf4538d
# ╟─ee2e0214-8599-4a8c-8e6c-04f9fdc2c316
# ╠═93e02752-4362-11ed-09ef-c9fb59bc64bc
# ╠═401503f0-6eab-47e6-86e4-3ea1484bcc49
# ╠═c8692ca0-c816-420f-bb45-bb67fe48c5e2
# ╠═74b24b3f-8704-4230-aa66-d31563658b1c
# ╟─fd8533c7-8bd2-4b5b-b7bc-1a61aa5b4fe7
# ╠═1e0245c8-47d6-4e82-b882-60ff54c9b932
# ╟─aafa068c-6a09-4f64-88b7-062553550c24
# ╠═a899c224-eb20-466d-9eed-4202adad6f02
# ╠═1e85f40d-2aac-4daa-ba4a-ad4a46d7da2f
# ╟─25a02cf6-addf-493b-84ac-ade04fbc3c48
# ╠═2f3ea1d5-5c20-4da3-af0d-1f4628cb8c4c
# ╟─9b6c9aed-1cb0-4575-9937-1f6c4bc460cb
# ╠═ba3478b4-13d5-44ff-9227-24301a59908e
# ╠═c4850845-b24e-4143-8689-83e25f1b7c6e
# ╟─76f862bb-c304-43dc-9351-70f710bc9923
# ╠═210e7853-928c-4ccf-b5c3-26f98dc7e48d
# ╠═162b836b-8952-4fa9-ad4a-e987b14160dc
# ╟─05f1a2dd-5c64-475d-8853-4e1d8077916a
# ╠═d1830211-01dd-4e08-abd1-6aba49ed4f3d
# ╠═c2cab92a-b0a7-4d3b-a15c-e731f740aa32
# ╠═b9f49a16-bf4e-436e-b2f7-b7b2ba1fd3b1
# ╠═82ca6a4b-5dfe-4496-9e32-8256473bcbc3
# ╠═f9e187a4-e80a-42c4-8daa-727deb4e44d7
