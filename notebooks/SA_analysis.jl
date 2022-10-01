### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2369d718-3b59-11ed-1243-7be9bca30bb0
using DrWatson

# ╔═╡ c57a8de6-2380-4cc2-afe2-25c6f77bd568
@quickactivate "RFIMSimulatedAnnealing"

# ╔═╡ 399f3c2d-cc5b-4f03-ad03-62724185cc19
using RFIMSimulatedAnnealing

# ╔═╡ 7534377e-4604-4500-a60c-70d915f337ed
using RFIMSimulatedAnnealing.MonteCarloMethods

# ╔═╡ 5b0f17d5-67f2-43c8-9e1f-d4fec4c6db4e
using .MonteCarloMethods: AbstractModel

# ╔═╡ 7aefb438-8661-4601-9848-5444389b72db
using Plots

# ╔═╡ 61097b03-696d-40c5-801b-964a2bf885e3
using Statistics

# ╔═╡ a66bee21-6b24-450b-b29a-4734d621d35e
using FFTW

# ╔═╡ f0a50e35-d4c7-4585-b2d6-8dc5b9cd90a9
using FastGaussQuadrature, LinearAlgebra, IntervalRootFinding

# ╔═╡ 73f6c72d-9995-4dad-87fa-eda04b7420a4
using Roots

# ╔═╡ 7c482e32-dfec-4a64-a0c6-11071bb81671
names(RFIMSimulatedAnnealing)

# ╔═╡ 81e239d2-4fda-4402-9f28-21f1e477f1f5
begin
	@show const Lvalue = 512
	@show const Jex = 1.0
	@show const Deltah = 0.95 * Jex 
	# bias_ratios = Float64[0, 0.0001, 0.001, 0.01, 0.1]
	# bias_ratios = Float64[0, 0.01, 0.025, 0.05, 0.1]
	# bias_ratios = Float64[0.025, 0.03125, 0.0375, 0.04375, 0.05]
	bias_ratios = Float64[0.001, 0.01, 0.0375, 0.05, 0.1]
	@show hext_values = Deltah .* bias_ratios
	const Tregimen = Float64[8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2.375, 2.325, 2.275, 2.25, 2.20, 2.15, 2.125, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.1]
end

# ╔═╡ 96190616-8bf8-44e3-b429-640607a156df
(0.375+0.5) /2

# ╔═╡ 9c75e7b2-1849-4299-bcf0-8838c1a719e2
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

# ╔═╡ 0f351599-49e9-4cf8-8d38-e93a05cf8154
function load_states(datapath)
	all_states = []
	for idx ∈ eachindex(bias_ratios)
		ising_model = RandomFieldIsingModel(Lvalue, Lvalue, Jex,
											hext_values[idx], Deltah)
		states = Vector{Float64}[]
		for (Tdx, Temp) ∈ enumerate(Tregimen)
		    flpath = MonteCarloMethods.SA_datapath(datapath, ising_model, Temp)
		    push!(states, load_state(flpath))
		end
		push!(all_states, states)
	end
	return all_states
end

# ╔═╡ e6250092-b266-4f57-a344-eba07b562291
all_states = load_states(datadir("Agate"))

# ╔═╡ da57e149-08d4-479c-8af1-5a79199508fb
function SA_heatmap(Temp, σstate, title_suffix)
	heatmap( reshape(σstate, (Lvalue, Lvalue));
                     clims = (-1, 1),
                     aspect_ratio = :equal,
                     title = "\$T = $(round(Temp; digits = 3))\\, J\$" * title_suffix 	)
end

# ╔═╡ 73b4aca5-e66e-41e8-9f3e-c5d4e3941734
function SA_gif(single_states; fps = 3, title_suffix = "")
	anim = @animate for (Temp, σstate) ∈ zip(Tregimen, single_states)
    	SA_heatmap(Temp, σstate, title_suffix)
	end
	gif(anim, fps = fps)
end

# ╔═╡ e462bdfd-d37d-4cb2-9481-1a019d2519ce
@bind bias_index html"""<input value="1" type="range" min="1" max="5"/>"""

# ╔═╡ 85d92807-333e-4a6b-a48e-ae1b33989c87
let
idx = bias_index
SA_gif(all_states[idx]; fps = 4,
	   title_suffix = "\$, \\, h_{\\mathrm{ex}} = $(hext_values[idx]) \\, J \$"
					  * "\$, \\, \\Delta h = $(Deltah) \\, J \$" )
end

# ╔═╡ 5be70f7a-2664-43d3-9d70-8e914190782a
four_plot_indices = [1, 2, 3, 5]

# ╔═╡ dff5bb0b-c185-4c54-b1f3-334876a7e766
@bind temperature_index html"""<input value="1" type="range" min="1" max="24"/>"""

# ╔═╡ 04f20fa1-ca64-4928-9dfc-f051cdc2733d
let

idx = four_plot_indices[1]
plt1 = SA_heatmap(Tregimen[temperature_index], all_states[idx][temperature_index],
	   	   "\$, \\, h_{\\mathrm{ex}} = $(round(hext_values[idx]; digits = 5)) \\, J \$" )

idx = four_plot_indices[2]
plt2 = SA_heatmap(Tregimen[temperature_index], all_states[idx][temperature_index],
	   	   "\$, \\, h_{\\mathrm{ex}} = $(round(hext_values[idx]; digits = 5)) \\, J \$" )

idx = four_plot_indices[3]
plt3 = SA_heatmap(Tregimen[temperature_index], all_states[idx][temperature_index],
	   	   "\$, \\, h_{\\mathrm{ex}} = $(round(hext_values[idx]; digits = 5)) \\, J \$" )

idx = four_plot_indices[4]
plt4 = SA_heatmap(Tregimen[temperature_index], all_states[idx][temperature_index],
	   	   "\$, \\, h_{\\mathrm{ex}} = $(round(hext_values[idx]; digits = 5)) \\, J \$" )

plot(plt1, plt2, plt3, plt4; 
	 layout = (2, 2), plot_title = "\$\\Delta h = $(Deltah) \\, J \$")

end

# ╔═╡ 0218e69f-db93-4f18-9fd9-29fcacdfb2f3
@bind temperature_index2 html"""<input value="1" type="range" min="1" max="24"/>"""


# ╔═╡ d96c579b-be81-4d03-834e-072d90c6e75f
let
idx = four_plot_indices[3]
plt1 = SA_heatmap(Tregimen[temperature_index2], all_states[idx][temperature_index2],
	   	   "\$, \\, h_{\\mathrm{ex}} = $(round(hext_values[idx]; digits = 5)) \\, J \$")
end

# ╔═╡ fcdb8c53-ff49-44ce-ab78-84048a27803c
md"""
## Decimation: real-space renormalization

Consider the image with dimensions $L_x \times L_y$. We seek to **`decimate`** it by replacing each $b_x \times b_y$ subimage with its average. For simplicity, we require that $b_i$ is an integer divisor of $L_i$. That will yield a total of 

```math
N_{\rm blocks} = \frac{L_x}{b_x} \cdot \frac{L_y}{b_y},
```

blocks. The renormalized image $\sigma^\prime$, with arguments $(i_b, j_b)$, is written in terms of the input image $\sigma$ as

```math
\sigma^\prime \left( i_b, j_b  \right) = \frac{1}{b_xb_y} \sum_{i = 1}^{b_x} \sum_{j = 1}^{b_y} \ \sigma \left[ i + (i_b - 1)b_x, j + (j_b - 1)b_y \right].
```

"""

# ╔═╡ 8d56ebaa-c76d-41e5-bc99-9a024d5b7d58
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

# ╔═╡ 1c931ac1-56af-4a67-84b7-7c4a9bc76b2a
md"""
### Renormalization flow
"""

# ╔═╡ f58dcf92-4474-44f3-9e23-3c2e52b9b7ad
@bind RG_b_power html"""<input value="3" type="range" min="0" max="6"/>"""

# ╔═╡ eecb0f44-e984-4ca1-abad-4f3f45afed66
let
hdx = 2
Tdx = length(Tregimen)
Tdx = 1
img = reshape(all_states[1][Tdx], Lvalue, Lvalue)
plt = heatmap(img; clims = (-1, 1), aspect_ratio = :equal,  
			  title = "\$ T = $(round(Tregimen[Tdx]; digits = 2))\\, J \$")
RGplt = heatmap( decimate(img; bx = 2^RG_b_power); 
				 clims = (-1, 1), aspect_ratio = :equal,
				 title = "\$ b_x = b_y = $(2^RG_b_power)\$")
plot(plt, RGplt; plot_title = "\$ h_{\\mathrm{ex}} = $(round(hext_values[hdx]; digits = 5)) \\, J \$")
end

# ╔═╡ 4713e228-d027-4846-bd0d-76fa99a637ba
md"""
### Fixed $b_x = b_y$ with variable temperature $T$
"""

# ╔═╡ 80061c86-0b05-4e20-9dd7-74fb7a8d71cc
@bind RG_Tdx html"""<input value="1" type="range" min="1" max="24"/>"""

# ╔═╡ d49c299e-4fd9-4097-b777-afcc079608e4
let
hdx = 3
Tdx = RG_Tdx
img = reshape(all_states[1][Tdx], Lvalue, Lvalue)
plt = heatmap(img; clims = (-1, 1), aspect_ratio = :equal,  
			  title = "\$ T = $(round(Tregimen[Tdx]; digits = 2))\\, J \$")
RGplt = heatmap( decimate(img; bx = 2^RG_b_power); 
				 clims = (-1, 1), aspect_ratio = :equal,
				 title = "\$ b_x = b_y = $(2^RG_b_power)\$")
plot(plt, RGplt; plot_title = "\$ h_{\\mathrm{ex}} = $(round(hext_values[hdx]; digits = 5)) \\, J \$")
end

# ╔═╡ bf091fd2-e561-49d5-b22d-306fe3fc1cea
md"""
### Renormalized spin distribution as a function of temperature
"""

# ╔═╡ b0c2ff6a-08e1-4e66-86ba-de15c809f083
function histogram_RG_state(state, blevel, title = false)
	img = reshape(state, Lvalue, Lvalue)
	RGimg = decimate(img; bx = blevel, by = blevel)
	histogram(RGimg[:]; label = false, normalize = false,
		  		xlabel = "Renormalized \$\\sigma \$ with \$ b_x = b_y = $blevel \$",
		  		xrange = (-1.2, 1.2),
				title = title,
				bins = LinRange(-1.05, 1.05, 15),
	)
	vline!([mean(RGimg[:])]; 
		   linestyle = :dash, 
	       # label = "Mean \$ \\sigma^\\prime \$",
		   label = false,
		   linewidth = 3, color = :orange)
end

# ╔═╡ af4576e0-9a6f-4336-b0b2-9eba036a1051
@bind RG_hist_Tdx html"""<input value="1" type="range" min="1" max="24"/>"""

# ╔═╡ 38cd5744-ba2e-4798-ad60-3300457983c4
let
Tdx = RG_hist_Tdx
blevel = 8

idx = four_plot_indices[1]
plt1 = histogram_RG_state(all_states[idx][Tdx], blevel,
					      "\$ h_{\\mathrm{ex}} / \\Delta h = $(round(bias_ratios[idx]; 										digits = 5))\$")

idx = four_plot_indices[2]
plt2 = histogram_RG_state(all_states[idx][Tdx], blevel,
					      "\$ h_{\\mathrm{ex}} / \\Delta h = $(round(bias_ratios[idx]; 										digits = 5))\$")

idx = four_plot_indices[3]
plt3 = histogram_RG_state(all_states[idx][Tdx], blevel,
					      "\$ h_{\\mathrm{ex}} / \\Delta h = $(round(bias_ratios[idx]; 										digits = 5))\$")

idx = four_plot_indices[4]
plt4 = histogram_RG_state(all_states[idx][Tdx], blevel,
					      "\$ h_{\\mathrm{ex}} / \\Delta h = $(round(bias_ratios[idx]; 										digits = 5))\$")

plot(plt1, plt2, plt3, plt4; plot_title = "\$ T = $(round(Tregimen[Tdx]; digits = 2))\\, J \$" )
end

# ╔═╡ e945de17-b127-4fe2-b1f6-defd0a2e1d5c
md"""
## First-Order Phase Transition tuned by $h_{\rm ex} / \Delta h$

When we consider the domain-excitation energy of $E_d(\ell)$, we can see that when $h_{\rm ex} = 0$, the uniform state is unstable towards domain formation. Meanwhile, when $\Delta h = 0$, the uniform ground state of the Ising model is preserved. These two states have a different symmetry in the thermodynamic limit. The latter has translation symmetry at the center of the Brillouin zone but breaks time-reversal symmetry, while the former breaks the translational symmetry while maintaining time-reversal symmetry. Thus, we expect a first-order phase transition to be tuned by the bias ratio $\eta \equiv h_{\rm ex} / \Delta h$ which is a measure of the strength of the external field to the random field strength.

We can obtain the expression for the critical value of the bias ratio $\eta_c$ by simultaneously solving the following equations for $\ell_c$ and $\eta_c$:

```math
\begin{aligned}
E_d(\ell) &= 0 = a \ell_c - b \ell_c \log \ell_c + c_c \ell_c^2,
\\
\left.\frac{\partial E_d}{\partial \ell}\right\vert_{\ell_c} &= 0 = a - b - b\log \ell + 2c_c \ell_c.
\end{aligned}
```

where $a = 8J$, $b = (8/\mathcal{C})(\Delta h^2 / J)$, and $c_c = 2 \eta_c \Delta h$. The solution is 

```math
\begin{aligned}
\ell_c &= \exp\left[ -1 - \mathcal{C}\left(\frac{ J}{\Delta h} \right)^2 \right],
\\
\eta_c &= \frac{4\Delta h}{\mathcal{C} J} \exp\left[ -1 - \mathcal{C}\left(\frac{ J}{\Delta h} \right)^2 \right]. 
\end{aligned}
```

Then, with $h_{\rm ex}^{(c)} = \eta_c \Delta h$, we see that for small random fields, $h_{\rm ex}^{(c)} \rightarrow 0$. Likewise, for strong random fields, the critical external field strength diverges quadratically with the disorder.
"""

# ╔═╡ 872e4941-f5d9-4100-bf76-d914cfc83958
hext_critical_point(J, Δh, C) = 4 * Δh^2 / (C * J) * exp( -(one(J) + C * J^2 / Δh^2) )

# ╔═╡ 4010b366-49d9-4988-89e4-9160f5b59ca6
hext_critical_point(J, Δh::AbstractArray, C) = [ hext_critical_point(J, dh, C) for dh ∈ Δh ]

# ╔═╡ 8b62b8b9-d14d-44e3-b4bc-7c4de63656bc
md"""
### Find the breakup length constant $\mathcal{C}$
"""

# ╔═╡ af3fc743-6c27-4e25-9b07-32c51b7a10fa
let
ising_model = RandomFieldIsingModel(Lvalue, Lvalue, Jex,
									0., Deltah)
flpath = MonteCarloMethods.SA_datapath(datadir("Agate"), ising_model, Tregimen[end])
state = load_state(flpath)
	
img = state
img = img .- mean(img)
img = reshape(img, Lvalue, Lvalue)

fft_img = fft(img)
cov_img = fftshift(abs.(ifft( abs2.(fft_img) )))
cov_img /= maximum(cov_img)

radii = Float64[rdx for rdx ∈ 1:Lvalue÷2]
field_of_points = [ sqrt((x - Lvalue ÷ 2)^2 + (y - Lvalue ÷ 2)^2) for x ∈ 1:Lvalue, y ∈ 1:Lvalue ]
avg_cov = similar(radii)
for (rdx, radius) ∈ enumerate(radii)
	avg_cov[rdx] = mean(cov_img[ radius .< field_of_points .< radius + one(radius) ])
end
plot(radii, avg_cov; label = false,
	 xlabel = "Radial separation", ylabel = "Azimuthally-averaged autocovariance")
min_rdx = argmin( abs2.(avg_cov .- exp(-one(eltype(avg_cov)))) )
vline!([radii[min_rdx]]; linestyle = :dash, linewidth = 3, color = :orange,
	   label = "\$ R_* = $(radii[min_rdx]) \$")
end

# ╔═╡ 3f71650d-63d7-4455-9d22-01cdd5c3800e
breakup_constant(Rstar, Δh, J) = (Δh / J)^2 * log(Rstar)

# ╔═╡ 19ad90fe-e071-4844-91bb-d348ce16eca5
Cbreakup = breakup_constant(14, Deltah, Jex)

# ╔═╡ 800e4a20-9210-444e-8023-652bb13ca772
hext_critical_point(Jex, Deltah, Cbreakup) / Deltah

# ╔═╡ 342b5bd7-f989-4318-a4a3-0e0428279da2
let
dhvals = 10.0 .^ LinRange(-1, 0.25, 100)
hext_c = hext_critical_point( Jex, dhvals, Cbreakup )
plot( dhvals, hext_c; 
	  xlabel = "\$ \\Delta h \$", ylabel = "\$ h_{\\mathrm{ex}}^{(c)} \$", 
      label = false, fillrange = minimum(hext_c), fillalpha = 0.25, 
      linestyle = :dash, linewidth = 2.5)
end

# ╔═╡ 5cdcecc8-c2ab-4fc7-8264-f32df3b818a0
md"""
### Domain distribution

The Imry-Ma-Binder domain energy above a uniform ground-state in the presence of a magnetic field is given by 

```math
E_d(\ell) = 8J\ell - \frac{8}{\mathcal{C}}\frac{\Delta h^2}{J} \ell \log \ell + 2 h_{\rm ex} \ell^2,
```

where $\mathcal{C}$ is a constant of order 1. One may estimate the distribution of domain sizes using the Boltzmann distribution as 

```math
\rho_d(\ell) = \exp\left( \theta - \frac{E_d(\ell)}{T} \right),
```

with the constant $\theta$ being defined by 

```math
{\rm e}^{-\theta} = \int_1^\infty {\rm d}\ell\, \exp\left( -\frac{E_d(\ell)}{T} \right).
```
"""

# ╔═╡ 1607040a-d64d-465a-b7d1-0bb317994e83
function domain_parameters(hex, J = Jex, Δh = Deltah, C = Cbreakup)
	return 8 * J, 8 / C * (Δh^2 / J), 2 * hex
end

# ╔═╡ 377722e3-9c4a-4a62-9309-f30252702998
function Edomain(x, hex, J = Jex, Δh = Deltah, C = Cbreakup)
	a, b, c = domain_parameters(hex, J, Δh, C)
	return @. a * x - b * x * log(x) + c * x^2 
end

# ╔═╡ 2cd860e9-3049-45dc-9b9b-1c772148bfe5
function Edomain_deriv(x, hex, J = Jex, Δh = Deltah, C = Cbreakup)
	a, b, c = domain_parameters(hex, J, Δh, C)
	one_num = one(a)
	return @. a - b * ( log(x) + one_num ) + 2 * c * x
end

# ╔═╡ 00c6756f-59f2-4750-b706-5a90c13f5196
function partition_domain(hex, Temp)
	Lvals, weights = gausslaguerre(32)
	Evals = Edomain(Lvals, hex)
	return dot( weights, @. exp(-Evals / Temp) * exp(Lvals))
end

# ╔═╡ acb3df2e-5139-496f-8e51-1780da7c23a0
function domain_dist(x, hex, Temp) 
	Evals = Edomain(x, hex)
	return @. exp( -Evals / Temp ) / partition_domain(hex, Temp)
end

# ╔═╡ 3d1d77ab-5950-40c0-b4f3-4b6f5c9a0537
md"""
### Plot the low-temperature domain distribution
"""

# ╔═╡ 02ce0000-7618-4480-9cf6-9a382f925767
begin
	num_dists = 13
	DdistString = "<input value=\"1\" type=\"range\" min=\"1\" max=\"$(num_dists)\"/>"
	@bind Ddist_hdx HTML(DdistString)
end

# ╔═╡ 82a7920c-6094-4157-a66b-c7e175f5d70a
begin
	num_temps = 20
	DdistTempString = "<input value=\"1\" type=\"range\" min=\"1\" max=\"$(num_temps)\"/>"
	@bind Ddist_Tdx HTML(DdistTempString)
end

# ╔═╡ 4e618f33-ec1d-4e4e-a874-f162a25e2c9a
let
hc = hext_critical_point(Jex, Deltah, Cbreakup)
hrange = hc
hvals = LinRange(hc - 0.05 * hrange, hc + 0.025 * hrange, num_dists)
Lvals = LinRange(0.00001, 70, 5000)
Tval = LinRange(0.1, 10, num_temps)[Ddist_Tdx]
plot(Lvals, domain_dist(Lvals, hvals[Ddist_hdx], Tval); label = false,
	 fillrange = 0, fillalpha = 0.25,
	 title = "\$ h_{\\mathrm{ex}} = $(round(hvals[Ddist_hdx]; digits = 5)) \\, J, \\quad T = $(round(Tval; digits = 3))\\, J \$",
	 xlabel = "Domain size \$ \\ell \$")
end

# ╔═╡ cc785a11-4e42-4917-a3c3-e495bfc24a4a
md"""
### Finite domain sizes that minimize $E_d(\ell)$
"""

# ╔═╡ bb6252da-2a9a-497b-9f7d-57ca978e3742
let
hc = hext_critical_point(Jex, Deltah, Cbreakup)
hvals = LinRange(0.1 * hc, 2 * hc, 100)
Lvals = similar(hvals)
for (hdx, hval) ∈ enumerate(hvals)
	val = zero(eltype(hvals))
	try val = find_zero(x -> Edomain_deriv(x, hval), 10_000)
	catch DomainError
		val = one(val)
	end
	Lvals[hdx] = val
end
plot(hvals, Lvals; markershape = :circle, yscale = :log10, label = false,
	 xlabel = "\$ h_{\\mathrm{ex}}\$",
	 ylabel = "Energy-minimizing domain sizes")
hc = hext_critical_point(Jex, Deltah, Cbreakup)
vline!([hc]; linestyle = :dash,
	    label = "\$ h_{\\mathrm{ex}}^{(c)} = $(round(hc; digits = 5))\\, J \$", linewidth = 3, color = :orange)
end

# ╔═╡ Cell order:
# ╠═2369d718-3b59-11ed-1243-7be9bca30bb0
# ╠═c57a8de6-2380-4cc2-afe2-25c6f77bd568
# ╠═399f3c2d-cc5b-4f03-ad03-62724185cc19
# ╠═7534377e-4604-4500-a60c-70d915f337ed
# ╠═5b0f17d5-67f2-43c8-9e1f-d4fec4c6db4e
# ╠═7aefb438-8661-4601-9848-5444389b72db
# ╠═7c482e32-dfec-4a64-a0c6-11071bb81671
# ╠═81e239d2-4fda-4402-9f28-21f1e477f1f5
# ╠═96190616-8bf8-44e3-b429-640607a156df
# ╠═9c75e7b2-1849-4299-bcf0-8838c1a719e2
# ╠═0f351599-49e9-4cf8-8d38-e93a05cf8154
# ╠═e6250092-b266-4f57-a344-eba07b562291
# ╠═da57e149-08d4-479c-8af1-5a79199508fb
# ╠═73b4aca5-e66e-41e8-9f3e-c5d4e3941734
# ╟─e462bdfd-d37d-4cb2-9481-1a019d2519ce
# ╟─85d92807-333e-4a6b-a48e-ae1b33989c87
# ╠═5be70f7a-2664-43d3-9d70-8e914190782a
# ╟─dff5bb0b-c185-4c54-b1f3-334876a7e766
# ╟─04f20fa1-ca64-4928-9dfc-f051cdc2733d
# ╟─0218e69f-db93-4f18-9fd9-29fcacdfb2f3
# ╟─d96c579b-be81-4d03-834e-072d90c6e75f
# ╟─fcdb8c53-ff49-44ce-ab78-84048a27803c
# ╠═61097b03-696d-40c5-801b-964a2bf885e3
# ╠═8d56ebaa-c76d-41e5-bc99-9a024d5b7d58
# ╟─1c931ac1-56af-4a67-84b7-7c4a9bc76b2a
# ╟─f58dcf92-4474-44f3-9e23-3c2e52b9b7ad
# ╟─eecb0f44-e984-4ca1-abad-4f3f45afed66
# ╟─4713e228-d027-4846-bd0d-76fa99a637ba
# ╟─80061c86-0b05-4e20-9dd7-74fb7a8d71cc
# ╠═d49c299e-4fd9-4097-b777-afcc079608e4
# ╟─bf091fd2-e561-49d5-b22d-306fe3fc1cea
# ╠═b0c2ff6a-08e1-4e66-86ba-de15c809f083
# ╟─af4576e0-9a6f-4336-b0b2-9eba036a1051
# ╟─38cd5744-ba2e-4798-ad60-3300457983c4
# ╟─e945de17-b127-4fe2-b1f6-defd0a2e1d5c
# ╠═872e4941-f5d9-4100-bf76-d914cfc83958
# ╠═800e4a20-9210-444e-8023-652bb13ca772
# ╠═4010b366-49d9-4988-89e4-9160f5b59ca6
# ╟─342b5bd7-f989-4318-a4a3-0e0428279da2
# ╟─8b62b8b9-d14d-44e3-b4bc-7c4de63656bc
# ╠═a66bee21-6b24-450b-b29a-4734d621d35e
# ╟─af3fc743-6c27-4e25-9b07-32c51b7a10fa
# ╠═3f71650d-63d7-4455-9d22-01cdd5c3800e
# ╠═19ad90fe-e071-4844-91bb-d348ce16eca5
# ╟─5cdcecc8-c2ab-4fc7-8264-f32df3b818a0
# ╠═f0a50e35-d4c7-4585-b2d6-8dc5b9cd90a9
# ╠═1607040a-d64d-465a-b7d1-0bb317994e83
# ╠═377722e3-9c4a-4a62-9309-f30252702998
# ╠═2cd860e9-3049-45dc-9b9b-1c772148bfe5
# ╠═00c6756f-59f2-4750-b706-5a90c13f5196
# ╠═acb3df2e-5139-496f-8e51-1780da7c23a0
# ╟─3d1d77ab-5950-40c0-b4f3-4b6f5c9a0537
# ╟─02ce0000-7618-4480-9cf6-9a382f925767
# ╟─82a7920c-6094-4157-a66b-c7e175f5d70a
# ╟─4e618f33-ec1d-4e4e-a874-f162a25e2c9a
# ╟─cc785a11-4e42-4917-a3c3-e495bfc24a4a
# ╠═73f6c72d-9995-4dad-87fa-eda04b7420a4
# ╟─bb6252da-2a9a-497b-9f7d-57ca978e3742
