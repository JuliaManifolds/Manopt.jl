### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 4a1dd24e-52c6-11ec-20a6-3d677b47806f
begin
	using Pkg
	Pkg.activate() # use global enrivonment of current Julia, esp Manopt in dev mode for now
	using Manifolds, Manopt, LinearAlgebra
end


# ╔═╡ 607c1fbf-0b58-4dc0-a0eb-1a682f97b4f7
md"""
# Using Primal-Dual Riemannian Semismooth Newton for ``S^2``-valued data
"""

# ╔═╡ 4522df74-6c30-4dde-bb3e-10d46be06c6c
begin
	experiment_name = "S2_Signal_TV_PDRSSN"
	export_orig = true
	export_primal = false
	export_table = false
	use_debug = true
	#
	# Automatic Script Settings
	current_folder = @__DIR__
	export_any = export_orig || export_primal || export_table
	results_folder = joinpath(current_folder, "Signal_TV")
	# Create folder if we have an export
	(export_any && !isdir(results_folder)) && mkdir(results_folder)

end

# ╔═╡ 7d3d39db-6791-4565-94c8-18a1959e16c6
md"""
# Settings
"""

# ╔═╡ 830dff11-07d0-4ba3-9de1-f35413c98cc2
signal_section_size = 4

# ╔═╡ 44b78430-91d4-4157-8431-f6cb0ad7aae8
α = 5.0

# ╔═╡ 68f01f9d-e5c4-45f4-aa75-1560d1948b54
σ = 0.5

# ╔═╡ ee68439a-1ce8-4f74-b721-500859eca196
τ = 0.5

# ╔═╡ 7a0460f1-afe5-46dc-a425-53616f4fb250
max_iterations = 50

# ╔═╡ 51f3b7b2-c9a7-42c6-a4bc-88a446aeb967
pixelM = Sphere(2)

# ╔═╡ 621077a4-2b6e-4e3c-8b51-ee3613d9c639
md"""
# Generate a signal
"""

# ╔═╡ a0738ce7-46d1-46ad-8c48-85d600a1c5de
base = [1.0, 0.0, 0.0]

# ╔═╡ 555fc740-9434-42c8-a647-66512adc8cff
X = π / 4 * [0.0, 1.0, 0.0]

# ╔═╡ 9b5f4ed8-b91f-45a7-ade0-d3828d7c0e0f
p1 = exp(pixelM, base, X)

# ╔═╡ 894a1d46-daae-4f15-b26e-dbf61a8a3e97
p2 = exp(pixelM, base, -X)

# ╔═╡ c4dbabb1-2b13-49de-81d6-be914a3e4ac5
f = vcat(fill(p1, signal_section_size), fill(p2, signal_section_size))

# ╔═╡ 95d6a845-9050-4bfe-b004-a8603c8271b8
md"""
# Model TV for a given pixelManifold
"""

# ╔═╡ 3ec756a4-491e-41f6-9e66-2616971a81ad
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d

# ╔═╡ b703566e-aba2-4d7d-8982-d258ec44c837
res(s::Int) = res([s])

# ╔═╡ b69bf0b3-94c5-447b-9ff6-fdb4e37876fc
res(s) = (length(s) > 1) ? [s..., length(s)] : s

# ╔═╡ 94f094e0-b59c-4f5f-8361-f4ce259b3cba
d = length(size(f))

# ╔═╡ a1518739-d0ca-4adb-92bd-e3e619da6171
s = size(f)

# ╔═╡ 7392be70-49df-4a22-9a8b-2a17ec0ebbb9
m = fill(base, size(f))

# ╔═╡ cd3495c6-a9ad-4876-bb55-211b65ce95de
M = PowerManifold(pixelM, NestedPowerRepresentation(), s...)

# ╔═╡ b83e5c79-4d58-4545-bbe6-2ded810ae9cd
M2 = PowerManifold(pixelM, NestedPowerRepresentation(), res(s)...)

# ╔═╡ fe6c4b15-1ec6-42a1-8774-c71bda764f31
m2(m) = repeat(m; inner=rep(length(size(m))))

# ╔═╡ 89c8b755-80fe-4d73-a694-42fd942a38b4
n = m2(m)

# ╔═╡ 8f53295a-7dc1-43fc-921c-2fa7093c056a
N = TangentSpaceAtPoint(M2,m2(m))

# ╔═╡ 34efcf04-cb4b-41f8-b31a-e6f29c848976
md"""
Remark: We actually should take ``N = TM`` being the tangent bundle. However, in the end we will be considering only the second part of ``T_nTM = T_{m2}M \times T_{m2}M``, i.e,  we only focus on ``T_{m2}M``. 

In practice this makes computations easier (a smaller Newton matrix) for the TV case.

Hence, with slight abuse of notation we use ``N = T_{m2}M`` and ``n = m2``
"""

# ╔═╡ 6055628d-bf1d-4a6e-ba05-4e8b63cebfb4
md"""
# Build TV functionals
"""

# ╔═╡ 927eb419-1bd2-45d6-a179-4106d8927abe
md"""
Define cost, i.e., ``F(p) + G(Λ(p))``
"""

# ╔═╡ a40ec11b-ba19-4c81-bfef-0c999b5af4ca
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2

# ╔═╡ 25323d12-c116-4edd-aa7c-0082e446d3eb
function Λ(M, x)
    return forward_logs(M, x) # on N=TM, namely in T_xM
end

# ╔═╡ 4182adef-3acc-457c-9276-161505e94fea
md"""
TODO in this TV example we only care about the base point of ``Λ(m)``, i.e, ``m2``. Should we just define it like that here?
"""

# ╔═╡ c436686e-2014-4b7a-aa03-63e2b128ce62
function prior(M, x)
    # inner 2-norm over logs, 1-norm over the pixel
    return norm(norm.(Ref(pixelM), x, Λ(M, x)), 1)
end

# ╔═╡ 8cfbac64-a470-400b-9406-89e84f6064d3
cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)

# ╔═╡ 252b65b4-d8e6-4600-b16f-87823d420f04
md"""
# Compute exact minimiser
"""

# ╔═╡ 471c3da2-9271-4ce2-847b-47741c542814
jump_height = distance(pixelM, f[signal_section_size], f[signal_section_size + 1])

# ╔═╡ 1eeb4c33-318c-47af-b214-08ffd1274715
δ = min(α / (signal_section_size * jump_height) , 1 / 2)

# ╔═╡ 2272ee4a-8c63-484e-ab88-104ada7e90b8
x_hat = shortest_geodesic(M, f, reverse(f), δ)

# ╔═╡ 685bb279-fcab-479d-bdc7-8d7c54be90f9
if export_orig
    orig_file = joinpath(results_folder, experiment_name * "-original.asy")
    asymptote_export_S2_data(orig_file; data=f)
    render_asymptote(orig_file)
end

# ╔═╡ 5dd23632-1006-498c-8052-870a7627b4ed
md"""
# Define mappings for PD-RSSN
"""

# ╔═╡ 50a669cd-968e-4743-a33b-27886854f5d1
md"""
2) Define ``\operatorname{prox}_{σF}``
"""

# ╔═╡ d991a086-1b54-4a60-9d36-44db0d99edb5
proxFidelity(M, λ, x) = prox_distance(M, λ / α, f, x, 2)

# ╔═╡ 8677325a-537e-4d17-8eab-f282b2892eaa
md"""
3) Define ``D_p\operatorname{prox}_{σF} ``
"""

# ╔═╡ 0666a26a-e8ee-4685-a8e6-82d7f70e5a92
diff_proxFidelity(M,λ,x,η) = differential_geodesic_startpoint(M,x,f,λ/(α + λ),η)

# ╔═╡ 14654b8f-4049-44cf-a361-3667fc27077c
md"""
4) Define ``\operatorname{prox}_{τ G^*}``
"""

# ╔═╡ 8459a30e-4072-466d-a9b8-7835f0b9ad88
function proxPriorDual(N, n, λ, ξ)
    return project_collaborative_TV(
            base_manifold(N), λ, n, ξ, Inf, Inf # @Ronny shouldnt we use 2 for the first inf?
        )
end

# ╔═╡ 76873d3e-eb5f-49ec-966d-3d6331987c07
md"""
5) Define ``D_C \operatorname{prox}_{τ G^*}``
"""

# ╔═╡ 146892f8-f8a7-4c11-ae72-0015613af7ce
γ = 0. # dual_regularisation

# ╔═╡ 6caf34cc-6b37-43b8-826f-5376a95e1909
isotropic=false

# ╔═╡ a2b8404a-b6ac-42f4-b75f-c5a58d28d314
function CdifferentialRegularizedproxPriorDual(N, n, λ, ξ, η; γ=0,isotropic=false)
	M2 = base_manifold(N)
    m2 = n
    power_size = power_dimensions(M2)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I

    J = zero_vector(N, n)
    
	if !isotropic || d==1
		for j in R
			for k in 1:d
				mⱼ = m2[M2, j..., k]
				ηⱼ = η[N, j..., k]
				g = norm(M2.manifold,mⱼ,ηⱼ)
				if !(j[k]==maxInd[k])
					ξⱼ = ξ[N, j..., k]
					if g/(1+λ*γ) <=1
						J[N, j..., k] +=  ξⱼ/(1+λ*γ)
					else
						J[N, j..., k] +=  1/g * (ξⱼ - 1/g^2 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*ηⱼ)
					end
				else
					# Taking care of the boundary equations
					# J[N, j... ,k] = zero_vector(M2.manifold,mⱼ)
				end
			end
		end
	else
		for j in R
			g = norm(M2.manifold, m2[M2, j..., 1], η[N, j..., :])
			for k in 1:d
				mⱼ = m2[M2, j..., k]
				ηⱼ = η[N, j..., k]
				if !(j[k]==maxInd[k])

					ξⱼ = ξ[N, j..., k]
					if g/(1+λ* γ) <=1
						J[N, j..., k] +=  ξⱼ/(1+λ* γ)
					else
						for κ in 1:d
							if κ != k
								J[N, j..., κ] += - 1/g^3 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*η[j,κ]
							else
								J[N, j..., k] += 1/g * (ξⱼ - 1/g^2 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*ηⱼ)
							end
						end
					end
				else
					# Taking care of the boundary equations
					# J[N, j... ,k] = zero_vector(M2.manifold,mⱼ)
				end
			end
		end
	end

    return J
end

# ╔═╡ 059fa57a-36f0-4445-bb8b-ce6e0e0232a9
diff_proxPriorDual(N, n, λ, ξ, η) = CdifferentialRegularizedproxPriorDual(N, n, λ, ξ, η,γ=γ, isotropic=isotropic)

# ╔═╡ bc3b91db-5275-42ac-a275-2973ac051f8a
function DΛ(M, m, ξm)
    return ProductRepr(
        repeat(ξm; inner=rep(length(size(m)))), differential_forward_logs(M, m, ξm)
    )
end

# ╔═╡ 2ba7238b-a364-479b-8a16-74f2a5f8fb4b
AdjDΛ(N, m, n, ξ) = adjoint_differential_forward_logs(M, m, ξ)

# ╔═╡ 5f6b0ebd-258a-48c6-b854-384f6eaf70b0
x0 = deepcopy(f)

# ╔═╡ 7cbf60b3-10d7-423e-afce-60fed311b9b6
ξ0 = zero_vector(N, n)

# ╔═╡ 281dbac5-5ec7-4908-8f4c-fd2aed6b9f8e
o = primal_dual_semismooth_Newton(
    M,
    N,
    cost,
    x0,
    ξ0,
    m,
    n,
    proxFidelity,
	diff_proxFidelity,
    proxPriorDual,
	diff_proxPriorDual,
	DΛ,
    AdjDΛ;
    primal_stepsize=σ,
    dual_stepsize=τ,
    debug=if use_debug
        [
            :Iteration,
            " ",
            DebugPrimalChange(),
            " | ",
            :Cost,
            "\n",
            100,
            :Stop,
        ]
    else
        missing
    end,
    record=if export_table
        [:Iteration, RecordPrimalChange(x0), RecordDualChange((ξ0, n)), :Cost]
    else
        missing
    end,
    stopping_criterion=StopAfterIteration(max_iterations),
    return_options=true,
)

# ╔═╡ Cell order:
# ╟─607c1fbf-0b58-4dc0-a0eb-1a682f97b4f7
# ╠═4a1dd24e-52c6-11ec-20a6-3d677b47806f
# ╠═4522df74-6c30-4dde-bb3e-10d46be06c6c
# ╟─7d3d39db-6791-4565-94c8-18a1959e16c6
# ╟─830dff11-07d0-4ba3-9de1-f35413c98cc2
# ╟─44b78430-91d4-4157-8431-f6cb0ad7aae8
# ╟─68f01f9d-e5c4-45f4-aa75-1560d1948b54
# ╟─ee68439a-1ce8-4f74-b721-500859eca196
# ╟─7a0460f1-afe5-46dc-a425-53616f4fb250
# ╟─51f3b7b2-c9a7-42c6-a4bc-88a446aeb967
# ╟─621077a4-2b6e-4e3c-8b51-ee3613d9c639
# ╟─a0738ce7-46d1-46ad-8c48-85d600a1c5de
# ╟─555fc740-9434-42c8-a647-66512adc8cff
# ╟─9b5f4ed8-b91f-45a7-ade0-d3828d7c0e0f
# ╟─894a1d46-daae-4f15-b26e-dbf61a8a3e97
# ╟─c4dbabb1-2b13-49de-81d6-be914a3e4ac5
# ╟─95d6a845-9050-4bfe-b004-a8603c8271b8
# ╟─3ec756a4-491e-41f6-9e66-2616971a81ad
# ╟─b703566e-aba2-4d7d-8982-d258ec44c837
# ╟─b69bf0b3-94c5-447b-9ff6-fdb4e37876fc
# ╟─94f094e0-b59c-4f5f-8361-f4ce259b3cba
# ╟─a1518739-d0ca-4adb-92bd-e3e619da6171
# ╟─7392be70-49df-4a22-9a8b-2a17ec0ebbb9
# ╟─cd3495c6-a9ad-4876-bb55-211b65ce95de
# ╟─b83e5c79-4d58-4545-bbe6-2ded810ae9cd
# ╟─fe6c4b15-1ec6-42a1-8774-c71bda764f31
# ╟─89c8b755-80fe-4d73-a694-42fd942a38b4
# ╟─8f53295a-7dc1-43fc-921c-2fa7093c056a
# ╟─34efcf04-cb4b-41f8-b31a-e6f29c848976
# ╟─6055628d-bf1d-4a6e-ba05-4e8b63cebfb4
# ╟─927eb419-1bd2-45d6-a179-4106d8927abe
# ╟─a40ec11b-ba19-4c81-bfef-0c999b5af4ca
# ╟─25323d12-c116-4edd-aa7c-0082e446d3eb
# ╟─4182adef-3acc-457c-9276-161505e94fea
# ╟─c436686e-2014-4b7a-aa03-63e2b128ce62
# ╟─8cfbac64-a470-400b-9406-89e84f6064d3
# ╟─252b65b4-d8e6-4600-b16f-87823d420f04
# ╟─471c3da2-9271-4ce2-847b-47741c542814
# ╟─1eeb4c33-318c-47af-b214-08ffd1274715
# ╟─2272ee4a-8c63-484e-ab88-104ada7e90b8
# ╟─685bb279-fcab-479d-bdc7-8d7c54be90f9
# ╟─5dd23632-1006-498c-8052-870a7627b4ed
# ╟─50a669cd-968e-4743-a33b-27886854f5d1
# ╟─d991a086-1b54-4a60-9d36-44db0d99edb5
# ╟─8677325a-537e-4d17-8eab-f282b2892eaa
# ╟─0666a26a-e8ee-4685-a8e6-82d7f70e5a92
# ╟─14654b8f-4049-44cf-a361-3667fc27077c
# ╟─8459a30e-4072-466d-a9b8-7835f0b9ad88
# ╟─76873d3e-eb5f-49ec-966d-3d6331987c07
# ╟─146892f8-f8a7-4c11-ae72-0015613af7ce
# ╟─6caf34cc-6b37-43b8-826f-5376a95e1909
# ╟─a2b8404a-b6ac-42f4-b75f-c5a58d28d314
# ╟─059fa57a-36f0-4445-bb8b-ce6e0e0232a9
# ╟─bc3b91db-5275-42ac-a275-2973ac051f8a
# ╠═2ba7238b-a364-479b-8a16-74f2a5f8fb4b
# ╠═5f6b0ebd-258a-48c6-b854-384f6eaf70b0
# ╠═7cbf60b3-10d7-423e-afce-60fed311b9b6
# ╠═281dbac5-5ec7-4908-8f4c-fd2aed6b9f8e
