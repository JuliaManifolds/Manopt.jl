### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 3e19b659-8855-4020-9173-fe4a1ffbbc08
using Pkg; Pkg.activate();

# ╔═╡ eb35384f-0822-4953-8d34-4ed5a0df71c5
begin
	using LinearAlgebra
	using RandomMatrices
	using Manopt
	using ManoptExamples
	using Manifolds
	using Random
	using RecursiveArrayTools
	using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ 587cf35f-882f-4e5a-8527-a53ba7c5644f
begin

    Random.seed!(56343658)

    N = 101

    # Eigenwerte vorgeben und orthogonal transformieren:

    #Diagonalmatrix mit Eigenwerten:
    eig_A = Matrix{Float64}(I, N, N)
    for i in 1:N
        eig_A[i, i] = i
    end

    # Zufällige orthogonale Matrix:
    O = rand(Haar(1),N)

    # Transformation von D:
    #A = O' * D * O
	
    for i in 1:N
		for j in 1:N
		if i > j
			eig_A[i,j] = 1.0
			end
			if i < j
			eig_A[i,j] = 1.0
		end
	end
	end
	B = zeros(N,N)
	for i in 1:N
		for j in 1:N
			if i > j
				B[i,j] = -1.0
			end
		end
	end
	println(eigvals(eig_A))
	println(eigvals(B))
end;

# ╔═╡ 1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
function f_prime(p)
    return (eig_A * p)'
end

# ╔═╡ cd8d3a84-c7bd-448b-9427-1140fbb8b7f1
function f_second_derivative(p)
    return eig_A
end

# ╔═╡ f66e335f-2472-430a-b134-e655f8278547
function vectortransport(M, q, p)
	return I - (1/norm(B*p)^2)*(B*p)*((B*p)')
end

# ╔═╡ e16a578c-3d64-11f0-057c-c1f978ba732a
begin
struct NewtonEquation{F, T, NM, Nrhs}
	f_prime::F
	f_second_prime::T
	A::NM
	b::Nrhs
end

function NewtonEquation(M, f_pr, f_sp)
	A = zeros(N+1,N+1)
	b = zeros(N+1)
	return NewtonEquation{typeof(f_pr), typeof(f_sp), typeof(A), typeof(b)}(f_pr, f_sp, A, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
    ne.A .= hcat(vcat(ne.f_second_prime(p.x[1]) - 1/(norm(B*p.x[1])^2) * (ne.f_prime(p.x[1])*(B*p.x[1])*B), p.x[1]'), vcat(-B*p.x[1], 0))
    ne.b .= vcat(ne.f_prime(p.x[1])' - p.x[2].*(B*p.x[1]), 0)
end
	
function (ne::NewtonEquation)(M, VB, p, p_trial)
	rhs_p_trial = ne.f_prime(p_trial.x[1])' - p_trial.x[2].*(B*p_trial.x[1])
    return vcat(vectortransport(M[1], p.x[1], p_trial.x[1])'*rhs_p_trial, 0)
end
end;

# ╔═╡ 7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
function solve_augmented_system(problem, newtonstate) 
	res = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return ArrayPartition(res[1:N], fill(res[N+1]))
end

# ╔═╡ 3232de6d-7b17-4f4c-9a3f-c669a1911c57
begin
	y0 = zeros(N)
	y0[1] = 1.0
	λ0 = 0.9
	# compute first estimate for Lagrangian as solution of linear system
	
	#A0 = hcat(vcat(f_second_derivative(y0) - 1/(norm(B*y0)^2) * (f_prime(y0)*(B*y0)*B), y0'), vcat(-B*y0, 0))
	#b0 = vcat(f_prime(y0)', 0)
	#res0 = (A0 \ b0)
	#λ0 = res0[N+1]

	#x0 = ArrayPartition(retract(Manifolds.Sphere(N-1),y0, res0[1:N]) , fill(λ0))
	x0 = ArrayPartition(y0, fill(λ0))
end;

# ╔═╡ 68ac302f-6362-4a99-92ab-76c56f40a4c0
M = ProductManifold(Manifolds.Sphere(N-1), Manifolds.Euclidean(;))

# ╔═╡ 576b004a-5ec0-4435-b6a8-1bcf9ea36b65
begin
	# adjust norms for computation of damping factors and stopping criterion
	pr_inv = Manifolds.InverseProductRetraction( ProjectionInverseRetraction(), LogarithmicInverseRetraction())
	rec = RecordChange(M;
    inverse_retraction_method=pr_inv);
end;

# ╔═╡ 427233d0-567d-4cac-930d-7c80178f2d09
begin
	NE = NewtonEquation(M, f_prime, f_second_derivative)
	
	st_res = vectorbundle_newton(M, TangentBundle(M), NE, x0; sub_problem=solve_augmented_system, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(M,1e-12,outer_norm=Inf)),
	retraction_method=ProductRetraction(ProjectionRetraction(), ExponentialRetraction()),
	stepsize=Manopt.AffineCovariantStepsize(M, theta_des=0.4, outer_norm=Inf),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, rec => :Change, :Stepsize],
	return_state=true
)
end

# ╔═╡ 66bb735a-a41d-4742-8581-fe48907b7489
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	stepsizes = get_record(st_res, :Iteration, :Stepsize)[1:end]
	res = get_record(st_res, :Iteration, :Iterate)[end]
end;

# ╔═╡ 285c7cdc-fdc7-43e8-99fa-17fd708d4328
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change[1:end], color = :blue)
	f
end

# ╔═╡ fd2bc12f-3287-405f-b42a-2696c7b30282
begin
	f_st = Figure(;)
	
    row_st, col_st = fldmod1(1, 2)
	
	Axis(f_st[row_st, col_st], title = string("Stepsizes"), xminorgridvisible = true, xticks = (1:length(stepsizes)), xlabel = "Iteration", ylabel = "α")
    scatterlines!(stepsizes[1:end-1], color = :blue)
	f_st
end

# ╔═╡ a7a13b69-dd2c-4f24-989d-dc3356633184
norm(eig_A*res.x[1] - res.x[2].*B*res.x[1])

# ╔═╡ Cell order:
# ╠═3e19b659-8855-4020-9173-fe4a1ffbbc08
# ╠═eb35384f-0822-4953-8d34-4ed5a0df71c5
# ╠═587cf35f-882f-4e5a-8527-a53ba7c5644f
# ╠═1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
# ╠═cd8d3a84-c7bd-448b-9427-1140fbb8b7f1
# ╠═f66e335f-2472-430a-b134-e655f8278547
# ╠═e16a578c-3d64-11f0-057c-c1f978ba732a
# ╠═7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
# ╠═3232de6d-7b17-4f4c-9a3f-c669a1911c57
# ╠═427233d0-567d-4cac-930d-7c80178f2d09
# ╠═68ac302f-6362-4a99-92ab-76c56f40a4c0
# ╠═576b004a-5ec0-4435-b6a8-1bcf9ea36b65
# ╠═66bb735a-a41d-4742-8581-fe48907b7489
# ╠═285c7cdc-fdc7-43e8-99fa-17fd708d4328
# ╠═fd2bc12f-3287-405f-b42a-2696c7b30282
# ╠═a7a13b69-dd2c-4f24-989d-dc3356633184
