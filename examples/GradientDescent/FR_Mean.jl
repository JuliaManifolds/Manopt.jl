using Manifolds, Manopt, Random
Random.seed!(42)
m=10
n=10
k=4
G = Gressmann(n,k)
pts = [
    SVDMPoint(random_point(G), Diagonal(1 .+ randn(4)), random_point(G)', k)
    for _ = 1:10
    ]
F(M::FixedRankMatrices, p) = sum([ 0.5*distance(M, p, q)^2 for q ∈ pts ])
gradF(M::FixedRankMatrices, p) = sum([ log(M, p, q) for q in pts ])
M = FixedRankMatrices(m,n,k)
x0 = deepcopy(pts[1])
gradient_descent(M,F,gradF,x0)