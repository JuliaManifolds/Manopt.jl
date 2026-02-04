using Manopt, Manifolds, LinearAlgebra, Test

Rxy(α) = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0; 0 0 1]
Rxz(α) = [cos(α)  0.0 sin(α); 0 1 0; -sin(α) 0 cos(α)]
Ryz(α) = [1.0 0 0; 0 cos(α) sin(α); 0 -sin(α) cos(α)]

M = Rotations(3)

# We generate a set of points that are “opposite” each other such that the mean is still I
pts = [
    Matrix{Float64}(I, 3,3),
    Rxy(0.25)*Rxz(0.05)*Ryz(-0.125),
    Rxy(-0.25)*Rxz(-0.05)*Ryz(0.125),
    Rxy(-0.05)*Rxz(0.125)*Ryz(-0.25),
    Rxy(0.05)*Rxz(-0.125)*Ryz(0.25),
    #outliers
    #Rxy(0.125)*Rxz(0.25)*Ryz(0.05),
    #Rxy(-0.125)*Rxz(0.25)*Ryz(0.05),
]
p0 = copy(M, pts[2])
# We do a full function approach here

F(M, p) = [distance(M, p, q) for q in pts]
function JF(M, p, B::AbstractBasis = DefaultOrthonormalBasis())
    d = manifold_dimension(M)
    n = length(pts)
    J = zeros(n,d)
    for (i,q) in enumerate(pts)
        dpq = distance(M, p,q)
        if dpq > 0
            J[i,:] = -1/dpq .* get_coordinates(M, p, log(M, p, q), B)
        end
    end
    return J
end

f = VectorGradientFunction(F, JF, length(pts);
    evaluation = AllocatingEvaluation(),
    function_type = FunctionVectorialType(),
    jacobian_type = CoordinateVectorialType(DefaultOrthonormalBasis())
)

qc = mean(M, pts)
cost(M, p) = sum(distance(M, p, q)^2 for q in pts)

q1 = LevenbergMarquardt(
    M, [f,], p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5,
    robustifier = [IdentityRobustifier(),],
    debug = [:Iteration, :Cost, " ", :damping_term, " ", :Iterate, "\n"],
)