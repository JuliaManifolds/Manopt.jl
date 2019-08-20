#
#   SVD decomposition of a matrix truncated to a rank
#
using Manopt
export truncated_svd

@doc doc"""
    truncated_svd(A, p)

# Input
* `A` – a real-valued matrix A of size mxn
* `p` – an integer $p \leq min\{ m, n \}$

# Output
* `U` – an orthonormal matrix of size mxp
* `V` – an orthonormal matrix of size nxp
* `S` – a diagonal matrix of size pxp with nonnegative and decreasing diagonal
        entries
"""

function truncated_svd(A::Array{Float64,2} = randn(42, 60), p::Int64 = 5)
    (m, n) = size(A)

    if p > min(m,n)
        throw( ErrorException("The Rank p=$p must be smaller than the smallest dimension of A = $min(m, n).") )
    end

    U = Grassmannian(m, p)
    V = Grassmannian(n, p)

    prod = [U, V]

    M = Product(prod)

    function cost(X)
        U = X.U;
        V = X.V;
        f = -.5*norm(U'*A*V, 'fro')^2;
    end

    function egrad(X)
        U = X.U;
        V = X.V;
        AV = A*V;
        AtU = A'*U;
        g.U = -AV*(AV'*U);
        g.V = -AtU*(AtU'*V);
    end

    function ehess(X, H)
        U = X.U;
        V = X.V;
        Udot = H.U;
        Vdot = H.V;
        AV = A*V;
        AtU = A'*U;
        AVdot = A*Vdot;
        AtUdot = A'*Udot;
        h.U = -(AVdot*AV'*U + AV*AVdot'*U + AV*AV'*Udot);
        h.V = -(AtUdot*AtU'*V + AtU*AtUdot'*V + AtU*AtU'*Vdot);
    end





    options.Delta_bar = 4*sqrt(2*p);
    [X, Xcost, info] = trustregions(problem, [], options); %#ok<ASGLU>
    U = X.U;
    V = X.V;


    Spp = U'*A*V;
    [Upp, Spp, Vpp] = svd(Spp);
    U = U*Upp;
    S = Spp;
    V = V*Vpp;


    if M.dim() < 512
        evs = hessianspectrum(problem, X);
        stairs(sort(evs));
        title(['Eigenvalues of the Hessian of the cost function ' ...
               'at the solution']);
    end

end
