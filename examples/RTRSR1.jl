### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d3df4e1c-bd58-4e13-99cf-7758b83c689e
begin
    using Pkg
    Pkg.activate()

    using Manifolds, Manopt
end

# ╔═╡ b24e567a-3ea3-441f-bbbd-acf0c12e9c20
md"""
# Riemannian Trust-Region Symmetric Rank-One Method

We will first consider Trust-Region methods on Riemannian manifolds in
general. To do this, we must construct a quadratic model of our
objective function $f$, which is defined on a Riemannian manifold
$\mathcal{M}$. But this is not, in general, a Euclidean space, which
makes the situation quite difficult. To avoid this problem, we choose a
retraction, $\operatorname{retr}_{\cdot} \left( \cdot \right)$, which provides a way to pull back the
objective function on the manifold $\mathcal{M}$ to a objective function
on the tangent space $\mathcal{T}_{x_k} \mathcal{M}$ at each iterate.\
Given a objective function $f \colon \ \mathcal{M} \to \mathbb{R}$ and a
current iterate $x_k \in \mathcal{M}$, we use $\operatorname{retr}_{x_k}$ to locally
map the minimization problem for $f$ on $\mathcal{M}$ into a
minimization problem for the pullback $\hat{f}_{x_k}$ of $f$ under
$\operatorname{retr}_{x_k}$ on the tangent space $\mathcal{T}_{x_k} \mathcal{M}$, which means in
each iteration we consider
$$\min \hat{f}_{x_k} (\xi_{x_k}) = f(\operatorname{retr}_{x_k}(\xi_{x_k})), \quad \xi_{x_k} \in \mathcal{T}_{x_k} \mathcal{M}.$$
The Riemannian metric turns the tangent space $\mathcal{T}_{x_k} \mathcal{M}$ into a
Euclidean space endowed with the inner product $g_{x_k}(\cdot, \cdot)$,
which enables us to construct a order-2 model of the pullback
$\hat{f}_{x_k}$: $$\begin{aligned}
    \hat{m}_k( s ) & = \hat{f}_{x_k}(0_{x_k}) + \mathrm{D} \, \hat{f}_{x_k}(0_{x_k}) [s] + \frac{1}{2} \mathrm{D}^2 \, \hat{f}_{x_k}(0_{x_k}) [s, s] \\
    & = f(x_k) + g_{x_k}(\operatorname{grad} f(x_k), s ) + \frac{1}{2} g_{x_k}( s, \operatorname{Hess} \hat{f}_{x_k}(0_{x_k}) [s]),\end{aligned}$$
using $\hat{f}_{x_k}(0_{x_k}) = f(\operatorname{retr}_{x_k}(0_{x_k})) = f(x_k)$ (see
[\[Retraction\]](#Retraction){reference-type="ref"
reference="Retraction"}) and the property
[\[PullbackGradient\]](#PullbackGradient){reference-type="ref"
reference="PullbackGradient"} since the local rigidity condition,
[\[LocalRigidity\]](#LocalRigidity){reference-type="ref"
reference="LocalRigidity"}, holds.\
To get a model of $f$, we "push forward" the model $\hat{m}_k(s)$
through the retraction $\operatorname{retr}_{x_k}$, i.e. $$\label{RiemannianModel}
    m_k = \hat{m}_k \circ \inverseRetract{x_k}$$ with
$$\label{RiemannianModelTangent}
    \hat{m}_k( s ) = f(x_k) + g_{x_k}(\operatorname{grad} f(x_k), s ) + \frac{1}{2} g_{x_k}( s, \operatorname{Hess} f(x_k) [s]),$$
where the quadratic term is given by the Riemannian Hessian of the
objective function $f$.\
In general, the model
[\[RiemannianModel\]](#RiemannianModel){reference-type="ref"
reference="RiemannianModel"} (with
[\[RiemannianModelTangent\]](#RiemannianModelTangent){reference-type="ref"
reference="RiemannianModelTangent"}) of the objective function $f$ is of
order 1 because
$\operatorname{Hess} \hat{f}_{x_k}(0_{x_k}) \neq \operatorname{Hess} f(x_k)$.
Since the Hessian operator of the pullback depends among other things on
the chosen retraction, further conditions must be imposed on the
retraction so that this equality holds (see [@AbsilMahonySepulchre:2008
Proposition 5.5.5]). However, for any retraction,
$\operatorname{Hess} \hat{f}_{x^*}(0_{x^*}) = \operatorname{Hess} f(x^*)$
holds, when $x^*$ is a stationary point of $f$, which means
$\operatorname{grad} f(x^*) = 0_{x^*}$ [@AbsilMahonySepulchre:2008
p. 138-139]. Nevertheless, if we would continue with
[\[RiemannianModelTangent\]](#RiemannianModelTangent){reference-type="ref"
reference="RiemannianModelTangent"}, we would get the so-called
Riemannian Trust-Region Newton, short RTR-Newton, method.\
For now we just consider a quadratic model of the pullback
$\hat{f}_{x_k}$: $$\label{RiemannianQuadraticModel}
    \hat{m}_k( s ) = f(x_k) + g_{x_k}(\operatorname{grad} f(x_k), s) + \frac{1}{2} g_{x_k}( s, \mathcal{H}_k [s]),$$
where $\mathcal{H}_k$ is some linear self-adjoint operator on
$\mathcal{T}_{x_k} \mathcal{M}$. This model
[\[RiemannianQuadraticModel\]](#RiemannianQuadraticModel){reference-type="ref"
reference="RiemannianQuadraticModel"} is easier to handle than the
pullback $\hat{f}_{x_k}$ and its purpose is to approximate
$\hat{f}_{x_k}$ within a suitable neighbourhood of the zero tangent
vector $0_{x_k} \in \mathcal{T}_{x_k} \mathcal{M}$ which we refer to as the
trust-region. By using the norm $\lVert \cdot \rVert_{x_k}$ induced by
the Riemannian metric, we can define the trust-region as
$$\label{RiemannianTrustRegion}
    \{ s \in \mathcal{T}_{x_k} \mathcal{M} \colon \ \lVert s \rVert_{x_k} \leq \Delta_k \},$$
where $\Delta_k > 0$ is the trust-region radius. We point out that this
trust-region now consists not of all points on the manifold which have
the Riemannian distance less or equal $\Delta_k$ to the current iterate
$x_k$, but of all tangent vectors in tangent space at the current
iterate $\mathcal{T}_{x_k} \mathcal{M}$ whose length is less or equal $\Delta_k$. The
chosen retraction, $\operatorname{retr}_{\cdot} \left( \cdot \right)$, defines for any iterate
$x_k \in \mathcal{M}$, a one-to-one correspondence $\operatorname{retr}_{x_k}$
between a neighborhood of $x_k$ in $\mathcal{M}$ and a neighborhood of
$0_{x_k}$ in the tangent space $\mathcal{T}_{x_k} \mathcal{M}$ (we remember
$\operatorname{retr}_{x_k}(0_{x_k}) = x_k$) [@AbsilBakerGallivan:2007 p. 304]. But
the chosen retraction, $\operatorname{retr}_{\cdot} \left( \cdot \right)$, applied to a tangent vector
$\xi_{x_k} \in \mathcal{T}_{x_k} \mathcal{M}$ does not in general lead to a point
$\operatorname{retr}_{x_k}(\xi_{x_k})$ whose distance to the starting point $x_k$ is
equal to the norm of the tangent vector $\xi_{x_k}$, i.e.
$\operatorname{dist}(x_k, \operatorname{retr}_{x_k}(\xi_{x_k})) \neq \lVert \xi_{x_k} \rVert_{x_k}$.\
Next, we compute the step $s_k \in \mathcal{T}_{x_k} \mathcal{M}$ as an (approximate)
solution of the trust-region subproblem given by the model
[\[RiemannianQuadraticModel\]](#RiemannianQuadraticModel){reference-type="ref"
reference="RiemannianQuadraticModel"} and the trust-region
[\[RiemannianTrustRegion\]](#RiemannianTrustRegion){reference-type="ref"
reference="RiemannianTrustRegion"}, i.e.
$$\label{Riemanniantrsubproblem}
    s_k = \arg \min_{\lVert s \rVert_{x_k} \leq \Delta_k} \hat{m}_k( s ) \in \mathcal{T}_{x_k} \mathcal{M}.$$
Since $\mathcal{T}_{x_k} \mathcal{M}$ is a Euclidean space, it is possible to adapt
classical methods in $R^n$ to compute a approximate minimizer of the
trust-region subproblem
[\[Riemanniantrsubproblem\]](#Riemanniantrsubproblem){reference-type="ref"
reference="Riemanniantrsubproblem"} [@AbsilBakerGallivan:2007 p. 304]. A
possible inner iteration method for this is given by
[@AbsilMahonySepulchre:2008 Algorithm 11], which is a generalization of
the tCG-method.\
This minimizer $s_k$ is then retracted back from $\mathcal{T}_{x_k} \mathcal{M}$ to
$\mathcal{M}$, i.e. $$\widetilde{x}_{k+1} = \operatorname{retr}_{x_k}(s_k).$$ This
point $\widetilde{x}_{k+1}$ is a candidate for the new iterate
$x_{k+1}$. The decisions on accepting or rejecting the candidate
$\widetilde{x}_{k+1}$ and on selecting the new trust-region radius
$\Delta_k$ are based on the quotient
$$\rho_k = \frac{\hat{f}_{x_k}(0_{x_k}) - \hat{f}_{x_k}(s_k)}{\hat{m}_k(0_{x_k}) - \hat{m}_k(s_k)} = \frac{f(x_k) - f(\operatorname{retr}_{x_k}(s_k))}{\hat{m}_k(0_{x_k}) - \hat{m}_k(s_k)} = \frac{f(x_k) - f(\widetilde{x}_{k+1})}{\hat{m}_k(0_{x_k}) - \hat{m}_k(s_k)}.$$
The same decision parameters are here used and the update of the
trust-region radius follows the same heuristics as in the Euclidean
case.\
We note that this "pullback-solve-retract" procedure distinguishes the
Riemannian Trust-Region approach from Euclidean Trust-Region methods,
which only require the "solve" part since they live in $\mathbb{R}^n$.
On a manifold, using the pullback $\hat{f}_{x_k}$ of the objective
function $f$ makes it possible to locally fall back to a friendly
Euclidean world (the tangent space $\mathcal{T}_{x_k} \mathcal{M}$) where classical
techniques can be applied, and the retraction $\operatorname{retr}_{x_k}$ brings the
result back to the manifold $\mathcal{M}$. A difficulty, from an
analysis perspective, is that the Riemannian Trust-Region approach does
not deal with a unique objective function $f$, but rather with a
succession of different pullbacks $\hat{f}_{x_k}$ of the objetcive
function [@AbsilBakerGallivan:2007 p. 305].\
Let us now turn to the quadratic term
$\mathcal{H}_k \colon \ \mathcal{T}_{x_k} \mathcal{M} \to \mathcal{T}_{x_k} \mathcal{M}$ in
[\[RiemannianQuadraticModel\]](#RiemannianQuadraticModel){reference-type="ref"
reference="RiemannianQuadraticModel"}. It can be shown, if
$\mathcal{H}_k$ is a sufficiently good approximation of the Hessian
operator $\operatorname{Hess} f(x_k)$ and under further (strong)
assumptions, among others on the chosen retraction, that the the
sequence $\{ x_k \}_k$ generated by the resulting method converges
q-superlinearly to a nondegenerate local minimizer $x^*$, i.e.
$\operatorname{grad} f(x^*) = 0_{x^*}$ and $\operatorname{Hess} f(x^*)$
is positive definite (see [@AbsilBakerGallivan:2007 Theorem 4.13]). A
possible choice fot that would be the Riemannian Hessian
$\mathcal{H}_k = \operatorname{Hess} f(x_k)$ and the exponential map
$\expOp$ (see [@AbsilMahonySepulchre:2008 p. 102]) as retraction. But as
in the Euclidean case, the application of $\operatorname{Hess} f(x_k)$
can be computationally too costly or $\operatorname{Hess} f(x_k)$
doesn't even exist. Therefore, we are looking for an approximation of
$\operatorname{Hess} f(x_k)$ that is easy to compute but still produces
a fast rate of convergence. This leads us to genralizing the SR1 update
[\[directSR1formula\]](#directSR1formula){reference-type="ref"
reference="directSR1formula"} for the Riemannian setup.\
We want to approximate the action of the Riemannian Hessian
$\operatorname{Hess} f(x_k)$ with a linear self-adjoint operator
$\mathcal{H}^\mathrm{SR1}_k \colon \ \mathcal{T}_{x_k} \mathcal{M} \to \mathcal{T}_{x_k} \mathcal{M}$,
which is updated (in each iteration) with information about the
curvature obtained by
$s_k, \operatorname{grad} f(x_k) \in \mathcal{T}_{x_k} \mathcal{M}$ and
$\operatorname{grad} f(\widetilde{x}_{k+1}) \in \mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M}$
to a new operator
$\mathcal{H}^\mathrm{SR1}_{k+1} \colon \ \mathcal{T}_{x_{k+1}} \mathcal{M} \to \mathcal{T}_{x_{k+1}} \mathcal{M}$,
which acts on the tangent space at the upcomming iterate $x_{k+1}$. To
be able to work with the information from different tangent spaces, we
use a so-called vector transport $\vectorTransportSymbol$ on a manifold
$\mathcal{M}$, which is a smooth map $$\begin{aligned}
    \vectorTransportSymbol \colon \ \mathcal{T} \mathcal{M} \oplus \mathcal{T} \mathcal{M} & \to \mathcal{T} \mathcal{M} \\
    (\eta_x, \xi_x) & \mapsto \vectorTransportDir{x}{\eta_x}(\xi_x)\end{aligned}$$
satisfying the following properties for all $x \in \mathcal{M}$:

1.  (Associated retraction)
    $\vectorTransportDir{x}{\eta_x}(\xi_x) \in \mathcal{T}_{\operatorname{retr}_{x}(\eta_x)} \mathcal{M}$
    for all $\xi_x \in \mathcal{T}_{x} \mathcal{M}$;

2.  (Consistency) $\vectorTransportDir{x}{0_x}(\xi_x) = \xi_x$ for all
    $\xi_x \in \mathcal{T}_{x} \mathcal{M}$;

3.  (Linearity)
    $\vectorTransportDir{x}{\eta_x}(a \xi_x + b \zeta_x) = a \vectorTransportDir{x}{\eta_x}(\xi_x) + b \vectorTransportDir{x}{\eta_x}(\zeta_x)$;

[@AbsilMahonySepulchre:2008 Definition 8.1.1]. A vector transport
$\vectorTransportSymbol^S \colon \ \mathcal{T} \mathcal{M} \oplus \mathcal{T} \mathcal{M} \to \mathcal{T} \mathcal{M}$
with associated retraction $\operatorname{retr}_{\cdot} \left( \cdot \right)$ is called isometric if it
satisfies
$$g_{\operatorname{retr}_{x}(\eta_x)}(\vectorTransportDir{x}{\eta_x}(\xi_x)[S], \vectorTransportDir{x}{\eta_x}(\zeta_x)[S]) = g_x (\xi_x, \zeta_x)$$
for all $\eta_x, \xi_x, \zeta_x \in \mathcal{T}_{x} \mathcal{M}$ and $x \in \mathcal{M}$
[@Huang:2013 p. 10]. We use $\vectorTransportSymbol^S$ to denote an
isometric vector transport.\
We are now able to summarize the information for the update in one
tangent space. Since, as in the Euclidean case, we want to update
$\mathcal{H}^\mathrm{SR1}_k$ before deciding whether or not to accept
the candidate $\widetilde{x}_{k+1}$, we use the tangent space at the
current iterate $\mathcal{T}_{x_k} \mathcal{M}$. For this we define
$y_k = {\mathrm{T}^{S}_{x_k, s_k}}^{-1} ( \operatorname{grad}f(\widetilde{x}_{k+1}) ) - \operatorname{grad}f(x_k) \in \mathcal{T}_{x_k} \mathcal{M}$,
where the associated retraction of $\vectorTransportSymbol^S$ is our
chosen retraction $\operatorname{retr}_{\cdot} \left( \cdot \right)$.\
To be able to create rank-one operators, we introduce the musical
isomorphism
$\flat \colon \ \mathcal{T}_{x} \mathcal{M} \ni \eta_{x} \mapsto \eta^{\flat}_{x} \in \cotangent{x}$
(see [@BergmannHerzogLouzeiroSilvaTenbrinckVidalNunez:2020:1 p. 6]). Put
simply, it means: $\eta^{\flat}_{x} \in \cotangent{x}$ represents the
flat of $\eta_{x} \in \mathcal{T}_{x} \mathcal{M}$, i.e.,
$\eta^{\flat}_{x} \colon \ \mathcal{T}_{x} \mathcal{M} \to \mathbb{R}, \; \xi_{x} \mapsto \eta^{\flat}_{x}[\xi_{x}] = g_{x} (\eta_{x}, \xi_{x})$.
This generalizes the notion of the transpose from the Euclidean case. It
can be shown that
$\eta_{x} \eta^{\flat}_{x} \colon \ \mathcal{T}_{x} \mathcal{M} \to \mathcal{T}_{x} \mathcal{M}$ is a
linear self-adjoint positive definite rank-one operator.\
With $s_k \in \mathcal{T}_{x_k} \mathcal{M}$, which generalizes the connection between
the current iterate $x_k$ and the candidate $\widetilde{x}_{k+1}$,
$y_k \in \mathcal{T}_{x_k} \mathcal{M}$, which generalizes the difference of the
gradients at $x_k$ and $\widetilde{x}_{k+1}$, and by introducing the
notion of the flat $\flat$, we can now define a self-adjoint rank-one,
short SR1, update for operators on the tangent space at the current
iterate $\mathcal{T}_{x_k} \mathcal{M}$: $$\label{RiemannianDirectSR1formula}
    \widetilde{\mathcal{H}}^{SR1}_{k+1} [\cdot] = \mathcal{H}^\mathrm{SR1}_k [\cdot] + \frac{(y_k - \mathcal{H}^\mathrm{SR1}_k [s_k]) (y_k - \mathcal{H}^\mathrm{SR1}_k [s_k])^{\flat} [\cdot] }{(y_k - \mathcal{H}^\mathrm{SR1}_k [s_k])^{\flat} [s_k]}.$$
We see immediately that
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"} creates a self-adjoint operator
if $\mathcal{H}^\mathrm{SR1}_k$ is self-adjoint. As in the Euclidean
case,
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"} inherits the positive
definiteness of $\mathcal{H}^\mathrm{SR1}_k$ if and only if
$(y_k - \mathcal{H}^\mathrm{SR1}_k [s_k])^{\flat} [s_k] > 0$ holds.\
The update
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"} has its origin from Riemannian
quasi-Newton methods, where it is required that the approximation of the
Hessian operator $\operatorname{Hess} f(x_{k+1})$ generated by the
corresponding update satisfies some kind of Riemannian quasi-Newton
equation (for more details see [@Huang:2013 Chapter 2]). For
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"}, it can be shown that
$$\widetilde{\mathcal{H}}^{SR1}_{k+1} [s_k] = y_k$$ holds, which
generalizes
[\[quasi-NewtonEquation\]](#quasi-NewtonEquation){reference-type="ref"
reference="quasi-NewtonEquation"} in this context.\
This update
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"} has also the disadvantage that
the numerator of the self-adjoint rank-one operator, which is added to
the current approximation $\mathcal{H}^\mathrm{SR1}_k$, can vanish. This
can also lead to numerical difficulties or even to the breakdown of the
corresponding method. To avoid this, we generalize the strategy we know
from the Euclidean case. The update
[\[directSR1formula\]](#directSR1formula){reference-type="ref"
reference="directSR1formula"} is applied only if
$$\label{RiemannianSafeguard}
    \lvert g_{x_k}(y_k - \mathcal{H}^{SR1}_k[s_k], \ s_k) \rvert \geq r \; \lVert y_k - \mathcal{H}^{SR1}_k[s_k] \rVert_{x_k} \ \lVert s_k \rVert_{x_k}$$
holds, where $r \in (0, 1)$ is again a small number.\
Right now the operator $\widetilde{\mathcal{H}}^{SR1}_{k+1}$ acts on the
tangent space at the current iterate $x_k$, but for the next iteration
we need an operator $\mathcal{H}^{SR1}_{k+1}$ on the tangent space at
the upcoming iterate $x_{k+1}$, which depends on whether we accept or
reject the candidate $\widetilde{x}_{k+1}$. Therefore, we transport the
operator $\widetilde{\mathcal{H}}^{SR1}_{k+1}$ into the tangent space
$\mathcal{T}_{x_{k+1}} \mathcal{M} = \mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M}$ if we accept
$\widetilde{x}_{k+1}$ as the next iterate $x_{k+1}$, i.e. if
$x_{k+1} = \widetilde{x}_{k+1}$ we set
$\mathcal{H}^{SR1}_{k+1} = \mathrm{T}^{S}_{x_k, s_k} \circ \widetilde{\mathcal{H}}^{SR1}_{k+1} \circ {\mathrm{T}^{S}_{x_k, s_k}}^{-1}$,
where $\mathrm{T}^{S}$ is the same isometric vector transport we use in
$y_k$. If we do not accept the candidate $\widetilde{x}_{k+1}$, i.e.
$x_{k+1} = x_k$, then we set
$\mathcal{H}^{SR1}_{k+1} = \widetilde{\mathcal{H}}^{SR1}_{k+1}$.\
We note that it is possible to define the update
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"} in the tangent space at
$\widetilde{x}_{k+1}$, in this case the operator
$\mathcal{H}^\mathrm{SR1}_k$ and $s_k$ would have to be transported to
$\mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M}$ and we would have to define $y_k$ in
$\mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M}$. The resulting algorithm would remain
equivalent since the vector transport is isometric
[@HuangAbsilGallivan:2014 p. 5].\
All this now discussed leads to the following algorithm, which can be
seen as a generalization of
[\[TR-SR1Method\]](#TR-SR1Method){reference-type="ref"
reference="TR-SR1Method"}. From now on we will refer to it as Riemannian
Trust-Region Symmetric Rank-One, short RTR-SR1, method:

::: {.algorithm}
::: {.algorithmic}
Riemannian manifold $(\mathcal{M}, g)$; isometric vector transport
$\vectorTransportSymbol^S$ with $\operatorname{retr}_{\cdot} \left( \cdot \right)$ as associated
retraction; continuously differentiable real-valued function $f$ on
$\mathcal{M}$, bounded below; initial iterate $x_0 \in \mathcal{M}$;
initial linear self-adjoint operator
$\mathcal{H}^\mathrm{SR1}_0 \colon \ \mathcal{T}_{x_0} \mathcal{M} \to \mathcal{T}_{x_0} \mathcal{M}$;
initial trust-region radius $\Delta_0 > 0$; safeguard tolerance
$r \in (0,1)$; acceptance tolerance $\rho^{\prime} \in (0, 0.1)$;
trust-region decrease factor $\tau_1 \in (0,1)$; trust-region increase
factor $\tau_2 > 1$; convergence tolerance $\varepsilon > 0$. Set
$k = 0$. Obtain $s_k$ by (approximately) solving
[\[Riemanniantrsubproblem\]](#Riemanniantrsubproblem){reference-type="ref"
reference="Riemanniantrsubproblem"} using
$\mathcal{H}_k = \mathcal{H}^\mathrm{SR1}_k$. Set
$\widetilde{x}_{k+1} = \operatorname{retr}_{x_k}(s_k)$ and
$y_k = {\mathrm{T}^{S}_{x_k, s_k}}^{-1} ( \operatorname{grad}f(\widetilde{x}_{k+1}) ) - \operatorname{grad}f(x_k)$.
Compute
$\widetilde{\mathcal{H}}^\mathrm{SR1}_{k+1} \colon \ \mathcal{T}_{x_{k}} \mathcal{M} \to \mathcal{T}_{x_{k}} \mathcal{M}$
by means of
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"}. Set
$\widetilde{\mathcal{H}}^\mathrm{SR1}_{k+1} = \mathcal{H}^\mathrm{SR1}_k$.
Compute
$\rho_k = \frac{f(x_k) - f(\widetilde{x}_{k+1})}{\hat{m}_k(0_{x_k}) - \hat{m}_k(s_k)}$.
Set $x_{k+1} = \widetilde{x}_{k+1}$ and
$\mathcal{H}^\mathrm{SR1}_{k+1} = T^{S}_{x_k, s_k} \circ \widetilde{\mathcal{H}}^\mathrm{SR1}_{k+1} \circ  {T^{S}_{x_k, s_k}}^{-1} \colon \ \mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M} \to \mathcal{T}_{\widetilde{x}_{k+1}} \mathcal{M}$.
Set $x_{k+1} = x_k$ and
$\mathcal{H}^\mathrm{SR1}_{k+1} = \widetilde{\mathcal{H}}^\mathrm{SR1}_{k+1} \colon \ \mathcal{T}_{x_{k}} \mathcal{M} \to \mathcal{T}_{x_{k}} \mathcal{M}$.
Set $\Delta_{k+1} = \tau_2 \ \Delta_k$. Set $\Delta_{k+1} = \Delta_k$.
Set $\Delta_{k+1} = \tau_1 \ \Delta_k$. Set $\Delta_{k+1} = \Delta_k$.
Set $k = k+1$. **Return** $x_k$.
:::
:::

We summarize the most important results of the convergence analysis of
[\[RTR-SR1Method\]](#RTR-SR1Method){reference-type="ref"
reference="RTR-SR1Method"}, which can be found in
[@HuangAbsilGallivan:2014]. For this we need some more assumptions,
which are among others also specific for the Riemannian setup.\
We assume throughout that $f \in C^2$ and we denote by $\Omega$ the
sublevel set of $x_0$, i.e.
$$\Omega = \{ x \in \mathcal{M} \colon \ f(x) \leq f(x_0) \}.$$
Furthermore, we assume for the retraction, $\operatorname{retr}_{\cdot} \left( \cdot \right)$, that
there exist $\mu > 0$ and $\delta_{\mu} > 0$ such that
$$\label{RetractionAssumption}
    \lVert \xi_x \rVert_x \geq \mu \ \operatorname{dist}(x, \operatorname{retr}_{x}(\xi_x)) \text{ for all } x \in \Omega, \text{ for all } \xi_x \in \mathcal{T}_{x} \mathcal{M}, \ \lVert \xi_x \rVert_x \leq \delta_{\mu}.$$
Such a condition is instrumental in the global convergence analysis of
Riemannian trust-region schemes [@HuangAbsilGallivan:2014 p. 7].\
As in the Euclidean case we require that the trust-region subproblem
[\[Riemanniantrsubproblem\]](#Riemanniantrsubproblem){reference-type="ref"
reference="Riemanniantrsubproblem"} is solved accurately enough, which
means that there exist positive constants $\sigma_1$ and $\sigma_2$,
sucht that $$\label{RiemannianAccuracy1}
    m_k(0_{x_k}) - m_k(s_k) \geq \sigma_1 \ \lVert \operatorname{grad} f(x_k) \rVert_{x_k} \ \min \Bigg \{ \Delta_k, \ \sigma_2 \ \frac{\lVert \operatorname{grad} f(x_k) \rVert_{x_k}}{\lVert \mathcal{H}_k \rVert} \Bigg \}$$
holds, and that exists a constant $\theta > 0$, such that
$$\label{RiemannianAccuracy2}
    \mathcal{H}_k [s_k] = - \operatorname{grad} f(x_k) + \delta_k \text{ with } \delta_k \in \mathcal{T}_{x_k} \mathcal{M}, \ \lVert \delta_k \rVert_{x_k} \leq \lVert \operatorname{grad} f(x_k) \rVert^{1 + \theta}_{x_k}, \text{ whenever } \lVert s_k \rVert_{x_k} < 0.8 \ \Delta_k,$$
holds. These conditions are generalizations of
[\[accuracy1\]](#accuracy1){reference-type="ref" reference="accuracy1"}
and [\[accuracy2\]](#accuracy2){reference-type="ref"
reference="accuracy2"}. The condition
[\[RiemannianAccuracy2\]](#RiemannianAccuracy2){reference-type="ref"
reference="RiemannianAccuracy2"} remains weaker than condition
[\[accuracy2\]](#accuracy2){reference-type="ref" reference="accuracy2"}.
The purpose of introducing $\delta_k$ in
[\[RiemannianAccuracy2\]](#RiemannianAccuracy2){reference-type="ref"
reference="RiemannianAccuracy2"} is to encompass stopping criteria such
as [@AbsilMahonySepulchre:2008 (7.10)] that do not require the
computation of an exact solution of the trust-region subproblem
[\[Riemanniantrsubproblem\]](#Riemanniantrsubproblem){reference-type="ref"
reference="Riemanniantrsubproblem"}. We point out that
[\[RiemannianAccuracy1\]](#RiemannianAccuracy1){reference-type="ref"
reference="RiemannianAccuracy1"} and
[\[RiemannianAccuracy2\]](#RiemannianAccuracy2){reference-type="ref"
reference="RiemannianAccuracy2"} hold if the approximate solution of
[\[Riemanniantrsubproblem\]](#Riemanniantrsubproblem){reference-type="ref"
reference="Riemanniantrsubproblem"} is obtained from
[@AbsilMahonySepulchre:2008 Algorithm 11], the tCG-method generalized
for the Riemannian setup [@HuangAbsilGallivan:2014 p. 7].\
The next assumption corresponds to the second in
[\[AssumptionsGlobalConvergence\]](#AssumptionsGlobalConvergence){reference-type="ref"
reference="AssumptionsGlobalConvergence"}

::: {.assumption}
[\[RiemannianAssumptionsGlobalConvergence\]]{#RiemannianAssumptionsGlobalConvergence
label="RiemannianAssumptionsGlobalConvergence"} The sequence of linear
operators $\{ \mathcal{H}^{\mathrm{SR1}}_k \}_k$ is bounded by a
constant $M$ such that
$\lVert \mathcal{H}^{\mathrm{SR1}}_k \rVert \leq M$ for all $k$.
:::

With all these assumptions, the global convergence of
[\[RTR-SR1Method\]](#RTR-SR1Method){reference-type="ref"
reference="RTR-SR1Method"} can be shown. The first two statements are
based on the results in [@AbsilMahonySepulchre:2008
7.4.1 Global convergence], which deal with the global convergence
analysis of general trust-region methods on Riemannian manifolds (i.e.
$\mathcal{H}_k$ in
[\[RiemannianQuadraticModel\]](#RiemannianQuadraticModel){reference-type="ref"
reference="RiemannianQuadraticModel"} is just a self-adjoint operator,
which approximates the $\operatorname{Hess} f(x_k)$ sufficiently good),
and the third statement generalizes
[\[GlobalConvergence\]](#GlobalConvergence){reference-type="ref"
reference="GlobalConvergence"}:

::: {.theorem}
[\[RiemannianGlobalConvergence\]]{#RiemannianGlobalConvergence
label="RiemannianGlobalConvergence"}

1.  If $f$ is bounded below on the sublevel set $\Omega$,
    [\[RiemannianAssumptionsGlobalConvergence\]](#RiemannianAssumptionsGlobalConvergence){reference-type="ref"
    reference="RiemannianAssumptionsGlobalConvergence"} holds, condition
    [\[RiemannianAccuracy1\]](#RiemannianAccuracy1){reference-type="ref"
    reference="RiemannianAccuracy1"} holds, and
    [\[RetractionAssumption\]](#RetractionAssumption){reference-type="ref"
    reference="RetractionAssumption"} is satisfied then
    $\lim_{k \rightarrow \infty} \operatorname{grad} f(x_k) = 0$.

2.  If $\mathcal{M}$ is compact, Assumption 3.1 holds, and
    [\[RiemannianAccuracy1\]](#RiemannianAccuracy1){reference-type="ref"
    reference="RiemannianAccuracy1"} holds then
    $\lim_{k \rightarrow \infty} \operatorname{grad} f(x_k) = 0$,
    $\{ x_k \}_k$ has at least one limit point, and every limit point of
    $\{ x_k \}_k$ is a stationary point of $f$.

3.  If the sublevel set $\Omega$ is compact, $f$ has a unique stationary
    point $x^*$ in $\Omega$, Assumption 3.1 holds, condition
    [\[RiemannianAccuracy1\]](#RiemannianAccuracy1){reference-type="ref"
    reference="RiemannianAccuracy1"} holds, and
    [\[RetractionAssumption\]](#RetractionAssumption){reference-type="ref"
    reference="RetractionAssumption"} is satisfied then $\{ x_k \}_k$
    converges to $x^*$.
:::

The local convergence analysis in [@HuangAbsilGallivan:2014] can be
viewed as a Riemannian generalization of the local convergence analysis
in [@ByrdKhalfanSchnabel:1996]. The derivation of the results was
subject to some hurdles that had to be overcome, which is why several
preparation lemmata were used. We summarize the required assumptions:

::: {.assumption}
[\[RiemannianAssumptionsLocalConvergence\]]{#RiemannianAssumptionsLocalConvergence
label="RiemannianAssumptionsLocalConvergence"}  \

1.  We assume that $\{ x_k \}_k$ converges to a point $x^*$.

2.  We let $\mathcal{U}_{trn}$ be a totally retractive neighborhood of
    $x^*$. This means that there is $\delta_{trn} > 0$ such that, for
    each $y \in \mathcal{U}_{trn}$, we have that
    $\operatorname{retr}_{y}(\mathcal{B}(0_y, \delta_{trn})) \supseteq \mathcal{U}_{trn}$
    and $\operatorname{retr}_{y}(\cdot)$ is a diffeomorphism on
    $\mathcal{B}(0_y, \delta_{trn})$, where
    $\mathcal{B}(0_y, \delta_{trn})$ denotes the ball of radius
    $\delta_{trn}$ in $\mathcal{T}_{y} \mathcal{M}$ centered at the origin $0_y$. We
    assume without loss of generality that
    $\{ x_k \}_k \subset \mathcal{U}_{trn}$.

3.  The point $x^*$ is a nondegenerate local minimizer of $f$, i.e.
    $\operatorname{grad} f(x^*) = 0$ and $\operatorname{Hess} f(x^*)$ is
    positive definite.

4.  There exists a constant $c$ such that for all
    $x, y \in \mathcal{U}_{trn}$,
    $$\lVert \operatorname{Hess} f(y) - \mathrm{T}^{S}_{x, \eta_x} \circ \operatorname{Hess} f(x) \circ {\mathrm{T}^{S}_{x, \eta_x}}^{-1} \rVert \leq c \ \operatorname{dist}(x,y),$$
    where $\eta_x = {\operatorname{retr}_{\cdot} \left( \cdot \right)_{x}}^{-1}(y)$.

5.  There exists a constant $c_0$ such that for all
    $x, y \in \mathcal{U}_{trn}$, all $\xi_x \in \mathcal{T}_{x} \mathcal{M}$ with
    $\operatorname{retr}_{x}(\xi_x) \in \mathcal{U}_{trn}$, and all
    $\xi_y \in \mathcal{T}_{y} \mathcal{M}$ with
    $\operatorname{retr}_{y}(\xi_y) \in \mathcal{U}_{trn}$, it holds that
    $$\lVert \operatorname{Hess} \hat{f}_y(\xi_y) - \mathrm{T}^{S}_{x, \eta_x} \circ \operatorname{Hess} \hat{f}_x(\xi_x) \circ {\mathrm{T}^{S}_{x, \eta_x}}^{-1} \rVert \leq c_0 \ (\lVert \xi_x \rVert_x + \lVert \xi_y \rVert_y + \lVert \eta_x \rVert_x),$$
    where $\eta_x = {\operatorname{retr}_{\cdot} \left( \cdot \right)_{x}}^{-1}(y)$,
    $\hat{f}_x(\cdot) = f \circ \operatorname{retr}_{x}(\cdot)$ and
    $\hat{f}_y(\cdot) = f \circ \operatorname{retr}_{y}(\cdot)$.

6.  For each iteration
    [\[RiemannianSafeguard\]](#RiemannianSafeguard){reference-type="ref"
    reference="RiemannianSafeguard"} holds.

7.  There exists $N$ such that, for all $k \geq N$ and all
    $t \in [0, 1]$, it holds that
    $\operatorname{retr}_{x_k}(t s_k) \in \mathcal{U}_{trn}$.
:::

With all these assumptions, a generalization of
[\[LocalConvergence\]](#LocalConvergence){reference-type="ref"
reference="LocalConvergence"} can be proved, which shows that the
$n + 1$-step q-superlinear convergence,
[\[n+1superlinear\]](#n+1superlinear){reference-type="ref"
reference="n+1superlinear"}, of the Euclidean SR1 method,
[\[TR-SR1Method\]](#TR-SR1Method){reference-type="ref"
reference="TR-SR1Method"}, is preserved when transferred to the
Riemannian setup:

::: {.theorem}
[\[RiemannianLocalConvergence\]]{#RiemannianLocalConvergence
label="RiemannianLocalConvergence"} If
[\[RiemannianAssumptionsGlobalConvergence\]](#RiemannianAssumptionsGlobalConvergence){reference-type="ref"
reference="RiemannianAssumptionsGlobalConvergence"} and
[\[RiemannianAssumptionsLocalConvergence\]](#RiemannianAssumptionsLocalConvergence){reference-type="ref"
reference="RiemannianAssumptionsLocalConvergence"} hold and the
subproblem is solved accurately enough for
[\[RiemannianAccuracy1\]](#RiemannianAccuracy1){reference-type="ref"
reference="RiemannianAccuracy1"} and
[\[RiemannianAccuracy2\]](#RiemannianAccuracy2){reference-type="ref"
reference="RiemannianAccuracy2"} to hold then, the sequence
$\{ x_k \}_k$ generated by
[\[RTR-SR1Method\]](#RTR-SR1Method){reference-type="ref"
reference="RTR-SR1Method"} is $n + 1$-step q-superlinear (where $n$
denotes the dimension of the manifold $\mathcal{M}$); i.e.,
$$\lim_{k \rightarrow \infty} \frac{\operatorname{dist}(x_{k+n+1}, x^*)}{\operatorname{dist}(x_k, x^*)} = 0.$$
:::

Thus, it can also be concluded here that the SR1 update for operators,
[\[RiemannianDirectSR1formula\]](#RiemannianDirectSR1formula){reference-type="ref"
reference="RiemannianDirectSR1formula"}, provides a promising approach
in the use of a Riemannian Trust-Region method.


"""

# ╔═╡ Cell order:
# ╟─b24e567a-3ea3-441f-bbbd-acf0c12e9c20
# ╠═d3df4e1c-bd58-4e13-99cf-7758b83c689e
