# # [Geodesic Regression](@id Geodesic Regression)
#
# Geodesic regression generalizes [linear regression](https://en.wikipedia.org/wiki/Linear_regression)
# to Riemannian manifolds. Let's first phrase it informally as follows:
#
# > For given data points ``d_1,\ldots,d_n`` on a Riemannian manifold ``\mathcal M`` find
# > the geodesic that “best explains” the data.
#
# The meaning of “best explain” has still to be clarified. We distinguish two cases: time labeled data and unlabeled data
#
# ## [Time labeled data](@id time-labelled-regression)
# if for each data item $d_i$ we are also given a time point $t_i\in\mathbb R$, which are pairwise different.
# Then we can use the least squares error to state the objetive function as [^Fletcher2013]
#
# ```math
# F(p,X) = \frac{1}{2}\sum_{i=1}^n d_{\mathcal M}^2(γ_{p,X}(t_i), d_i),
# ```
#
# where ``d_{\mathcal M}`` is the Riemannian distance and ``γ_{p,X}`` is the geodesic
# with ``γ(0) = p`` and ``\dot\gamma(0) = X``.
#
# For the real-valued case ``\mathcal M = \mathbb R^m`` the solution ``(p^*, X^*)`` is given in closed form
# as follows: with ``d^* = \frac{1}{n}\displaystyle\sum_{i=1}^{n}d_i`` and ``t^* = \frac{1}{n}\displaystyle\sum_{i=1}^n t_i``
# we get
#
# ```math
#  X^* = \frac{\sum_{i=1}^n (d_i-d^*)(t-t^*)}{\sum_{i=1}^n (t_i-t^*)^2}
# \quad\text{ and }\quad
# p^* = d^* - t^*X^*
# ```
#
# and hence the linear regression result is the line ``γ_{p^*,x^*}(t) = p^* + tX^*``.
#
# On a Riemannian manifold we can phrase this as an optimization problem on the [tangent bundle](https://en.wikipedia.org/wiki/Tangent_bundle),
# i.e. the disjoiint union of all tangent spaces, as
#
# ```math
# \operatorname*{arg\,min}_{(p,X) \in \mathrm{T}\mathcal M} F(p,X)
# ```
#
# ## [Unlabeled data](@id unlabeled-data)
#
# If we are not given time points $t_i$, then the optimization problem extends – informally speaking –
# to also finding the “best fitting” (in the sense of smallest error).
# To formalize, the objective function here reads
#
#
# ```math
# F(p, X, t) = \frac{1}{2}\sum_{i=1}^n d_{\mathcal M}^2(γ_{p,X}(t_i), d_i),
# ```
#
# where ``t = (t_1,\ldots,t_n) \in \mathbb R^n`` is now an additional parameter of the objective function.
#
# For the Euclidean case, the result is given by the first principal component of a principal component analysis,
# see [PCR](https://en.wikipedia.org/wiki/Principal_component_regression), i.e. with ``p^* = \frac{1}{n}\displaystyle\sum_{i=1}^n d_i``
# the direction ``X^*`` is obtained by defining the zero mean data matrix
#
# ```math
# D = \bigl(d_1-p^*, \ldots, d_n-p^*\bigr) \in \mathbb R^{m,n}
# ```
#
# and taking ``X^*`` as an eigenvector to the larges eigenvalue of ``D^{\mathrm{T}}D``.
# The optimal “time labels” are then just the projections ``t_i = ⟨d_i,X^*⟩``, ``i=1,\ldots,n``.
# Hence the error ``d_i - (p^* + t^*_iX^*)`` has the smalles two norm (or the ``t_i^*`` is the best possible time point with respect to said error).
#
# On a Riemannian manifold this can be stated as a problem on the product manifold ``\mathcal N = \mathrm{T}\mathcal M \times \mathbb R^n``, i.e.
#
# ```math
#   \operatorname*{arg\,min}_{\bigl((p,X),t\bigr)\in\mathcal N} F(p, X, t).
# ```
#
# In this tutorial we present an approach to solve this using an alternating gradient descent scheme.
#
#
# [^Fletcher2013]:
# > Fletcher, P. T., _Geodesic regression and the theory of least squares on Riemannian manifolds_,
# > International Journal of Computer Vision(105), 2, pp. 171–185, 2013.
# > doi: [10.1007/s11263-012-0591-y](https://doi.org/10.1007/s11263-012-0591-y)
