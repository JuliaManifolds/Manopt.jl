# [Gradients](@id GradientFunctions)

For a function $f\colon\mathcal M→ℝ$
the Riemannian gradient $\operatorname{grad}f(x)$ at $x∈\mathcal M$
is given by the unique tangent vector fulfilling

$\langle \operatorname{grad}f(x), \xi\rangle_x = D_xf[\xi],\quad
\forall \xi ∈ T_x\mathcal M,$
where $D_xf[\xi]$ denotes the differential of $f$ at $x$ with respect to
the tangent direction (vector) $\xi$ or in other words the directional
derivative.

This page collects the available gradients.

```@autodocs
Modules = [Manopt]
Pages   = ["gradients.jl"]
```
