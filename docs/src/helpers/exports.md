# [Exports](@id Exports)

Exports aim to provide a consistent generation of images of your results. For example if you [record](@ref RecordOptions) the trace your algorithm walks on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html), you can easily export this trace to a rendered image using [`asymptote_export_S2_signals`](@ref) and render the result with [Asymptote](https://sourceforge.net/projects/asymptote/).
Despite these, you can always [record](@ref RecordOptions) values during your iterations,
and export these, for example to `csv`.

## Asymptote

The following functions provide exports both in graphics and/or raw data using [Asymptote](https://sourceforge.net/projects/asymptote/).

```@autodocs
Modules = [Manopt]
Pages   = ["Asymptote.jl"]
```
