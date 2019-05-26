# [Exports](@id Exports)

Exports aim to provide a consistent generation of images of your results. For example if you [record](@ref RecordOptions) the trace your algorithm walks on the [`Sphere`](@ref)`(2)`, you yan easily export this trace to a rendered image using [`asyExportS2Signals`](@ref) and render the result with [Asymptote](https://sourceforge.net/projects/asymptote/).
Despite these, you can always [record](@ref RecordOptions) values during your iterations,
and export these, for example to `csv`. 

## Asymptote
The following functions provide exports both in graphics and/or raw data using [Asymptote](https://sourceforge.net/projects/asymptote/).

```@docs
renderAsymptote
```
## Exports for specific Manifolds
```@docs
asyExportS2Signals
asyExportS2Data
asyExportSPDData
```
