```@meta
CurrentModule = Manopt
```
# [Debug Options](@id SectionDebug)
```@docs
DebugOptions
Debug
```
## Functions
```@docs
optionsHasDebug
getDebugOptions
setDebugOptions!
getDebugFunction
setDebugFunction!
updateDebugValues!
getVerbosity
```
## Debug Helpers for certain solvers
```@docs
cyclicProximalPointDebug
gradientDebug
subGradientDebug
```

## String production helpers for the debug output
```@docs
getCostString
getIterationString
getKeyValueString
getLastChangeString
getNormGradientString
getNormSubGradientString
getStopReasonString
```
