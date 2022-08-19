#Need to explicitly add unregistered dependencies first: https://github.com/JuliaLang/Pkg.jl/pull/1628
push!(LOAD_PATH, "../src/")
using Documenter, PowerSimulationNODE

makedocs(
    #format = Documenter.HTML(mathengine = Documenter.MathJax()),
    modules = [PowerSimulationNODE],
    sitename = "PowerSimulationNODE",
)

#Deploy locally 
using LiveServer
serve(dir = "docs/build")

#Deploy via GitHub
#= deploydocs(
    repo = "https://github.com/HodgeLab/PowerSimulationNODE.git",
    branch = "gh-pages",
    devbranch = "main",
) =#
