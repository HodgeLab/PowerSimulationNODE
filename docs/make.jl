push!(LOAD_PATH, "../src/")

using Documenter, PowerSimulationNODE


makedocs(
    #format = Documenter.HTML(mathengine = Documenter.MathJax()),
    modules = [PowerSimulationNODE],
    sitename = "PowerSimulationNODE",
)

deploydocs(repo = "https://github.com/HodgeLab/PowerSimulationNODE.git",
    branch = "gh-pages",
    devbranch = "main")
