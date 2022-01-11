push!(LOAD_PATH,"../src/")

using Documenter, PowerSimulationNODE

makedocs(sitename="My Documentation")
makedocs(
    format = Documenter.HTML(mathengine = Documenter.MathJax()),
    sitename = "PowerSimulationNODE",
)

deploydocs(
    repo = "https://github.com/HodgeLab/PowerSimulationNODE.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    devurl = "dev",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"],
)