using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

main_paths = ["."]
for main_path in main_paths
    format(
        main_path;
        whitespace_ops_in_indices = true,
        remove_extra_newlines = true,
        verbose = true,
        always_for_in = true,
        whitespace_typedefs = true,
        # always_use_return = true. # Disabled since it throws a lot of false positives
    )
end
