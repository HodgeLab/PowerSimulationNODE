
function find_connecting_branches(sys, area_name)
    connecting_branches = []
    ac_branches = PSY.get_components(PSY.Device, sys, x -> typeof(x) <: PSY.ACBranch)
    for b in ac_branches
        from_bus = PSY.get_from(PSY.get_arc(b))
        to_bus = PSY.get_to(PSY.get_arc(b))
        if (PSY.get_name(PSY.get_area(from_bus)) == area_name) &&
           (PSY.get_name(PSY.get_area(to_bus)) != area_name)
            push!(connecting_branches, b)
        end
        if (PSY.get_name(PSY.get_area(to_bus)) == area_name) &&
           (PSY.get_name(PSY.get_area(from_bus)) != area_name)
            push!(connecting_branches, b)
        end
    end
    return connecting_branches
end

function all_line_trips(sys, t_fault)
    perturbations = []
    lines = PSY.get_components(PSY.Line, sys)
    for l in lines
        push!(
            perturbations,
            PowerSimulationsDynamics.BranchTrip(t_fault, PSY.Line, PSY.get_name(l)),
        )
    end
    return perturbations
end
