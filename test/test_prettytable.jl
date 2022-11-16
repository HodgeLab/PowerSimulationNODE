path = (joinpath(pwd(), "test-pretty-print-dir"))
!isdir(path) && mkdir(path)

try
    test1 = TrainParams(train_id = "01")
    test2 = TrainParams(train_id = "10", lb_loss = 1.0)
    test3 = TrainParams(train_id = "02", lb_loss = 1.0)
    PowerSimulationNODE.serialize(test1, joinpath(path, "train_1.json"))
    PowerSimulationNODE.serialize(test2, joinpath(path, "train_2.json"))
    PowerSimulationNODE.serialize(test3, joinpath(path, "train_3.json"))

    print_train_parameter_overview(path)

finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
