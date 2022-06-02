#Make a new directory 
#Create three new TrainParams and save them as JSON (make two of the )
#pretty print them to check the 

path = (joinpath(pwd(), "test-pretty-print-dir"))
!isdir(path) && mkdir(path)

try
    test1 = TrainParams(train_id = "test_1")
    test2 = TrainParams(train_id = "test_2", maxiters = 2000)
    PowerSimulationNODE.serialize(test1, joinpath(path, "train_1.json"))
    PowerSimulationNODE.serialize(test2, joinpath(path, "train_2.json"))

    print_train_parameter_overview(path)

finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
