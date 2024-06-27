# -*- coding: utf-8 -*-

from optimizer import run

optimizer = ["MVO", "JAYA", "PSO"]
objectivefunc = ["svr_opt_func"]
NumOfRuns = 1
params = {"PopulationSize": 100, "Iterations": 50}


# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_details": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
