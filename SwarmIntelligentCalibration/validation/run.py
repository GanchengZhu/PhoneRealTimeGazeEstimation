#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

for regressor in ["KNN", "RIDGE", "XGB", "SVR", "PSO-SVR", "JAYA-SVR", "MVO-SVR"]:
    for data_source in ["calibration_all", "calibration_zjugaze"]:
        command = "python smooth_pursuit_calibration.py --data_source %s  --regression_type %s" % (data_source, regressor)
        print(command)
        os.system(command)
