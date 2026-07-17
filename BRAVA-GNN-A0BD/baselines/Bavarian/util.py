#! /usr/bin/env python3
#
# Utilities for the other scripts
#
# Copyright 2020 Matteo Riondato <rionda@acm.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os.path
import sys
import numpy as np
import pandas as pd


def parseJSONFromFile(filename):
    res = dict()
    try:
        with open(filename, "r") as jsonf:
            res = json.load(jsonf)
    except EnvironmentError as e:
        sys.exit(f"Error reading the file '{filename}': {e}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error parsing the JSON in {filename}: {e}")
    return res


def getConf(filename, requiredKeys):
    conf = parseJSONFromFile(filename)
    # Read in an included configuration specified in the conf file
    # Keys that are present in the conf file take precedence over those in
    # the included configuration.
    if "include" in conf:
        for (k, v) in parseJSONFromFile(conf["include"]).items():
            if k not in conf:
                conf[k] = v
    # Check that the conf has all the necessary keys
    if "algos" not in conf:
        sys.exit("Error: the configuration must include a list of algorithms to run.")
    if "filename" not in conf:
        sys.exit("Error: the configuration must include a filename.")
    if "name" not in conf:
        sys.exit(
            "Error: the configuration must include a name for the" "configuration."
        )
    for key in ("deltas", "mctrials") + requiredKeys:
        if key not in conf:
            sys.exit(
                "Error: the configuration must include a (list of) "
                f"value(s) for {key}"
            )
    if "reps" not in conf:
        # Single repetition if not otherwise specified
        conf["reps"] = 1
    if "datadir" in conf:
        # Prepend datadir to the filename, if specified
        conf["filename"] = os.path.join(conf["datadir"], conf["filename"])
    # Set the right extension for logfiles depending on the flags
    conf["ext"] = ".txt"
    if "flags" in conf:
        if "-j" in conf["flags"]:
            conf["ext"] = ".json"
    else:
        conf["flags"] = ""
    return conf


# Class to store the results of a single Bavarian run
class Result:
    def __init__(self, filename, exactBCs):
        self.res = parseJSONFromFile(filename)
        # We use this class for the results of all types of experiment.
        # Depending on the experiment type, we may
        if "epsilon" not in self.res["settings"]:
            self.res["settings"]["epsilon"] = 0
            self.res["results"]["epsratio"] = 0
        else:
            self.res["results"]["epsratio"] = (
                self.res["results"]["epsilon"] / self.res["settings"]["epsilon"]
            )
        if "multiplier" not in self.res["settings"]:
            self.res["settings"]["multiplier"] = 0
        if "samples" not in self.res["settings"]:
            self.res["settings"]["samples"] = 0
        if "timeout" not in self.res["settings"]:
            self.res["settings"]["timeout"] = 0
        if "iterations" not in self.res["results"]:
            self.res["results"]["iterations"] = -1
        if "max_iterations" not in self.res["results"]:
            self.res["results"]["max_iterations"] = -1
        if "nomc_epsilon" not in self.res["results"]:
            self.res["results"]["nomc_epsilon"] = 0
        if "samples" not in self.res["results"]:
            self.res["results"]["samples"] = -1
        # XXX: The following is not needed for more recent results.
        # Fix to handle the wrong computation of Massart Lemma.
        # if self.res["settings"]["method"] == "ab":
        #    samples = self.res["settings"]["samples"]
        #    if samples == 0:
        #        samples = self.res["results"]["samples"]
        #    eraBound = np.sqrt(
        #        self.res["results"]["ewimpy"]
        #        * 2
        #        * np.log(self.res["properties"]["nodes"])
        #        / samples
        #    )
        #    logTerm = np.log(3) - np.log(self.res["settings"]["delta"])
        #    self.res["results"]["nomc_epsilon"] = (
        #        2 * eraBound
        #        + (logTerm + np.sqrt(logTerm * (logTerm + 4 * samples * eraBound)))
        #        / samples
        #        + np.sqrt(logTerm / (2 * samples))
        #    )
        sampleBCs = self.res["results"]["approximations"]
        if exactBCs:
            self.stats = getErrorStats(
                self.res["results"]["epsilon"], sampleBCs, exactBCs
            )
        else:
            self.stats = {
                "wrong": -1,
                "avg": 0,
                "stdev": 0,
                "min": 0,
                "1stq": 0,
                "median": 0,
                "3rdq": 0,
                "max": 0,
            }

    def getTuple(self):
        return (
            self.res["settings"]["method"],
            self.res["settings"]["delta"],
            self.res["settings"]["mctrials"],
            self.res["settings"]["samples"],
            self.res["settings"]["timeout"],
            self.res["settings"]["epsilon"],
            self.res["settings"]["multiplier"],
            self.res["results"]["epsilon"],
            self.res["results"]["epsratio"],
            self.res["results"]["nomc_epsilon"],
            self.res["results"]["mcera"],
            self.res["results"]["erabound"],
            self.res["results"]["rabound"],
            self.res["results"]["iterations"],
            self.res["results"]["max_iterations"],
            self.res["results"]["samples"],
            int(
                np.ceil(
                    np.log(2 / self.res["settings"]["delta"])
                    * 2
                    / pow(self.res["results"]["epsilon"], 2)
                )
            ),
            self.res["total_time"] / 1000,  # convert to seconds
            self.stats["wrong"],
            self.stats["avg"],
            self.stats["stdev"],
            self.stats["min"],
            self.stats["1stq"],
            self.stats["median"],
            self.stats["3rdq"],
            self.stats["max"],
        )


# Compute stats on the estimation errors for a single run
def getErrorStats(eps, sampleBCs, exactBCs):
    assert exactBCs.keys() == sampleBCs.keys()
    errs = []
    wrong = 0
    for node in exactBCs:
        absErr = abs(exactBCs[node] - sampleBCs[node])
        errs.append(absErr)
        if absErr > eps:
            wrong += 1
            sys.stderr.write(
                f"WrongEps: exact={exactBCs[node]}, sample={sampleBCs[node]}\n"
            )
        # We do not use the relative error for anything, but if we do in the
        # future, here it is:
        # if exactBCs[node] > 0:
        #    relErr = (absErr / exactBCs[node]) * 100
        # else:
        #    assert(sampleBCs[node] == 0)
        #    relErr = 0
    errMin, err1stq, errMed, err3rdq, errMax = np.percentile(errs, [0, 25, 50, 75, 100])
    return {
        "wrong": wrong,
        "avg": np.mean(errs),
        "stdev": np.std(errs),
        "min": errMin,
        "1stq": err1stq,
        "median": errMed,
        "3rdq": err3rdq,
        "max": errMax,
    }


# Parse the log files using the conf and comparing the results with the exact
# ones. Return an array of Results objects
def getResults(conf, reqConfs, expType, exactBCs):
    res = []
    for algo in conf["algos"]:
        print(f"algo: {algo}")
        for delta in conf["deltas"]:
            print(f"  delta: {delta}")
            for mct in conf["mctrials"]:
                print(f"    mct: {mct}")
                if expType == "time" or expType == "size":
                    for val in conf[reqConfs[0]]:
                        print(f"      {reqConfs[0]}: {val}")
                        for rep in range(conf["reps"]):
                            print(f"        rep: {rep}")
                            logFile = os.path.join(
                                conf["logdir"],
                                f"{expType}_{conf['name']}"
                                f"_{algo}_{delta}_{mct}_{val}"
                                f"_{rep}_out.json",
                            )
                            res.append(Result(logFile, exactBCs).getTuple())
                elif expType == "mult":
                    for eps in conf["epsilons"]:
                        print(f"      epsilon: {eps}")
                        for mult in conf["multipliers"]:
                            print(f"        mult: {mult}")
                            for rep in range(conf["reps"]):
                                print(f"          rep: {rep}")
                                logFile = os.path.join(
                                    conf["logdir"],
                                    f"{expType}_{conf['name']}"
                                    f"_{algo}_{delta}_{mct}_{eps}_{mult}"
                                    f"_{rep}_out.json",
                                )
                            res.append(Result(logFile, exactBCs).getTuple())
    df = pd.DataFrame.from_records(
        data=res,
        columns=[
            "method",
            "delta",
            "mctrials",
            "set_samples",
            "timeout",
            "set_epsilon",
            "multiplier",
            "epsilon",
            "epsratio",
            "nomc_epsilon",
            "mcera",
            "erabound",
            "rabound",
            "iterations",
            "max_iterations",
            "samples",
            "nomc_samples",
            "total_time",
            "wrong",
            "err_avg",
            "err_stdev",
            "err_min",
            "err_1stq",
            "err_median",
            "err_3rdq",
            "err_max",
        ],
    )
    return df
