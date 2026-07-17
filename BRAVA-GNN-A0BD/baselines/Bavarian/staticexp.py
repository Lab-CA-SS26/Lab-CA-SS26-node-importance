#! /usr/bin/env python3
#
# Run an experiment using the static sampling algorithm (either fixed time or
# fixed size)
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

import os.path
import subprocess
import sys

import util

executable = "bavarian"


def usage():
    sys.stderr.write(f"Usage: {sys.argv[0]} {{time|size}} CONFIG_FILE\n")


# Check the sanity of the command line arguments
def checkArgsSanity():
    if len(sys.argv) > 1 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.stderr.write("Error: wrong number of arguments.\n")
        usage()
        sys.exit(2)
    if sys.argv[1] != "time" and sys.argv[1] != "size":
        sys.exit(
            f"Error: invalid experiment type {sys.argv[1]!r}. Must be "
            "one of 'time' or 'size'."
        )


def main():
    global executable
    # Check the sanity of the command line arguments
    checkArgsSanity()
    expType = sys.argv[1]
    if expType == "time":
        reqConf = "timeouts"
    else:  # implied expType == "size"
        reqConf = "samples"
    conf = util.getConf(sys.argv[-1], (reqConf,))
    if expType == "time":
        exact = util.parseJSONFromFile(
            os.path.join(conf["exactdir"], conf["name"] + ".json")
        )
        exact["total_time"] /= 1000  # convert to seconds
    if "-t" in conf["flags"]:
        sys.exit(
            "You don't want to run this script with '-t TIMEOUT' specified in "
            "the 'flags' configuration key. Use the 'timeouts' configuration "
            "key and the 'time' experiment type."
        )
    if "bindir" in conf:
        executable = os.path.join(conf["bindir"], executable)
    # Let's go!
    for algo in conf["algos"]:
        print(f"algo: {algo}")
        for delta in conf["deltas"]:
            print(f"  delta: {delta}")
            for mct in conf["mctrials"]:
                print(f"    mct: {mct}")
                for val in conf[reqConf]:
                    args = [executable] + conf["flags"]
                    if expType == "time":
                        # the zero is for the sample size, which is ignored
                        # when running with fixed time
                        timeVal = (val / 100) * (exact["total_time"])
                        args += ["-t", str(timeVal), "0"]
                        print(f"      timeout: {timeVal} ({val}% of exact)")
                    else:  # implied expType == "size"
                        args += [str(val)]
                        print(f"      samples: {val}")
                    args += [str(mct), algo, str(delta), conf["filename"]]
                    for rep in range(conf["reps"]):
                        logBase = (
                            f"{expType}_{conf['name']}_{algo}_{delta}_"
                            f"{mct}_{val}_{rep}"
                        )
                        print(f"        rep: {rep}")
                        try:
                            if "logdir" not in conf:
                                print("### start of process output ###")
                                subprocess.run(args, check=True)
                                print("### end of process output ###")
                            else:
                                # buffering=1 activates line buffering (maybe,
                                # because it is not very clear what
                                # subprocess.run( ) does with these files (does
                                # it re-open them?)
                                with open(
                                    os.path.join(
                                        conf["logdir"], logBase + "_out" + conf["ext"]
                                    ),
                                    mode="wt",
                                    buffering=1,
                                ) as outl, open(
                                    os.path.join(conf["logdir"], logBase + "_err.txt"),
                                    mode="wt",
                                    buffering=1,
                                ) as errl:
                                    subprocess.run(
                                        args, stdout=outl, stderr=errl, check=True
                                    )
                        except subprocess.CalledProcessError as e:
                            logInfo = " See the error message above."
                            if "logdir" in conf:
                                path = os.path.join(
                                    conf["logdir"], f"{logBase}_err.txt"
                                )
                                logInfo = f" Check logfile {path}."
                            sys.exit("Error running the program: " f"{e}{logInfo}")
                        except (
                            FileExistsError,
                            FileNotFoundError,
                            PermissionError,
                        ) as e:
                            sys.exit(f"Error writing the log files: {e}")
                        except OSError as e:
                            sys.exit(f"Error running the program: {e}")
                    # end of reps loop
                # end of vals loop
            # end of mct loop
        # end of delta loop
    # end of algo loop


if __name__ == "__main__":
    main()
