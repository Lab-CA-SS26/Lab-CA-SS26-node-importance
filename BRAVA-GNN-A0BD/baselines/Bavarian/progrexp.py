#! /usr/bin/env python3
#
# Run an experiment using the progressive sampling algorithm (either fixed time
# or fixed size)
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

executable = "progrbavarian"


def usage():
    sys.stderr.write(f"Usage: {sys.argv[0]} {{iter|mult}} CONFIG_FILE\n")


# Check the sanity of the command line arguments
def checkArgsSanity():
    if len(sys.argv) > 1 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.stderr.write("Error: wrong number of arguments.\n")
        usage()
        sys.exit(2)
    if sys.argv[1] != "iter" and sys.argv[1] != "mult":
        sys.exit(
            f"Error: invalid experiment type {sys.argv[1]!r}. Must be "
            "one of 'iter' or 'mult'."
        )


def main():
    global executable
    # Check the sanity of the command line arguments
    checkArgsSanity()
    expType = sys.argv[1]
    reqConf = "multipliers"
    if expType == "iter":
        reqConf = "iterations"
    conf = util.getConf(sys.argv[-1], (reqConf, "epsilons"))
    # Use the right values
    values = conf[reqConf]
    # Read the optional keys
    flags = ""
    ext = ".txt"
    if "flags" in conf:
        flags = conf["flags"]
        if "-j" in flags:
            ext = ".json"
    if "bindir" in conf:
        executable = os.path.join(conf["bindir"], executable)
    # Let's go!
    for algo in conf["algos"]:
        print(f"algo: {algo}")
        for delta in conf["deltas"]:
            print(f"  delta: {delta}")
            for mct in conf["mctrials"]:
                print(f"    mct: {mct}")
                for eps in conf["epsilons"]:
                    print(f"      eps: {eps}")
                    for val in values:
                        args = [executable] + flags
                        if expType == "mult":
                            # the one is for the number of iterations, which
                            # is ignored when running with a fixed multiplier
                            args += ["-m", str(val), "1"]
                        else:  # implied expType == "iter"
                            args += [str(val)]
                        args += [str(eps), str(mct), algo, str(delta), conf["filename"]]
                        print(f"        {expType}: {val}")
                        for rep in range(conf["reps"]):
                            logBase = (
                                f"{expType}_{conf['name']}_{algo}_"
                                f"{delta}_{mct}_{eps}_{val}_{rep}"
                            )
                            print(f"          rep: {rep}")
                            try:
                                if "logdir" not in conf:
                                    print("### start of process output ###")
                                    subprocess.run(args, check=True)
                                    print("### end of process output ###")
                                else:
                                    # buffering=1 activates line buffering
                                    # (maybe, because it is not very clear what
                                    # subprocess.run( ) does with these files
                                    # (does it re-open them?)
                                    with open(
                                        os.path.join(
                                            conf["logdir"], logBase + "_out" + ext
                                        ),
                                        mode="wt",
                                        buffering=1,
                                    ) as outl, open(
                                        os.path.join(
                                            conf["logdir"], logBase + "_err.txt"
                                        ),
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
                # end of eps loop
            # end of mct loop
        # end of delta loop
    # end of algo loop


if __name__ == "__main__":
    main()
