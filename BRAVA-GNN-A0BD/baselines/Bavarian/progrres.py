#! /usr/bin/env python3
#
# Process the results of an experiment run using ./progrexp.py
#
# Copyright 2021 Matteo Riondato <rionda@acm.org>
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
import sys
import pandas as pd
import numpy as np
import util
import progrexp


def main():
    # Check the sanity of the command line arguments
    progrexp.checkArgsSanity()
    expType = sys.argv[1]
    confFile = sys.argv[-1]
    # Get the conf
    if expType == "mult":
        reqConfs = ("epsilons", "multipliers")
    else:
        sys.exit("support for 'iter' experiment type not implemented yet")
    conf = util.getConf(confFile, reqConfs + ("logdir", "resdir", "exactdir"))
    # Collect exact results, if present.
    exactPath = os.path.join(conf["exactdir"], conf["name"] + ".json")
    if os.path.exists(exactPath):
        print("Collecting exact results")
        exact = util.parseJSONFromFile(exactPath)
    else:
        exact = {"results": None, "total_time": 0}
    # Collect the approximate results
    df = util.getResults(conf, reqConfs, expType, exact["results"])
    df.drop(columns=["delta", "mctrials"])
    # Analyze the results
    table = df.groupby(["method", "set_epsilon", "multiplier"]).agg(
        # eps_min=pd.NamedAgg(column="epsilon", aggfunc=np.min),
        eps_median=pd.NamedAgg(column="epsilon", aggfunc=np.median),
        # eps_max=pd.NamedAgg(column="epsilon", aggfunc=np.max),
        # epsratio_min=pd.NamedAgg(column="epsratio", aggfunc=np.min),
        epsratio_median=pd.NamedAgg(column="epsratio", aggfunc=np.median),
        # epsratio_max=pd.NamedAgg(column="epsratio", aggfunc=np.max),
        # nomc_epsilon_min=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.min),
        nomc_epsilon_median=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.median),
        # nomc_epsilon_max=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.max),
        # samples_min=pd.NamedAgg(column="samples", aggfunc=np.min),
        samples_median=pd.NamedAgg(column="samples", aggfunc=np.median),
        # samples_max=pd.NamedAgg(column="samples", aggfunc=np.max),
        # nomc_samples_min=pd.NamedAgg(column="nomc_samples", aggfunc=np.min),
        # nomc_samples_median=pd.NamedAgg(column="nomc_samples", aggfunc=np.median),
        # nomc_samples_max=pd.NamedAgg(column="nomc_samples", aggfunc=np.max),
        # iterations_min=pd.NamedAgg(column="iterations", aggfunc=np.min),
        iterations_median=pd.NamedAgg(column="iterations", aggfunc=np.median),
        # iterations_max=pd.NamedAgg(column="iterations", aggfunc=np.max),
        # max_iterations_min=pd.NamedAgg(column="max_iterations", aggfunc=np.min),
        max_iterations_median=pd.NamedAgg(column="max_iterations", aggfunc=np.median),
        # max_iterations_max=pd.NamedAgg(column="max_iterations", aggfunc=np.max),
        # runtime_min=pd.NamedAgg(column="total_time", aggfunc=np.min),
        # runtime_max=pd.NamedAgg(column="total_time", aggfunc=np.max),
        # runtime_median=pd.NamedAgg(column="total_time", aggfunc=np.median),
    )
    outfile = os.path.join(conf["resdir"], f'progressive-{conf["name"]}.tex')
    with open(outfile, 'wt') as out:
        out.write(table.to_latex(multirow=True))


if __name__ == "__main__":
    main()
