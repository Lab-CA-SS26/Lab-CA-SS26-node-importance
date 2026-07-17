#! /usr/bin/env python3
#
# Process the results of an experiment run using ./staticexp.py
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
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import staticexp
import util

# Setup matplotlib
plt.rc("text", usetex=True)  # use latex
plt.rc("font", family="sans-serif", size="15")  # 'default' fontsize
plt.rc("axes", titlesize=18)  # fontsize of the axes title
plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
plt.rc("ytick", labelsize=18)  # fontsize of the tick labels
plt.rc("legend", fontsize=12)  # fontsize of the legend
plt.rc("figure", titlesize=18)  # fontsize of the figure title
plt.rcParams["legend.markerscale"] = 0.5  # smaller markers in legends
plt.rcParams["legend.borderaxespad"] = 0.2  # smaller border with axes.
plt.rcParams["legend.borderpad"] = 0.2  # smaller inner border .
plt.rcParams["legend.labelspacing"] = 0.2  # smaller vertical space between labels.
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{siunitx}"
    r"\sisetup{detect-all}"  # force siunitx to use your fonts
    r"\usepackage{helvet}"  # set the normal font here
    r"\usepackage{sansmath}"  # load sansmath so math -> helvet
    r"\sansmath"  # tell latex to actually use sansmath
)


# epsilon vs mctrials
def figureEpsVsMCTrials(df, conf):
    for method, methodrows in df.groupby(["method"]):
        # A bit of pandas magic:
        # 1) group the rows by (delta, set_samples, mctrials)
        # 2) for each group, get the mean of the epsilon values;
        # 3) group by the result only by (delta, set_samples), so we can plot
        # epsilon vs mctrials, one curve for each value of 'set_samples'.
        rows = (
            methodrows.groupby(["delta", "set_samples", "mctrials"])
            .agg(
                eps_min=pd.NamedAgg(column="epsilon", aggfunc=np.min),
                eps_max=pd.NamedAgg(column="epsilon", aggfunc=np.max),
                eps_median=pd.NamedAgg(column="epsilon", aggfunc=np.median),
                nomc_eps_min=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.min),
            )
            .groupby(["delta", "set_samples"])
        )
        fig, ax = plt.subplots(1, 1)
        for name, grouprows in rows:
            delta, samplesize = name
            ax.fill_between(
                grouprows.index.get_level_values("mctrials"),
                grouprows["eps_min"],
                grouprows["eps_max"],
                alpha=0.5,
            )
            ax.plot(
                grouprows.index.get_level_values("mctrials"),
                grouprows["eps_median"],
                label=r"$m=$" f"{samplesize}",
            )
        # 20201015 MR: Commenting out because it only plots the nomc curve for
        # the last sample size, which, although interesting, is not really
        # informative, and plotting the nomc curve for _all_ the sample sizes
        # makes for a very messy figure.
        #
        # Python scoping rules allow us to use grouprows outside of the for
        # loop with the last value it had, which makes for an easy plotting.
        # ax.plot(
        #    grouprows.index.get_level_values("mctrials"),
        #    grouprows["nomc_eps_min"],
        #    label=r"$m=$" f"{samplesize} nomc",
        #    alpha=0.5,
        #    linestyle="dashed",
        # )
        # We want the entry for the nomc curve to be the first one
        # in the legend.
        # handles, labels = ax.get_legend_handles_labels()
        # newHandles = [handles[-1]] + [x for x in handles[:-1]]
        # newLabels = [labels[-1]] + [x for x in labels[:-1]]
        # ax.legend(newHandles, newLabels)
        plt.ylabel(r"error bound $\varepsilon$")
        plt.xlabel(r"MC trials $k$")
        plt.title(r"$\varepsilon$ vs $k$" f'-- {conf["name"]} -- {method}')
        plt.tight_layout(pad=0.2)
        # TODO: Consider adding other details to filename.
        figName = os.path.join(conf["resdir"], f'epsvsmct-{conf["name"]}-{method}.pdf')
        try:
            plt.savefig(figName, bbox_inches="tight")
        except OSError as e:
            sys.exit(f"Error writing figure: {e}")
        plt.clf()


# time vs mctrials
def figureTimeVsMCTrials(df, conf, exactTime):
    # The code is essentially the same as for figureEpsVsMCTrials but we
    # aggregate the time over the runs, instead of the epsilon, and we plot an
    # additional curve for the exact time.
    for method, methodrows in df.groupby(["method"]):
        rows = (
            methodrows.groupby(["delta", "set_samples", "mctrials"])
            .agg({"total_time": np.mean})
            .groupby(["delta", "set_samples"])
        )
        for name, grouprows in rows:
            delta, samplesize = name
            plt.plot(
                grouprows.index.get_level_values("mctrials"),
                grouprows["total_time"],
                label=r"$m=$" f"{samplesize}",
            )
        if exactTime != 0:
            plt.plot(
                grouprows.index.get_level_values("mctrials"),
                [exactTime] * len(conf["mctrials"]),
                label="exact",
            )
        plt.legend()
        plt.ylabel("time (seconds)")
        plt.xlabel(r"MC trials $k$")
        plt.title(r"time vs MC trials $k$ -- " f'{conf["name"]} -- {method}')
        plt.tight_layout(pad=0.2)
        # TODO: Consider adding other details to filename.
        figName = os.path.join(conf["resdir"], f'timevsmct-{conf["name"]}-{method}.pdf')
        try:
            plt.savefig(figName, bbox_inches="tight")
        except OSError as e:
            sys.exit(f"Error writing figure: {e}")
        plt.clf()


# epsilon vs size for a single estimator
def figureEpsVsSize(df, conf):
    mctrialsToPlot = 100
    epsvssz = (
        df.groupby(["method", "delta", "set_samples", "mctrials"])
        .agg(
            eps_min=pd.NamedAgg(column="epsilon", aggfunc=np.min),
            eps_max=pd.NamedAgg(column="epsilon", aggfunc=np.max),
            eps_median=pd.NamedAgg(column="epsilon", aggfunc=np.median),
            nomc_eps_min=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.min),
            err_max_min=pd.NamedAgg(column="err_max", aggfunc=np.min),
            err_max_max=pd.NamedAgg(column="err_max", aggfunc=np.max),
            err_max_median=pd.NamedAgg(column="err_max", aggfunc=np.median),
        )
        .groupby(["method", "delta", "mctrials"], as_index=False)
    )
    fig, ax = plt.subplots(1, 1)
    for name, group in epsvssz:
        method, delta, mctrials = name
        if mctrials == mctrialsToPlot:
            if method == "ab":
                color = "tab:blue"
                marker = "o"
            elif method == "rk":
                color = "tab:orange"
                marker = "^"
            elif method == "bp":
                color = "tab:cyan"
                marker = "s"
            else:
                raise ValueError("Unrecognized method")
            ax.fill_between(
                group.index.get_level_values("set_samples"),
                group["eps_min"],
                group["eps_max"],
                color=color,
                alpha=0.5,
            )
            ax.plot(
                group.index.get_level_values("set_samples"),
                group["eps_median"],
                label=f"{method} -- this work",
                color=color,
                marker=marker,
                linewidth=2,
                markersize=11,
            )
            if group["err_max_max"][0] != 0:
                ax.fill_between(
                    group.index.get_level_values("set_samples"),
                    group["err_max_min"],
                    group["err_max_max"],
                    color=color,
                    alpha=0.5,
                )
                ax.plot(
                    group.index.get_level_values("set_samples"),
                    group["err_max_median"],
                    label=f"{method} -- emp. max. error",
                    linestyle="dotted",
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=11,
                )
            ax.plot(
                group.index.get_level_values("set_samples"),
                group["nomc_eps_min"],
                label=f"{method} -- prev. work",
                linestyle="dashed",
                color=color,
                marker=marker,
                linewidth=2,
                markersize=11,
            )
    # switch order of legend so it looks like: bp-actual, ab-actual, rk-actual
    # (?) bp, ab, rk, bp-prev, ab-prev, rk-prev,
    # handles, labels = ax.get_legend_handles_labels()
    # if group["err_max_max"][0] != 0:
    #    newHandles = [
    #        handles[4],
    #        handles[1],
    #        handles[7],
    #        handles[3],
    #        handles[0],
    #        handles[6],
    #        handles[5],
    #        handles[2],
    #        handles[8],
    #    ]
    #    newLabels = [
    #        labels[4],
    #        labels[1],
    #        labels[7],
    #        labels[3],
    #        labels[0],
    #        labels[6],
    #        labels[5],
    #        labels[2],
    #        labels[8],
    #    ]
    # else:
    #    newHandles = [
    #        handles[2],
    #        handles[0],
    #        handles[4],
    #        handles[3],
    #        handles[1],
    #        handles[5],
    #    ]
    #    newLabels = [labels[2], labels[0], labels[4], labels[3], labels[1], labels[5]]
    # ax.legend(newHandles, newLabels)
    plt.ylabel(r"error bound $\varepsilon$")
    plt.xlabel(r"sample size $m$")
    plt.title(r"$\varepsilon$ vs $m$ -- " f'{conf["name"]} -- k={mctrialsToPlot}')
    plt.tight_layout(pad=0.2)
    # TODO: Consider adding other details to filename.
    figName = os.path.join(conf["resdir"], f'epsvssz-{conf["name"]}.pdf')
    try:
        plt.savefig(figName, bbox_inches="tight")
    except OSError as e:
        sys.exit(f"Error writing figure: {e}")
    plt.clf()


# epsilon vs size for a single estimator
def figureEpsVsSizeSingleEstimator(df, conf):
    # The code is essentially the same as for figureEpsVsMCTrials but we plot
    # the epsilon vs the sample size ('set_samples') instead of the MC trials.
    for method, methodrows in df.groupby(["method"]):
        rows = (
            methodrows.groupby(["delta", "set_samples", "mctrials"])
            .agg(
                eps_min=pd.NamedAgg(column="epsilon", aggfunc=np.min),
                eps_max=pd.NamedAgg(column="epsilon", aggfunc=np.max),
                eps_median=pd.NamedAgg(column="epsilon", aggfunc=np.median),
                # The non-MC epsilon does not depend on the number of trials,
                # only on the number of samples but we aggregate it anyway,
                # because it makes plotting easier (see below)
                nomc_eps_min=pd.NamedAgg(column="nomc_epsilon", aggfunc=min),
            )
            .groupby(["delta", "mctrials"])
        )
        fig, ax = plt.subplots(1, 1)
        for name, grouprows in rows:
            delta, mctrials = name
            ax.fill_between(
                grouprows.index.get_level_values("set_samples"),
                grouprows["eps_min"],
                grouprows["eps_max"],
                alpha=0.5,
            )
            ax.plot(
                grouprows.index.get_level_values("set_samples"),
                grouprows["eps_median"],
                label=r"$k=$" f"{mctrials}",
            )
        # Python scoping rules allow us to use grouprows outside of the for
        # loop with the last value it had, which makes for an easy plotting.
        ax.plot(
            grouprows.index.get_level_values("set_samples"),
            grouprows["nomc_eps_min"],
            label="prev. work",
        )
        # We want the entry for the nomc curve to be the first one
        # in the legend.
        handles, labels = ax.get_legend_handles_labels()
        newHandles = [handles[-1]] + [x for x in handles[:-1]]
        newLabels = [labels[-1]] + [x for x in labels[:-1]]
        ax.legend(newHandles, newLabels)
        plt.ylabel(r"error bound $\varepsilon$")
        plt.xlabel(r"sample size $m$")
        plt.title(
            r"error bound $\varepsilon$ vs sample size $m$ -- "
            f'{conf["name"]} -- {method}'
        )
        plt.tight_layout(pad=0.2)
        # TODO: Consider adding other details to filename.
        figName = os.path.join(
            conf["resdir"], f'epsvsszsingle-{conf["name"]}-{method}.pdf'
        )
        try:
            plt.savefig(figName, bbox_inches="tight")
        except OSError as e:
            sys.exit(f"Error writing figure: {e}")
        plt.clf()


# time vs size
def figureTimeVsSize(df, conf, exactTime):
    mctrialsToPlot = 100
    timevssz = (
        df.groupby(["method", "delta", "set_samples", "mctrials"])
        .agg(
            time_min=pd.NamedAgg(column="total_time", aggfunc=np.min),
            time_max=pd.NamedAgg(column="total_time", aggfunc=np.max),
            time_median=pd.NamedAgg(column="total_time", aggfunc=np.median),
            #             {"total_time": np.mean}
        )
        .groupby(["method", "delta", "mctrials"], as_index=False)
    )
    fig, ax = plt.subplots(1, 1)
    div = 1
    if exactTime != 0:
        div = exactTime / 100
    for name, group in timevssz:
        method, delta, mctrials = name
        if mctrials == mctrialsToPlot:
            if method == "ab":
                color = "tab:blue"
                marker = "o"
            elif method == "rk":
                color = "tab:orange"
                marker = "^"
            elif method == "bp":
                color = "tab:cyan"
                marker = "s"
            else:
                raise ValueError("Unrecognized method")
            ax.fill_between(
                group.index.get_level_values("set_samples"),
                group["time_min"] / div,
                group["time_max"] / div,
                color=color,
                alpha=0.5,
            )
            ax.plot(
                group.index.get_level_values("set_samples"),
                group["time_median"] / div,
                label=f"{method}",
                color=color,
                marker=marker,
                linewidth=2,
                markersize=10,
            )
    if exactTime != 0:
        ax.plot(
            group.index.get_level_values("set_samples"),
            [100] * len(conf["samples"]),
            label="exact",
            color="tab:purple",
            linewidth=2,
        )
    #    # switch order of legend so it looks like: exact, bp, rk, ab
    #    handles, labels = ax.get_legend_handles_labels()
    #    newHandles = [handles[3], handles[1], handles[2], handles[0]]
    #    newLabels = [labels[3], labels[1], labels[2], labels[0]]
    # else:
    #    # switch order of legend so it looks like: bp, rk, ab
    #    handles, labels = ax.get_legend_handles_labels()
    #    newHandles = [handles[1], handles[2], handles[0]]
    #    newLabels = [labels[1], labels[2], labels[0]]
    # ax.legend(newHandles, newLabels)
    if exactTime != 0:
        plt.ylabel("time (\\% of exact)")
    else:
        plt.ylabel("time (seconds)")
    plt.xlabel(r"sample size $m$")
    plt.title(r"runtime vs $m$ -- " f'{conf["name"]} -- k={mctrialsToPlot}')
    plt.tight_layout(pad=0.2)
    # TODO: Consider adding other details to filename.
    figName = os.path.join(conf["resdir"], f'timevssz-{conf["name"]}.pdf')
    try:
        plt.savefig(figName, bbox_inches="tight")
    except OSError as e:
        sys.exit(f"Error writing figure: {e}")
    plt.clf()


# time vs size single estimator
def figureTimeVsSizeSingle(df, conf, exactTime):
    # The code is essentially the same as for figureEpsVsSize but we aggregate
    # the time over the runs, instead of the epsilon, and we plot an additional
    # curve for the exact time.
    for method, methodrows in df.groupby(["method"]):
        rows = (
            methodrows.groupby(["delta", "set_samples", "mctrials"])
            .agg(
                time_min=pd.NamedAgg(column="total_time", aggfunc=np.min),
                time_max=pd.NamedAgg(column="total_time", aggfunc=np.max),
                time_median=pd.NamedAgg(column="total_time", aggfunc=np.median),
                time_mean=pd.NamedAgg(column="total_time", aggfunc=np.mean),
            )
            .groupby(["delta", "mctrials"])
        )
        for name, grouprows in rows:
            delta, mctrials = name
            plt.plot(
                grouprows.index.get_level_values("set_samples"),
                grouprows["time_mean"],
                label=r"$k=$" f"{mctrials}",
            )
        if exactTime != 0:
            plt.plot(
                grouprows.index.get_level_values("set_samples"),
                [exactTime] * len(conf["samples"]),
                label="exact",
            )
        plt.legend()
        plt.ylabel("time (seconds)")
        plt.xlabel(r"sample size $m$")
        plt.title(r"time vs sample size $m$ -- " f'{conf["name"]} -- {method}')
        plt.tight_layout(pad=0.2)
        # TODO: Consider adding other details to filename.
        figName = os.path.join(
            conf["resdir"], f'timevsszsingle-{conf["name"]}-{method}.pdf'
        )
        try:
            plt.savefig(figName, bbox_inches="tight")
        except OSError as e:
            sys.exit(f"Error writing figure: {e}")
        plt.clf()


def figureErrorStatsVsSize(df, conf):
    pass


# composition of the error
def figureEpsComposition(df, conf):
    def twicemean(x):
        return 2 * np.mean(x)

    # We generate one figure per (method, set_samples)
    for methodsamples, methodrows in df.groupby(["method", "set_samples"]):
        method, samples = methodsamples
        rows = (
            methodrows.groupby(["delta", "mctrials"])
            .agg(
                {
                    "epsilon": np.mean,
                    "mcera": twicemean,
                    "erabound": twicemean,
                    "rabound": twicemean,
                    # the nomc_epsilon should be the same in every run (with
                    # the same parameters). We take the min just for
                    # aggregation.
                    "nomc_epsilon": np.min,
                }
            )
            .groupby(["delta"])
        )
        for name, grouprows in rows:
            # Stacked bar idea from
            # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
            # See that page if you want to add confidence interval bars to the
            # sub columns.
            xloc = np.arange(
                0, len(grouprows.index.get_level_values("mctrials")) / 5, 0.2
            )
            plt.bar(
                xloc,
                grouprows["mcera"],
                0.175,
                label=r"$2 \times k\text{-MCERA}$",
                color="tab:blue",
            )
            plt.bar(
                xloc,
                grouprows["erabound"] - grouprows["mcera"],
                0.175,
                label=r"to-$2\times\text{ERA}$ bound $\rho$",
                bottom=grouprows["mcera"],
                color="tab:orange",
            )
            plt.bar(
                xloc,
                grouprows["rabound"] - grouprows["erabound"],
                0.175,
                label=r"to-$2\times\text{RA}$ bound $r$",
                bottom=grouprows["erabound"],
                color="tab:cyan",
            )
            plt.bar(
                xloc,
                grouprows["epsilon"] - grouprows["rabound"],
                0.175,
                label=r"to-SD bound $\varepsilon$",
                bottom=grouprows["rabound"],
                color="tab:purple",
            )
        # Python scoping rules allow us to use grouprows and xloc outside of
        # the for loop with the last value it had
        plt.xticks(xloc, grouprows.index.get_level_values("mctrials"))
        plt.legend()
        plt.ylabel(r"error bound $\varepsilon$")
        plt.xlabel(r"MC trials $k$")
        plt.title(
            r"$\varepsilon$ decomposition-- "
            f'{conf["name"]} -- {method} -- '
            r"$m=$"
            f"{samples}"
        )
        plt.tight_layout(pad=0.2)
        # TODO: Consider adding other details to filename.
        figName = os.path.join(
            conf["resdir"], f'epscomp-{conf["name"]}-{method}-{samples}.pdf'
        )
        try:
            plt.savefig(figName, bbox_inches="tight")
        except OSError as e:
            sys.exit(f"Error writing figure: {e}")
        plt.clf()


def figureOneMinusEpsVsTime(df, conf, exactTime):
    def meanoneminusx(x):
        return 1 - np.mean(x)

    rows = (
        df.groupby(["method", "delta", "set_samples", "mctrials"])
        .agg({"epsilon": meanoneminusx, "total_time": np.mean})
        .groupby(["method", "delta", "set_samples", "mctrials"])
    )
    for name, group in rows:
        method, delta, samplesize, mctrials = name
        # TODO: we need a better way of assigning colors and alpha that
        # does not depend on hard-coded values
        if method == "ab":
            marker = "o"
        elif method == "rk":
            marker = "s"
        elif method == "bp":
            marker = "*"
        else:
            raise ValueError("Unrecognized method")
        if samplesize == 10000:
            color = (0, 0, 1)
        elif samplesize == 25000 or samplesize == 20000:
            color = (0, 1, 0)
        elif samplesize == 50000 or samplesize == 30000:
            color = (1, 0, 0)
        elif samplesize == 75000 or samplesize == 40000:
            color = (0, 0, 0)
        elif samplesize == 75000 or samplesize == 50000:
            color = (0, 1, 1)
        else:
            raise ValueError("Unrecognized samplesize")
        if mctrials == 2:
            alpha = 0.2
        elif mctrials == 5:
            alpha = 0.4
        elif mctrials == 10:
            alpha = 0.5
        elif mctrials == 50:
            alpha = 0.6
        elif mctrials == 100:
            alpha = 0.7
        elif mctrials == 200:
            alpha = 1
        else:
            raise ValueError("Unrecognized mctrials")
        plt.scatter(
            group["epsilon"],
            group["total_time"],
            label=f"{method}, {samplesize}, {mctrials}",
            marker=marker,
            color=color,
            alpha=alpha,
        )
        if exactTime != 0:
            plt.axhline(y=exactTime, label="exact")
    plt.legend(bbox_to_anchor=(2.2, 0), loc="lower center", ncol=6)
    plt.xlabel("1-epsilon")
    plt.ylabel("time (seconds)")
    plt.title(f'time vs 1-eps -- {conf["name"]}')
    plt.tight_layout(pad=0.2)
    figName = os.path.join(conf["resdir"], f'oneminusepsvstime-{conf["name"]}.pdf')
    try:
        plt.savefig(figName, bbox_inches="tight")
    except OSError as e:
        sys.exit(f"Error writing figure: {e}")
    plt.clf()


def figureEpsVsTime(df, conf):
    mctrialsToPlot = 100
    if mctrialsToPlot not in conf["mctrials"]:
        mctrialsToPlot = max(conf["mctrials"])
    rows = (
        df.groupby(["method", "delta", "timeout", "mctrials"])
        .agg(
            eps_min=pd.NamedAgg(column="epsilon", aggfunc=np.min),
            eps_max=pd.NamedAgg(column="epsilon", aggfunc=np.max),
            eps_median=pd.NamedAgg(column="epsilon", aggfunc=np.median),
            eps_mean=pd.NamedAgg(column="epsilon", aggfunc=np.mean),
            nomc_eps_min=pd.NamedAgg(column="nomc_epsilon", aggfunc=np.min),
        )
        .groupby(["method", "delta", "mctrials"])
    )
    fig, ax = plt.subplots(1, 1)
    for name, group in rows:
        method, delta, mctrials = name
        if mctrials == mctrialsToPlot:
            if method == "ab":
                color = "tab:blue"
                marker = "o"
            elif method == "rk":
                color = "tab:orange"
                marker = "^"
            elif method == "bp":
                color = "tab:cyan"
                marker = "s"
            else:
                raise ValueError("Unrecognized method")
            ax.fill_between(
                conf["timeouts"],
                group["eps_min"],
                group["eps_max"],
                color=color,
                alpha=0.5,
            )
            ax.plot(
                conf["timeouts"],
                group["eps_median"],
                label=f"{method}",
                color=color,
                marker=marker,
                linewidth=2,
                markersize=11,
            )
            # print the bound from the previous work
            # ax.plot(
            #     conf["timeouts"],
            #     group["nomc_eps_min"],
            #     label=f"{method} -- prev. work",
            #     linestyle="dashed",
            #     color=color,
            #     marker=marker,
            #     linewidth=2,
            #     markersize=11,
            # )

    # Add percent sign on x-axis ticks.
    ax.set_xticks(conf["timeouts"])
    ax.set_xticklabels(
        [str(x) + r"\%" for x in conf["timeouts"]],
    )
    # switch order of legend so it looks like: bp, rk, ab
    handles, labels = ax.get_legend_handles_labels()
    newHandles = [handles[1], handles[2], handles[0]]
    newLabels = [labels[1], labels[2], labels[0]]
    # switch order of legend so it looks like:
    # bp-prev, rk-prev, ab-prev, bp, rk, ab
    # newHandles = [
    #     handles[3],
    #     handles[5],
    #     handles[1],
    #     handles[2],
    #     handles[4],
    #     handles[0],
    # ]
    # newLabels = [labels[3], labels[5], labels[1], labels[2], labels[4], labels[0]]
    ax.legend(newHandles, newLabels)
    plt.ylabel(r"error bound $\varepsilon$")
    plt.xlabel(r"time (\% of exact algorithm's runtime)")
    plt.title(r"$\varepsilon$ vs time -- " f'{conf["name"]} -- k={mctrialsToPlot}')
    plt.tight_layout(pad=0.2)
    # TODO: Consider adding other details to filename.
    figName = os.path.join(conf["resdir"], f'epsvstime-{conf["name"]}.pdf')
    try:
        plt.savefig(figName, bbox_inches="tight")
    except OSError as e:
        sys.exit(f"Error writing figure: {e}")
    plt.clf()


def main():
    # Check the sanity of the command line arguments
    staticexp.checkArgsSanity()
    expType = sys.argv[1]
    confFile = sys.argv[-1]
    # Get the conf
    if expType == "time":
        reqConf = "timeouts"
    else:  # implied expType == "size"
        reqConf = "samples"
    conf = util.getConf(confFile, (reqConf, "logdir", "resdir", "exactdir"))
    # Collect exact results, if present.
    exactPath = os.path.join(conf["exactdir"], conf["name"] + ".json")
    if os.path.exists(exactPath):
        print("Collecting exact results")
        exact = util.parseJSONFromFile(exactPath)
        exact["total_time"] /= 1000  # convert to seconds
    else:
        exact = {"results": None, "total_time": 0}
    # Collect the approximate results
    print("Collecting approximate results")
    df = util.getResults(conf, (reqConf,), expType, exact["results"])
    # Analyze the results
    print("Creating images")
    if expType == "size":
        # 20201015 MR: the following figure is not very informative
        # figureEpsVsMCTrials(df, conf)
        # TODO: do we want this one?
        # figureTimeVsMCTrials(df, conf, exact["total_time"])
        # OK
        figureEpsVsSize(df, conf)
        # 20201015 MR: the following figure is not very informative
        # figureEpsVsSizeSingle(df, conf)
        # OK
        figureTimeVsSize(df, conf, exact["total_time"])
        # 20201015 MR: the following figure is not very informative
        # figureTimeVsSizeSingle(df, conf, exact["total_time"])
        # OK
        figureEpsComposition(df, conf)
        # TODO: do we want this?
        # figureOneMinusEpsVsTime(df, conf, exact["total_time"])
    else:  # assuming expType == "time"
        # TODO: Decide what images to create here.
        figureEpsVsTime(df, conf)


if __name__ == "__main__":
    main()
