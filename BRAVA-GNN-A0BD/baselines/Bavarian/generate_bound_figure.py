#! /usr/bin/env python3

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)

# Results obtained using bin/schedule -f … with the right parameters (see text)
zerozerofive = [1560, 1240, 984, 856, 772]
zerozerotwo = [3872, 3120, 2480, 2160, 1936]
zerozeroone = [7744, 6240, 4960, 4320, 3872]
k = [10, 20, 50, 100, 200]

plt.plot(
    k,
    zerozeroone,
    label=r"$\bar{\varepsilon}=0.01$",
    marker="s",
    linewidth=2,
    markersize=10,
)
plt.plot(
    k,
    zerozerotwo,
    label=r"$\bar{\varepsilon}=0.02$",
    marker="^",
    linewidth=2,
    markersize=12,
)
plt.plot(
    k,
    zerozerofive,
    label=r"$\bar{\varepsilon}=0.05$",
    marker="o",
    linewidth=2,
    markersize=10,
)
plt.title(r"Minimum sample size $\mathsf{m}_*(k,0.1,\bar{\varepsilon})$")
plt.xlabel(r"MC-trials $k$")
plt.ylabel(r"$\mathsf{m}_*(k,0.1,\bar{\varepsilon})$")
plt.legend(fontsize=16)
plt.savefig("minsize.pdf", bbox_inches="tight")


# plt.minorticks_on()
# plt.grid(which='both', axis='y', linestyle='--')
