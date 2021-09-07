import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
n_array = [int(i) for i in np.linspace(10, 300, 50)]
burn_in = 10000
chain_size = 500000

for color, K in [("#E29D26", 2), ("#5D80B4", 20)]:
    var_theo_array = []
    var_simu_array = []

    for n in n_array:
        delta_x = 1.0 / n


        def proba(p):
            rand = np.random.rand()
            if rand <= 1 - p:
                return p + delta_x
            elif rand <= 1 - p / (K - 1):
                return p
            else:
                return p - delta_x


        var_theo_array.append((K - 1) / (K * K * n))

        x = 1 - 1 / K
        x_chain = []
        for t in range(burn_in):
            x = proba(x)
        for i in range(chain_size):
            x = proba(x)
            x_chain.append(x)

        var_simu_array.append(np.var(x_chain))

    plt.plot(n_array, var_simu_array, color=color, alpha=0.3, label="Simulations for $K={0}$".format(K))
    theo_label = "$ \\frac{K-1}{K^2 \\times n }$ " + " for $K={0}$".format(K)
    plt.plot(n_array, var_theo_array, linestyle="--", color=color, label=theo_label)

plt.xlabel("number of sites (n)")
plt.ylabel("Phenotypic variance at mutational equilibrium")
plt.legend()
plt.tight_layout()
plt.savefig("mutational-variance.pdf", format="pdf")
plt.clf()
plt.close('all')
