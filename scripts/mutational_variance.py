import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
n_array = [int(i) for i in np.linspace(10, 300, 50)]
burn_in = 10000
chain_size = 500000

for color, K in [("#5D80B4", 2), ("#EB6231", 20), ("#857BA1", 100)]:
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


        x_array = np.linspace(0, 1, n)
        if K == 2:
            p_array = np.array([np.exp(i * (2 * (K - 1) - i * K) * n / (K - 1)) for i in x_array])
        else:
            p_array = np.array([(1 - i) * np.exp(2 * i * n) for i in x_array])
        p_array /= np.sum(p_array)
        x = np.mean([x_array[i] * p * n for i, p in enumerate(p_array)])
        var_theo_array.append(np.sum([(x_array[i] - x) ** 2 * p for i, p in enumerate(p_array)]))

        x_chain = []
        for t in range(burn_in):
            x = proba(x)
        for i in range(chain_size):
            x = proba(x)
            x_chain.append(x)

        var_simu_array.append(np.var(x_chain))

    if K == 0:
        plt.plot(n_array, var_theo_array, color="#8FB03E", label="Theoretical ($K={0}$)".format(K))
    plt.plot(n_array, var_simu_array, color=color, label="Simulated ($K={0}$)".format(K))

plt.plot(n_array, [1 / (4 * n) for n in n_array], linestyle="--", color="#E29D26", label="$\\frac{\\delta x}{4}$")
plt.xlabel("number of sites (n)")
plt.ylabel("Phenotypic variance at mutational equilibrium")
plt.legend()
plt.tight_layout()
plt.savefig("mutational-variance.pdf", format="pdf")
plt.clf()
plt.close('all')
