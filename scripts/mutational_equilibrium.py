import numpy as np
import scipy.stats
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()

for K in [2, 20]:
    n = 300
    delta_x = 1.0 / n
    plt.figure()
    x_array = np.linspace(0, 1, n)
    normal = scipy.stats.norm(1 - 1 / K, np.sqrt((K - 1) / (n * K * K))).pdf
    p_array = np.array([normal(x) for x in x_array])


    def proba(p):
        rand = np.random.rand()
        if rand <= 1 - p:
            return p + delta_x
        elif rand <= 1 - p / (K - 1):
            return p
        else:
            return p - delta_x


    x = 1 - 1 / K
    burn_in = 10000
    chain_size = 1000000
    x_chain = []

    for t in range(burn_in):
        x = proba(x)

    for i in range(chain_size):
        x = proba(x)
        x_chain.append(x)

    plt.hist(x_chain, density=True, bins=15, color="#5D80B4", label="Simulation for $K={0}$ and $n={1}$".format(K, n))
    plt.plot(x_array, p_array, color="#E29D26", label="Theoretical for $K={0}$ and $n={1}$".format(K, n))
    plt.ylabel("Density")
    plt.xlabel("Phenotype at equilibrium ($x$)")
    plt.xlim((0, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig("mutational-equilibrium-K_{0}.pdf".format(K), format="pdf")
    plt.clf()
    plt.close('all')
