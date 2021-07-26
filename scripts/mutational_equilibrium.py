import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()

for K in [2, 100]:
    n = 300
    delta_x = 1.0 / n
    plt.figure()
    x_array = np.linspace(0, 1, n)
    if K == 2:
        p_array = np.array([np.exp(i * (2 * (K - 1) - i * K) * n / (K - 1)) for i in x_array])
    else:
        p_array = np.array([(1 - i) * np.exp(2 * i * n) for i in x_array])
    p_array /= np.sum(p_array)
    p_array *= n


    def proba(p):
        rand = np.random.rand()
        if rand <= 1 - p:
            return p + delta_x
        elif rand <= 1 - p / (K - 1):
            return p
        else:
            return p - delta_x


    x = np.mean([x_array[i] * p for i, p in enumerate(p_array)])
    burn_in = 1000
    chain_size = 100000
    x_chain = []

    for t in range(burn_in):
        x = proba(x)

    for i in range(chain_size):
        x = proba(x)
        x_chain.append(x)

    plt.hist(x_chain, density=True, bins=15, color="#5D80B4", label="Simulated ($K={0},~n={1}$)".format(K, n))
    plt.plot(x_array, p_array, color="#E29D26", label="Theoretical ($K={0},~n={1}$)".format(K, n))
    plt.ylabel("Density")
    plt.xlabel("Phenotype at equilibrium ($x$)")
    plt.xlim((0, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig("mutational-equilibrium-K_{0}.pdf".format(K), format="pdf")
    plt.clf()
    plt.close('all')
