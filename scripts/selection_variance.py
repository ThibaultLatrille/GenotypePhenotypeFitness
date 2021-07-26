import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
n_array = [int(i) for i in np.linspace(10, 300, 50)]
burn_in = 10000
chain_size = 500000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default="theoretical_eq", type=str, dest="output")
    parser.add_argument('--exon_size', default=300, type=int, dest="n")
    parser.add_argument('--population_size', default=10000, type=float, dest="population_size")
    parser.add_argument('--alpha', default=-118, type=float, dest="alpha")
    parser.add_argument('--gamma', default=1.0, type=float, dest="gamma")
    parser.add_argument('--beta', default=1.686, type=float, dest="beta")
    args, unknown = parser.parse_known_args()

    for color, K in [("#5D80B4", 2), ("#EB6231", 20), ("#857BA1", 100)]:
        var_theo_array = []
        var_simu_array = []

        for n in n_array:
            delta_x = 1.0 / n

            def fitness(p):
                delta_g = args.alpha + args.gamma * n * p
                edg = np.exp(args.beta * delta_g)
                w = np.log(1 / (1 + edg))
                return np.exp(4 * args.population_size * w)


            def sel_coeff(p):
                delta_g = args.alpha + args.gamma * n * p
                edg = np.exp(args.beta * delta_g)
                return - args.gamma * args.beta * edg / (1 + edg)


            def p_fix(s):
                S = 4 * args.population_size * s
                if abs(S) < 1e-4:
                    return 1 + S / 2
                else:
                    return S / (1 - np.exp(-S))


            def proba(p):
                rand = np.random.rand()
                sc = sel_coeff(p)
                q_rhs = p_fix(sc) * (1 - p)
                q_stay = p * (K - 2) / (K - 1)
                q_lhs = p_fix(-sc) * p / (K - 1)
                total = (q_rhs + q_lhs + q_stay)
                if rand <= q_rhs / total:
                    return p + delta_x
                elif rand <= (q_rhs + q_stay) / total:
                    return p
                else:
                    return p - delta_x


            x_array = np.linspace(0, 1, n)
            p_array = np.array([fitness(i) * np.exp(2 * i * (1 - i) * n) for i in x_array])
            p_array /= np.sum(p_array)
            var_theo_array.append(np.sum([(x_array[i] - 0.5) ** 2 * p for i, p in enumerate(p_array)]))

            x = np.mean([x_array[i] * p * n for i, p in enumerate(p_array)])

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

    plt.plot(n_array, [1 / (4 * n) for n in n_array], color="#E29D26", linestyle="--", label="$V=\\delta x / 4$")
    plt.xlabel("number of sites (n)")
    plt.ylabel("Phenotypic variance at mutation-selection equilibrium")
    plt.legend()
    plt.tight_layout()
    plt.savefig("selection-variance.pdf", format="pdf")
    plt.clf()
    plt.close('all')
