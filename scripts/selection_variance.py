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

    for color, K in [("#E29D26", 2), ("#5D80B4", 20)]:
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
    plt.ylabel("Phenotypic variance at mutation-selection equilibrium")
    plt.legend()
    plt.tight_layout()
    plt.savefig("selection-variance.pdf", format="pdf")
    plt.clf()
    plt.close('all')
