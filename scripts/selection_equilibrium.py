import numpy as np
import scipy.stats
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default="theoretical_eq", type=str, dest="output")
    parser.add_argument('--exon_size', default=300, type=int, dest="n")
    parser.add_argument('--population_size', default=10000, type=float, dest="population_size")
    parser.add_argument('--alpha', default=-118, type=float, dest="alpha")
    parser.add_argument('--gamma', default=1.0, type=float, dest="gamma")
    parser.add_argument('--beta', default=1.686, type=float, dest="beta")
    args, unknown = parser.parse_known_args()
    delta_x = 1.0 / args.n


    def fitness(p):
        delta_g = args.alpha + args.gamma * args.n * p
        edg = np.exp(args.beta * delta_g)
        w = np.log(1 / (1 + edg))
        return np.exp(4 * args.population_size * w)


    def sel_coeff(p):
        delta_g = args.alpha + args.gamma * args.n * p
        edg = np.exp(args.beta * delta_g)
        return - args.gamma * args.beta * edg / (1 + edg)


    def p_fix(s):
        S = 4 * args.population_size * s
        return S / (1 - np.exp(-S))


    for K in [2, 20]:

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


        plt.figure()
        points = args.n * 10
        x_array = np.linspace(0, 1, points)
        normal = scipy.stats.norm(1 - 1 / K, np.sqrt((K - 1) / (args.n * K * K))).logpdf
        p_array = np.array([normal(x) for x in x_array])
        f = np.array([np.log(fitness(x)) for x in x_array])
        p_array = np.exp(p_array + f, dtype=np.longdouble)
        p_array /= np.sum(p_array, dtype=np.longdouble)
        p_array *= points

        x = 1 - 1 / K
        burn_in = 10000
        chain_size = 1000000
        x_chain = []

        for t in range(burn_in):
            x = proba(x)

        for i in range(chain_size):
            x = proba(x)
            x_chain.append(x)

        plt.hist(x_chain, density=True, bins=6 if K > 10 else 20, color="#5D80B4",
                 label="Simulation for $K={0}$ and $n={1}$".format(K, args.n))
        plt.plot(x_array, p_array, color="#E29D26", label="Theoretical for $K={0}$ and $n={1}$".format(K, args.n))
        plt.ylabel("Density")
        plt.xlabel("Phenotype at equilibrium ($x$)")
        plt.xlim((0.2, 0.6))
        plt.legend()
        plt.tight_layout()
        plt.savefig("selection-equilibrium-K_{0}.pdf".format(K), format="pdf")
        plt.clf()
        plt.close('all')
