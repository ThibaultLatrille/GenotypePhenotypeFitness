import numpy as np
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


    for K in [2, 100]:

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
        x_array = np.linspace(0, 1, args.n + 1)
        if K == 2:
            p_array = np.array([fitness(i) * np.exp(2 * i * (1 - i) * args.n) for i in x_array])
        else:
            p_array = np.array([fitness(i) * (1 - i) * np.exp(2 * i * args.n) for i in x_array])
        p_array /= np.sum(p_array)
        p_array *= args.n

        x = np.mean([x_array[i] * p for i, p in enumerate(p_array)])
        burn_in = 1000
        chain_size = 100000
        x_chain = []

        for t in range(burn_in):
            x = proba(x)

        for i in range(chain_size):
            x = proba(x)
            x_chain.append(x)

        plt.hist(x_chain, density=True, bins=15, color="#5D80B4", label="Simulated ($K={0},~n={1}$)".format(K, args.n))
        plt.plot(x_array, p_array, color="#E29D26", label="Theoretical ($K={0},~n={1}$)".format(K, args.n))
        plt.ylabel("Density")
        plt.xlabel("Phenotype at equilibrium ($x$)")
        plt.xlim((0, 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig("selection-equilibrium-K_{0}.pdf".format(K), format="pdf")
        plt.clf()
        plt.close('all')
