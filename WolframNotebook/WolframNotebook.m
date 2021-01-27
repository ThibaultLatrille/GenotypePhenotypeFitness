ClearAll["Global`*"]

S[x_, K_] := -Log[(1-x)/x] - Log[K-1]

w[x_, K_] = (1-x)*S[x, K]/(1-Exp[-S[x, K]])-x/(K-1)*S[x, K]/(1-Exp[S[x, K]]) + x *(K-2)/(K-1) // FullSimplify 

Plot[{w[x, 2], w[x, 5], w[x, 10], w[x, 20]}, {x, 0, 1},
    PlotLegends -> {"K=2", "K=5", "K=10", "K=20"}, AxesLabel -> {"x", "ω"}]

Export["manuscript/artworks/wolfram_omega_x.pdf", %]

Series[w[x, K], {x, 0, 1}] // FullSimplify

NSolve[w[x, 2] == 1 && 0 < x && x < 1, x]

NSolve[w[x, 3] == 1 && 0 < x && x < 1, x]

NSolve[w[x, 4] == 1 && 0 < x && x < 1, x]

NSolve[w[x, 8] == 1 && 0 < x && x < 1, x]

dw[x_, K_] = D[w[x, K], x] // FullSimplify

Plot[{dw[x, 2], dw[x, 5], dw[x, 10], dw[x, 20]}, {x, 0, 1},
    PlotLegends -> {"K=2", "K=5", "K=10", "K=20"}, AxesLabel -> {"x", "dω/dx"}]

Export["manuscript/artworks/wolfram_domegadx_x.pdf", %]

NSolve[dw[x, 2] == 0 && 0 < x && x < 1, x]

NSolve[dw[x, 3] == 0 && 0 < x && x < 1, x]

NSolve[dw[x, 4] == 0 && 0 < x && x < 1, x]

NSolve[dw[x, 5] == 0 && 0 < x && x < 1, x]

dw[(K-1)/K, K] // ExpandAll

Limit[dw[x, K], K -> \[Infinity]]
