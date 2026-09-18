#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Burgers 1D — schema WENO5 + SSP-RK3 e soluzione esatta entropica.

Contiene tutto quello che serve al progetto di Calcolo Scientifico
(Progetto 14) "High order FD discretization of Burgers' equation in 1D with
WENO":

  * il solutore numerico di  u_t + (u^2/2)_x = 0  con
      - flux splitting di Lax-Friedrichs,
      - ricostruzione WENO5 alle interfacce di cella,
      - avanzamento temporale SSP-RK3 con dt adattato alla CFL;
  * la soluzione esatta entropica, calcolata con la formula di
    **Hopf-Lax-Oleinik**:
        u(x,t) = (f*)'(p*),   p* = (x - y*)/t,
        y* = argmin_y [ G(y) + t f*((x-y)/t) ],   G(y) = int u0
    che per Burgers (f*(p) = p^2/2) diventa
        y* = argmin_y [ G(y) + (x-y)^2/(2t) ],    u(x,t) = (x - y*)/t.
    I punti stazionari di J(y) sono le radici dell'equazione delle
    caratteristiche y + u0(y) t = x; fra essi si sceglie pero' quello che
    MINIMIZZA J (condizione di entropia), non quello con u0 minimo.

Uso:
    python3 burgers_weno.py                 # errori L1 + grafico su file
    python3 burgers_weno.py --show          # come sopra, ma apre la finestra
    python3 burgers_weno.py --nx 800 --t 0.9

La funzione `old_min_rule` (in fondo) riproduce la versione precedente —
`u = min{ u0(y) : y + u0(y) t = x }` — e serve solo a documentare, su casi con
piu' urti che interagiscono, che quella regola NON seleziona la soluzione
entropica (si vedano README e test_soluzione_esatta.py).
"""

import argparse
import os

import numpy as np

# ----------------------------------------------------------------------------
# Parametri di default (quelli del progetto)
# ----------------------------------------------------------------------------
AMP = 1.0          # ampiezza della condizione iniziale
XMIN, XMAX = -np.pi, np.pi
NX = 200           # celle della griglia numerica
CFL = 0.45         # coefficiente di Courant (limite pratico 0.5 per WENO5+RK3)
T_FINAL = 0.4      # tempo finale della simulazione
NX_EXACT = 2000    # punti della griglia di ricerca per la soluzione esatta
EXT_FACTOR = 3.0   # estensione minima del dominio di ricerca (in unita' di L)

PAD = 3            # ghost cells necessarie allo stencil di 5 punti di WENO5
EPS = 1e-36        # stabilizzatore dei pesi non lineari


def u0(x):
    """u(x,0) = -sin(2x): la IC del progetto (piu' urti che interagiscono e
    regioni di curvatura diversa)."""
    return AMP * np.sin(np.pi + 2 * x)


# ----------------------------------------------------------------------------
# WENO5 + SSP-RK3
# ----------------------------------------------------------------------------
def weno_reconstruct(f, n, side):
    """Ricostruzione WENO5 del flusso all'interfaccia j+1/2.

    side='m' usa lo stencil orientato a sinistra (f_{j-2}...f_{j+2}),
    side='p' lo stencil orientato a destra   (f_{j-1}...f_{j+3});
    i pesi lineari ottimali si scambiano nei due casi.
    """
    fpad = np.pad(f, (PAD, PAD), mode='wrap')
    if side == 'm':
        v0 = fpad[PAD - 2: PAD - 2 + n]   # f_{j-2}
        v1 = fpad[PAD - 1: PAD - 1 + n]   # f_{j-1}
        v2 = fpad[PAD    : PAD     + n]   # f_j
        v3 = fpad[PAD + 1: PAD + 1 + n]   # f_{j+1}
        v4 = fpad[PAD + 2: PAD + 2 + n]   # f_{j+2}
        p0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2
        p1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3
        p2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4
        b0 = (13.0 / 12.0) * (v0 - 2 * v1 + v2) ** 2 + (1.0 / 4.0) * (v0 - 4 * v1 + 3 * v2) ** 2
        b1 = (13.0 / 12.0) * (v1 - 2 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - v3) ** 2
        b2 = (13.0 / 12.0) * (v2 - 2 * v3 + v4) ** 2 + (1.0 / 4.0) * (3 * v2 - 4 * v3 + v4) ** 2
        d0, d1, d2 = 0.1, 0.6, 0.3
    else:                                  # side == 'p'
        v0 = fpad[PAD - 1: PAD - 1 + n]   # f_{j-1}
        v1 = fpad[PAD    : PAD     + n]   # f_j
        v2 = fpad[PAD + 1: PAD + 1 + n]   # f_{j+1}
        v3 = fpad[PAD + 2: PAD + 2 + n]   # f_{j+2}
        v4 = fpad[PAD + 3: PAD + 3 + n]   # f_{j+3}
        p0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2
        p1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3
        p2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4
        b0 = (13.0 / 12.0) * (v0 - 2 * v1 + v2) ** 2 + (1.0 / 4.0) * (v0 - 4 * v1 + 3 * v2) ** 2
        b1 = (13.0 / 12.0) * (v1 - 2 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - v3) ** 2
        b2 = (13.0 / 12.0) * (v2 - 2 * v3 + v4) ** 2 + (1.0 / 4.0) * (3 * v2 - 4 * v3 + v4) ** 2
        d0, d1, d2 = 0.3, 0.6, 0.1
    w0 = d0 / (b0 + EPS) ** 2
    w1 = d1 / (b1 + EPS) ** 2
    w2 = d2 / (b2 + EPS) ** 2
    return (w0 * p0 + w1 * p1 + w2 * p2) / (w0 + w1 + w2)


def compute_L(u, dx):
    """Residuo spaziale L(u) = -(F_{j+1/2} - F_{j-1/2})/dx (forma conservativa)."""
    n = len(u)
    f = 0.5 * u * u                       # flusso fisico f(u) = u^2/2
    alpha = np.max(np.abs(u))             # velocita' caratteristica massima
    fp = 0.5 * (f + alpha * u)            # splitting di Lax-Friedrichs
    fm = 0.5 * (f - alpha * u)
    F_hat = weno_reconstruct(fp, n, 'm') + weno_reconstruct(fm, n, 'p')
    F_ext = np.concatenate([[F_hat[-1]], F_hat])   # condizioni periodiche
    return -(F_ext[1:] - F_ext[:-1]) / dx


def ssp_rk3(u, dx, dt):
    """Terzo ordine SSP (Shu-Osher) in tre stadi."""
    L0 = compute_L(u, dx)
    u1 = u + dt * L0
    L1 = compute_L(u1, dx)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L1)
    L2 = compute_L(u2, dx)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L2)


def solve_weno(u0_arr, dx, T_final, cfl):
    """Integra in tempo con dt adattato alla condizione CFL."""
    u = u0_arr.copy()
    t = 0.0
    while t < T_final:
        maxU = np.max(np.abs(u))
        dt = cfl * dx / maxU if maxU > 1e-14 else 1e-12
        if t + dt > T_final:
            dt = T_final - t
        u = ssp_rk3(u, dx, dt)
        t += dt
    return u


# ----------------------------------------------------------------------------
# SOLUZIONE ESATTA — formula di Hopf-Lax-Oleinik
# ----------------------------------------------------------------------------
def primitiva(u0_func, ys, u0sy=None):
    """G(y) = int_{ys[0]}^{y} u0(s) ds, con la regola del trapezio."""
    if u0sy is None:
        u0sy = np.nan_to_num(np.array([u0_func(y) for y in ys]), nan=0.0)
    G = np.zeros(np.size(ys))
    G[1:] = np.cumsum(0.5 * (u0sy[1:] + u0sy[:-1]) * np.diff(ys))
    return G


def exact_solution(u0_func, x_vec, t, Ns=NX_EXACT, ext_factor=EXT_FACTOR):
    """Soluzione entropica di Burgers (formula di Hopf-Lax-Oleinik).

    Per f(u) = u^2/2 si ha f*(p) = p^2/2, quindi

        u(x,t) = (x - y*)/t,   y* = argmin_y [ G(y) + (x-y)^2 / (2t) ].

    I punti stazionari del funzionale J(y) = G(y) + (x-y)^2/(2t) sono le
    radici di  g(y) = y + u0(y) t - x  (le caratteristiche che arrivano in x);
    fra i candidati si sceglie quello che MINIMIZZA J — questa e' la
    condizione di entropia, e non equivale a prendere il minimo di u0.
    Se nessuna caratteristica arriva in x (ventaglio di rarefazione) il minimo
    di J cade sul bordo/kink della primitiva e la stessa formula restituisce
    u = x/t, cioe' la nota soluzione del ventaglio.
    """
    x_vec = np.asarray(x_vec, dtype=float)
    if t < 1e-14:                                    # t = 0: la IC stessa
        return np.array([u0_func(xi) for xi in x_vec])

    xmin, xmax = x_vec[0], x_vec[-1]
    L = xmax - xmin
    # max|u0| = max|f'(u0)| sul dominio esteso -> dimensione dell'estensione
    maxU0 = 0.1
    Nc = 500
    for j in range(Nc + 1):
        v = u0_func(xmin - L + j * (3.0 * L / Nc))
        if np.isfinite(v) and abs(v) > maxU0:
            maxU0 = abs(v)
    ext = max(ext_factor * L, 2.0 * maxU0 * t) + 2.0
    xs = np.linspace(xmin - ext, xmax + ext, Ns)
    u0s = np.nan_to_num(np.array([u0_func(y) for y in xs]), nan=0.0)
    G = primitiva(u0_func, xs, u0s)

    def G_at(y):
        i = np.clip(np.searchsorted(xs, y), 1, Ns - 1)
        frac = (y - xs[i - 1]) / (xs[i] - xs[i - 1])
        return G[i - 1] + frac * (G[i] - G[i - 1])

    def J(y, x):
        return G_at(y) + (x - y) ** 2 / (2.0 * t)

    out = np.zeros(x_vec.size)
    for idx, x in enumerate(x_vec):
        gs = xs + u0s * t - x
        # (1) candidati: zeri di g(y) = 0 (punti stazionari di J, bisezione)
        best_y, best_J = None, np.inf
        for j in np.where(gs[:-1] * gs[1:] <= 0)[0]:
            lo, hi, glo = xs[j], xs[j + 1], gs[j]
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                gm = mid + u0_func(mid) * t - x
                if glo * gm <= 0:
                    hi = mid
                else:
                    lo, glo = mid, gm
            y = 0.5 * (lo + hi)
            Jv = J(y, x)
            if Jv < best_J:
                best_y, best_J = y, Jv
        # (2) fallback: se nessuna caratteristica arriva in x (vento di
        #     rarefazione) si minimizza J sulla griglia, con raffinamento
        if best_y is None:
            j0 = int(np.argmin(G + (x - xs) ** 2 / (2.0 * t)))
            a, b = xs[max(0, j0 - 1)], xs[min(Ns - 1, j0 + 1)]
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            c, d = b - phi * (b - a), a + phi * (b - a)
            fc, fd = J(c, x), J(d, x)
            for _ in range(80):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - phi * (b - a)
                    fc = J(c, x)
                else:
                    a, c, fc = c, d, fd
                    d = a + phi * (b - a)
                    fd = J(d, x)
            best_y = 0.5 * (a + b)
        out[idx] = (x - best_y) / t                  # u(x,t) = (x - y*)/t
    return out


def old_min_rule(u0_func, x_vec, t, Ns=NX_EXACT, ext_factor=EXT_FACTOR):
    """Versione PRECEDENTE (non corretta) della soluzione esatta:

        u(x,t) = min{ u0(y) : y + u0(y) t = x }.

    Serve solo a documentare l'errore corretto in questo progetto: quando piu'
    urti interagiscono, il minimo di u0 fra le caratteristiche che arrivano in
    (x,t) non e' la selezione entropica (si veda test_soluzione_esatta.py).
    """
    x_vec = np.asarray(x_vec, dtype=float)
    if t < 1e-14:
        return np.array([u0_func(xi) for xi in x_vec])
    xmin, xmax = x_vec[0], x_vec[-1]
    L = xmax - xmin
    maxU0 = 0.1
    Nc = 500
    for j in range(Nc + 1):
        v = u0_func(xmin - L + j * (3.0 * L / Nc))
        if np.isfinite(v) and abs(v) > maxU0:
            maxU0 = abs(v)
    ext = max(ext_factor * L, 2.0 * maxU0 * t) + 2.0
    xs = np.linspace(xmin - ext, xmax + ext, Ns)
    u0s = np.nan_to_num(np.array([u0_func(y) for y in xs]), nan=0.0)
    out = np.zeros(x_vec.size)
    for idx, x in enumerate(x_vec):
        gs = xs + u0s * t - x
        cands = []
        for j in np.where(gs[:-1] * gs[1:] <= 0)[0]:
            lo, hi, glo = xs[j], xs[j + 1], gs[j]
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                gm = mid + u0_func(mid) * t - x
                if glo * gm <= 0:
                    hi = mid
                else:
                    lo, glo = mid, gm
            u0c = u0_func(0.5 * (lo + hi))
            if np.isfinite(u0c):
                cands.append(u0c)
        out[idx] = min(cands) if cands else u0s[int(np.argmin(np.abs(gs)))]
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Burgers 1D: WENO5 + SSP-RK3 contro la soluzione esatta di Hopf-Lax-Oleinik")
    ap.add_argument("--nx", type=int, default=NX, help="celle della griglia numerica")
    ap.add_argument("--cfl", type=float, default=CFL, help="coefficiente CFL")
    ap.add_argument("--t", type=float, default=T_FINAL, help="tempo finale")
    ap.add_argument("--nx-exact", type=int, default=NX_EXACT,
                    help="punti della griglia di ricerca della soluzione esatta")
    ap.add_argument("--show", action="store_true", help="apre la finestra del grafico")
    ap.add_argument("--no-figure", action="store_true", help="non salva il grafico")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, dx = np.linspace(XMIN, XMAX, args.nx, endpoint=False, retstep=True)
    x = x + 0.5 * dx
    u_ini = u0(x)
    u_exact = exact_solution(u0, x, args.t, args.nx_exact, EXT_FACTOR)
    u_weno = solve_weno(u_ini, dx, args.t, args.cfl)
    err = np.sum(np.abs(u_weno - u_exact)) * dx / (XMAX - XMIN)

    print(f"Nx={args.nx}  CFL={args.cfl}  T={args.t}")
    print(f"errore L1 relativo (WENO5 vs esatta) = {err:.6e}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, u_ini, 'k--', linewidth=1.5, label='t=0')
    plt.plot(x, u_exact, 'b-', linewidth=2.5, label='Esatta (Hopf-Lax-Oleinik)')
    plt.plot(x, u_weno, 'r-', linewidth=2, label='WENO5')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Burgers, t={args.t:.3f}, Nx={args.nx}, CFL={args.cfl}')
    plt.legend()
    plt.grid(alpha=0.3)
    if not args.no_figure:
        out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.pardir, "figure"))
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "burgers_weno.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"grafico salvato in {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
