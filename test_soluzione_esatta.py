#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test della soluzione esatta di Burgers (unittest, come in SampleLab).

    python3 -m unittest -v test_soluzione_esatta

Cosa si verifica:

1. senza urti (t < T_b) la formula di Hopf-Lax-Oleinik e la vecchia regola
   "min{ u0(y) : y + u0(y)t = x }" coincidono (la soluzione e' a un sol valore);
2. dopo la formazione degli urti la formula di Hopf-Lax-Oleinik riproduce un
   riferimento indipendente ad alta risoluzione, mentre la vecchia regola NO
   (e' l'errore corretto in questo progetto: il minimo di u0 fra le
   caratteristiche che arrivano in (x,t) non e' la selezione entropica);
3. nel punto di urto viene scelto lo stato entropico corretto;
4. shock di Riemann: posizione data da Rankine-Hugoniot (s = (u_l+u_r)/2);
5. ventaglio di rarefazione: u = x/t dentro il ventaglio;
6. lo schema WENO5 + SSP-RK3 converge alla soluzione esatta (smoke test).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "codice"))
import burgers_weno as bw            # noqa: E402


# ----------------------------------------------------------------------------
# Riferimento indipendente: Lax-Friedrichs/Rusanov del primo ordine su griglia
# fine (schema conservativo, indipendente dal WENO del progetto).
# ----------------------------------------------------------------------------
def riferimento_rusanov(u0, xmin, xmax, t_final, nx=1000, cfl=0.4):
    x = np.linspace(xmin, xmax, nx)
    dx = x[1] - x[0]
    u = np.asarray(u0(x), dtype=float)
    t = 0.0
    while t < t_final:
        a = max(1e-12, float(np.abs(u).max()))
        dt = cfl * dx / a
        if t + dt > t_final:
            dt = t_final - t
        f = 0.5 * u ** 2
        fr = 0.5 * (f[1:] + f[:-1]) - 0.5 * a * (u[1:] - u[:-1])
        un = u.copy()
        un[1:-1] = u[1:-1] - dt / dx * (fr[1:] - fr[:-1])
        un[0], un[-1] = un[1], un[-2]
        u = un
        t += dt
    return x, u


def errore_medio(sol, rif, frazione_scartata=0.10):
    """Media degli scarti, scartando la frazione peggiore di punti (gli urti,
    dove il riferimento del primo ordine e' largamente diffusivo)."""
    d = np.abs(sol - rif)
    soglia = np.quantile(d, 1.0 - frazione_scartata)
    return float(d[d <= soglia].mean()), float(d.max())


class TestSoluzioneEsatta(unittest.TestCase):

    # IC "del progetto": u0(x) = -sin(2x), con T_b = 1/2
    u0_progetto = staticmethod(bw.u0)

    def test_senza_urti_le_due_regole_coincidono(self):
        """Per t < T_b = 0.5 la soluzione e' a un sol valore: le due formule
        devono dare lo stesso risultato (qui la correzione non cambia nulla)."""
        x = np.linspace(-np.pi, np.pi, 201)
        nuova = bw.exact_solution(self.u0_progetto, x, 0.4)
        vecchia = bw.old_min_rule(self.u0_progetto, x, 0.4)
        self.assertLess(np.abs(nuova - vecchia).max(), 1e-6)
        # in regime liscio la soluzione esatta e' u0(x0) con x0 = x - u0(x0)t
        x0 = x.copy()
        for _ in range(150):                       # iterazione di punto fisso
            x0 = x - self.u0_progetto(x0) * 0.4
        self.assertLess(np.abs(nuova - self.u0_progetto(x0)).max(), 1e-9)

    def test_contro_riferimento_ad_alta_risoluzione(self):
        """t = 1.0 > T_b: la formula di Hopf-Lax-Oleinik riproduce il
        riferimento indipendente, la vecchia regola no."""
        t = 1.0
        x = np.linspace(-np.pi, np.pi, 301)
        nuova = bw.exact_solution(self.u0_progetto, x, t)
        vecchia = bw.old_min_rule(self.u0_progetto, x, t)
        xr, ur = riferimento_rusanov(self.u0_progetto, -np.pi, np.pi, t, nx=800)
        rif = np.interp(x, xr, ur)

        media_nuova, _ = errore_medio(nuova, rif)
        media_vecchia, max_vecchia = errore_medio(vecchia, rif)
        self.assertLess(media_nuova, 0.06, f"errore medio Hopf-Lax troppo alto: {media_nuova}")
        self.assertGreater(max_vecchia, 0.5,
                           "il test non discrimina: la vecchia regola e' troppo vicina")
        self.assertLess(media_nuova, media_vecchia,
                        "Hopf-Lax-Oleinik deve essere piu' accurata della vecchia regola")

    def test_stato_entropico_accanto_all_urto(self):
        """u0 = sin(2*pi*x) su [-2,2], t = 0.5: l'urto e' in x = 0.5 (per
        simmetria) e appena a sinistra lo stato e' quello della caratteristica
        con y + u0(y)t = x, cioe' circa +0.72. La vecchia regola prendeva il
        minimo fra TUTTE le caratteristiche che arrivano (anche le piu'
        lontane) e restituiva circa -0.77, che non e' la soluzione fisica."""
        def u0(z):
            return np.sin(2.0 * np.pi * np.asarray(z, dtype=float))

        x = np.array([0.486])
        nuova = float(bw.exact_solution(u0, x, 0.5)[0])
        vecchia = float(bw.old_min_rule(u0, x, 0.5)[0])
        self.assertAlmostEqual(nuova, 0.72, delta=0.05)
        self.assertLess(vecchia, 0.0)              # il "ramo sbagliato"
        self.assertGreater(nuova - vecchia, 1.0)

    def test_urto_di_riemann_rankine_hugoniot(self):
        """Riemann u_l = 1 (x<0), u_r = 0 (x>0): s = (u_l+u_r)/2 = 0.5, quindi
        l'urto si trova in x = 0.5 t."""
        def u0(z):
            return np.where(np.asarray(z, dtype=float) < 0.0, 1.0, 0.0)

        for t in (0.2, 0.4, 0.8):
            xs = 0.5 * t
            v_sx = float(bw.exact_solution(u0, np.array([xs - 0.1]), t)[0])
            v_dx = float(bw.exact_solution(u0, np.array([xs + 0.1]), t)[0])
            self.assertAlmostEqual(v_sx, 1.0, delta=1e-6)
            self.assertAlmostEqual(v_dx, 0.0, delta=1e-6)

    def test_ventaglio_di_rarefazione(self):
        """Riemann u_l = -1 (x<0), u_r = +1 (x>0): ventaglio centrato, con
        u(x,t) = x/t per |x| <= t e valori piatti fuori dal ventaglio."""
        def u0(z):
            return np.where(np.asarray(z, dtype=float) < 0.0, -1.0, 1.0)

        t = 0.4
        x = np.array([-0.5 * t, -0.1 * t, 0.0, 0.25 * t, 0.5 * t, -2.0 * t, 2.0 * t])
        atteso = np.array([-0.5, -0.1, 0.0, 0.25, 0.5, -1.0, 1.0])
        v = bw.exact_solution(u0, x, t)
        self.assertLess(np.abs(v - atteso).max(), 1e-6)

    def test_weno_converge_alla_soluzione_esatta(self):
        """Smoke test dello schema: a T = 0.4 l'errore medio di WENO5 rispetto
        alla soluzione esatta e' piccolo e cala raffinando la griglia."""
        def errore(nx):
            x, dx = np.linspace(bw.XMIN, bw.XMAX, nx, endpoint=False, retstep=True)
            x = x + 0.5 * dx
            u_esatta = bw.exact_solution(bw.u0, x, 0.4)
            u_num = bw.solve_weno(bw.u0(x), dx, 0.4, bw.CFL)
            return float(np.abs(u_num - u_esatta).mean())

        e200, e400 = errore(200), errore(400)
        self.assertLess(e200, 0.05)
        self.assertLess(e400, e200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
