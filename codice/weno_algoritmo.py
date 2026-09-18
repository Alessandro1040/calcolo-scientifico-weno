# ---------------------------------------------------------------------------
# SCHELETRO DELL'ALGORITMO WENO mostrato dall'applicazione web nel pannello
# "Algoritmo" (codice Python eseguito nel browser con Pyodide).
#
# NON e' un modulo Python autonomo: l'app lo racchiude in una funzione
#     def step_fn(u, dx, dt):  ...  return uNew
# e lo richiama a ogni passo temporale, dentro il ciclo adattivo che gestisce
# il dt (CFL). Sono quindi gia' definite, nell'ambiente in cui il codice viene
# eseguito: `u` (array numpy delle celle), `dx`, `dt`, `FLUX.f(v)`,
# `FLUX.fprime(v)` e `numpy` (come `np`). I marcatori `# __PARAM:k__` sono
# sostituiti dall'app con il valore scelto nel pannello.
#
# k = 1 -> upwind, k = 2 -> WENO3, k = 3 -> WENO5.
# ---------------------------------------------------------------------------

# --- IPERPARAMETRO: ordine k (1=upwind, 2=WENO3, 3=WENO5) ---
k = 2  # __PARAM:k__
# EPS: stabilizzazione per i pesi WENO. Valore preso dal controllo UI (wenoEps). Se 0 → uso un floor adattivo.
EPS = 1e-8
# ---------------------------------------------------------------
n = len(u)

def gh(v, i):
    # ghost cells: estensione costante al bordo (outflow/extrapolation)
    if i < 0:
        return v[0]
    if i >= n:
        return v[n-1]
    return v[i]

def weno3m(v, j):
    a, b, c = gh(v,j-1), gh(v,j), gh(v,j+1)
    p0 = -0.5*a + 1.5*b
    p1 = 0.5*b + 0.5*c
    b0 = (b-a)**2
    b1 = (c-b)**2
    w0 = (1/3)/((b0+EPS)**2)
    w1 = (2/3)/((b1+EPS)**2)
    return (w0*p0+w1*p1)/(w0+w1)

def weno3p(v, j):
    a, b, c = gh(v,j), gh(v,j+1), gh(v,j+2)
    p0 = 0.5*a + 0.5*b
    p1 = 1.5*b - 0.5*c
    b0 = (b-a)**2
    b1 = (c-b)**2
    w0 = (2/3)/((b0+EPS)**2)
    w1 = (1/3)/((b1+EPS)**2)
    return (w0*p0+w1*p1)/(w0+w1)

def weno5m(v, j):
    v0,v1,v2,v3,v4 = gh(v,j-2), gh(v,j-1), gh(v,j), gh(v,j+1), gh(v,j+2)
    p0 = (1/3)*v0 + (-7/6)*v1 + (11/6)*v2
    p1 = (-1/6)*v1 + (5/6)*v2 + (1/3)*v3
    p2 = (1/3)*v2 + (5/6)*v3 + (-1/6)*v4
    b0 = (13/12)*(v0-2*v1+v2)**2 + (1/4)*(v0-4*v1+3*v2)**2
    b1 = (13/12)*(v1-2*v2+v3)**2 + (1/4)*(v1-v3)**2
    b2 = (13/12)*(v2-2*v3+v4)**2 + (1/4)*(3*v2-4*v3+v4)**2
    w0 = 0.1/((b0+EPS)**2)
    w1 = 0.6/((b1+EPS)**2)
    w2 = 0.3/((b2+EPS)**2)
    return (w0*p0+w1*p1+w2*p2)/(w0+w1+w2)

def weno5p(v, j):
    v0,v1,v2,v3,v4 = gh(v,j-1), gh(v,j), gh(v,j+1), gh(v,j+2), gh(v,j+3)
    p0 = (1/3)*v4 + (-7/6)*v3 + (11/6)*v2
    p1 = (-1/6)*v3 + (5/6)*v2 + (1/3)*v1
    p2 = (1/3)*v2 + (5/6)*v1 + (-1/6)*v0
    b0 = (13/12)*(v4-2*v3+v2)**2 + (1/4)*(v4-4*v3+3*v2)**2
    b1 = (13/12)*(v3-2*v2+v1)**2 + (1/4)*(v3-v1)**2
    b2 = (13/12)*(v2-2*v1+v0)**2 + (1/4)*(3*v2-4*v1+v0)**2
    w0 = 0.1/((b0+EPS)**2)
    w1 = 0.6/((b1+EPS)**2)
    w2 = 0.3/((b2+EPS)**2)
    return (w0*p0+w1*p1+w2*p2)/(w0+w1+w2)

def compute_L(v):
    # alpha = max characteristic speed (safety floor added)
    alpha = 0.0
    for j in range(n):
        a = abs(FLUX.fprime(v[j]))
        if a > alpha: alpha = a
    if alpha < 1e-12:
        alpha = 1e-12
    f = [FLUX.f(v[j]) for j in range(n)]
    fp = [0.5*(f[j]+alpha*v[j]) for j in range(n)]
    fm = [0.5*(f[j]-alpha*v[j]) for j in range(n)]
    # Ricostruzione di f+ (stencil sinistro) e f- (stencil destro)
    if k == 1:
        rm = lambda vv, j: gh(vv, j)
        rp = lambda vv, j: gh(vv, j+1)
    elif k == 2:
        rm, rp = weno3m, weno3p
    else:
        rm, rp = weno5m, weno5p
    fp_hat = [rm(fp, j) for j in range(n)]
    fm_hat = [rp(fm, j) for j in range(n)]
    # Flusso totale ai bordi: usa valori ai bordi (no wrap) grazie a gh() che usa estensione costante
    F_hat = [0.0]*(n+1)
    F_hat[0] = fp_hat[0] + fm_hat[0]
    for j in range(n):
        F_hat[j+1] = fp_hat[j] + fm_hat[j]  # F_{j+1/2}
    L = [-(F_hat[j+1]-F_hat[j])/dx for j in range(n)]
    return L

# SSP-RK3 (3 stadi)
L0 = compute_L(u)
u1 = [u[j] + dt*L0[j] for j in range(n)]
L1 = compute_L(u1)
u2 = [0.75*u[j] + 0.25*(u1[j] + dt*L1[j]) for j in range(n)]
L2 = compute_L(u2)
u3 = [(1/3)*u[j] + (2/3)*(u2[j] + dt*L2[j]) for j in range(n)]
uNew = u3
return uNew
