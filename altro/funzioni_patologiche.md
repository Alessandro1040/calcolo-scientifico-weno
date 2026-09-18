# Funzioni patologiche per il simulatore di leggi di conservazione

## Come usarle
1. Apri `calcoloscientificopython.html`
2. Seleziona flusso **✏️ personalizzata**
3. Incolla il codice `f(u)` nel campo
4. Seleziona IC **✏️ personalizzata**
5. Incolla il codice `u(x,0)` nel campo
6. Regola eventualmente Nx, CFL, T, xmin, xmax
7. Premi **Ricalcola**

---

## 🧪 CASO 1: Flusso a crescita esplosiva + IC a gradino ripido
### f(u) — flusso personalizzato
```
u*u*u*u*u/5
```
### u(x,0) — IC personalizzata
```
x<0?2:-1.5
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.2–0.3
- **T**: 0.02–0.05 (molto piccolo! Il flusso u⁵/5 è estremamente ripido)
- **xmin**: -2, **xmax**: 2
- **Schema**: WENO3 (k=2) o MUSCL

### Cosa testa
- f(u) = u⁵/5 → f'(u) = u⁴ → velocità caratteristica che cresce come u⁴
- Con uₗ=2, uᵣ=-1.5, la velocità massima è 2⁴=16 (vs Burgers dove è 2)
- I pesi WENO degenerano perché βᵣ ∝ (Δf)² ∝ (Δu⁵)² → overflow numerico
- Il passo temporale adattivo deve scendere a dt ~ CFL·Δx/16, rendendo la simulazione lentissima
- **Risultato atteso**: WENO esplode in NaN quasi subito; MUSCL con limitatore MC/superbee dovrebbe resistere ma con molti passi

---

## 🧪 CASO 2: Flusso discontinuo nella derivata + IC a dente di sega
### f(u) — flusso personalizzato
```
Math.abs(u)
```
### u(x,0) — IC personalizzata
```
x-Math.floor(x)-0.5
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.3–0.4
- **T**: 0.1–0.3
- **xmin**: -2, **xmax**: 2
- **Schema**: WENO3 o MUSCL

### Cosa testa
- f(u) = |u| → f'(u) = sign(u) → **discontinua in u=0** (derivata non definita)
- La soluzione esatta (metodo delle caratteristiche) usa `numDeriv` che approssima f'(0) ≈ (|ε|-|-ε|)/(2ε) = 0, ma la vera derivata non esiste
- Le caratteristiche hanno velocità ±1 a seconda del segno di u, creando un pattern a "V" negli urti
- La condizione iniziale a dente di sega (onda triangolare periodica) genera shock multipli che si formano e interagiscono
- **Risultato atteso**: la soluzione esatta potrebbe dare risultati inaspettati vicino a u=0; lo schema numerico potrebbe oscillare

---

## 🧪 CASO 3: Flusso con punto di flesso + IC a gradino doppio
### f(u) — flusso personalizzato
```
u*u*u/3-u*u/2
```
### u(x,0) — IC personalizzata
```
x<0?1.5:(x<1?-0.5:0.8)
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.3–0.4
- **T**: 0.1–0.3
- **xmin**: -2, **xmax**: 3
- **Schema**: KT (Kurganov-Tadmor) o MUSCL

### Cosa testa
- f(u) = u³/3 − u²/2 → f'(u) = u² − u = u(u-1) → **due zeri** (in u=0 e u=1)
- f''(u) = 2u − 1 → punto di flesso in u=0.5 → flusso **non convesso**
- La soluzione esatta con entropia di Lax (minimo tra candidati) NON è corretta: servirebbe l'inviluppo convesso di Osher-Oleinik
- Doppio gradino Riemann: due discontinuità → il rilevatore di salti ne trova più di uno e disabilita il solutore di Riemann esatto, forzando la ricerca caratteristiche che fallisce
- **Risultato atteso**: la soluzione esatta mostra artefatti (strutture non fisiche); lo schema numerico potrebbe essere più accurato dell'esatta!

---

## 🧪 CASO 4: Flusso oscillante + IC a gradino (Riemann classico)
### f(u) — flusso personalizzato
```
Math.sin(5*u)
```
### u(x,0) — IC personalizzata
```
x<0?2:-2
```
### Setup consigliato
- **Nx**: 400–800
- **CFL**: 0.2–0.3
- **T**: 0.05–0.15
- **xmin**: -2, **xmax**: 2
- **Schema**: MUSCL o KT

### Cosa testa
- f(u) = sin(5u) → f'(u) = 5cos(5u) → oscilla rapidamente tra -5 e +5
- Con uₗ=2, uᵣ=-2, la velocità caratteristica salta da 5cos(10)≈-2.7 a 5cos(-10)≈-2.7 (stesso segno!), ma internamente oscilla selvaggiamente
- La soluzione esatta con ricerca caratteristiche trova MOLTI zeri di g(x₀) perché f'(u₀(x₀)) oscilla → candidati multipli → l'euristica del minimo può scegliere quello sbagliato
- **Risultato atteso**: caos totale nella soluzione esatta; lo schema numerico produce una struttura complessa ma stabile

---

## 🧪 CASO 5: Flusso a crescita super-esponenziale + IC liscia ripida
### f(u) — flusso personalizzato
```
Math.exp(u*u)
```
### u(x,0) — IC personalizzata
```
2*Math.exp(-x*x/0.1)
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.05–0.1 (molto basso!)
- **T**: 0.005–0.02 (piccolissimo!)
- **xmin**: -3, **xmax**: 3
- **Schema**: MUSCL-Hancock (l'unico che potrebbe sopravvivere)

### Cosa testa
- f(u) = exp(u²) → f'(u) = 2u·exp(u²) → **crescita super-esponenziale**
- Con u₀ massimo ≈ 2, f'(2) = 4·e⁴ ≈ 218 → velocità caratteristica pazzesca
- dt = CFL·Δx/218 → per Nx=200, Δx=0.03, dt ≈ 0.45·0.03/218 ≈ 6·10⁻⁵ → servono ~300 passi per T=0.02
- I βᵣ di WENO contengono (Δf)² dove Δf ≈ exp(u²_max) − exp(u²_min) → overflow numerico immediato
- **Risultato atteso**: WENO muore subito; MUSCL con CFL bassissimo potrebbe farcela ma con MOLTI passi; la soluzione esatta potrebbe avere problemi di convergenza della bisezione

---

## 🧪 CASO 6: Flusso identicamente nullo + IC discontinua (test di trasparenza)
### f(u) — flusso personalizzato
```
0
```
### u(x,0) — IC personalizzata
```
x<0?1:-1
```
### Setup consigliato
- **Nx**: 100–200
- **CFL**: 0.5
- **T**: 1.0
- **xmin**: -2, **xmax**: 2
- **Schema**: qualsiasi

### Cosa testa
- f(u) = 0 → f'(u) = 0 → velocità caratteristica zero → **niente si muove**
- La soluzione esatta deve restituire u(x,t) = u₀(x) per ogni t
- Lo schema numerico deve conservare esattamente il gradino (nessuna evoluzione)
- Sembra banale, ma è un test di **trasparenza**: se lo schema introduce evoluzione quando non dovrebbe, c'è un bug
- **Risultato atteso**: linea perfettamente sovrapposta a t=0 per ogni t. Se si muove, c'è un errore nel codice.

---

## 🧪 CASO 7: Flusso lineare a tratti + IC a gradino (test di non-linearità)
### f(u) — flusso personalizzato
```
u<0?u:2*u
```
### u(x,0) — IC personalizzata
```
x<0?1:-1
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.3–0.4
- **T**: 0.1–0.3
- **xmin**: -2, **xmax**: 2
- **Schema**: MUSCL o KT

### Cosa testa
- f(u) = u per u<0, f(u) = 2u per u≥0 → f'(u) = 1 per u<0, f'(u) = 2 per u≥0
- **Derivata discontinua in u=0** (ma f è continua)
- Con uₗ=1, uᵣ=-1: la parte sinistra viaggia a velocità 2, la destra a velocità 1 → si scontrano → shock
- La `numDeriv` in u=0 dà (f(ε)-f(-ε))/(2ε) = (2ε-(-ε))/(2ε) = 1.5, che non è né 1 né 2
- **Risultato atteso**: la soluzione esatta usa una velocità media di 1.5 in u=0, che non è corretta; lo schema numerico potrebbe oscillare localmente

---

## 🧪 CASO 8: Flusso con singolarità + IC che la attraversa (il più patologico)
### f(u) — flusso personalizzato
```
1/(1+u*u)
```
### u(x,0) — IC personalizzata
```
3*Math.sin(x)
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.2–0.3
- **T**: 0.1–0.5
- **xmin**: -π, **xmax**: π
- **Schema**: MUSCL-Hancock

### Cosa testa
- f(u) = 1/(1+u²) → f'(u) = -2u/(1+u²)² → velocità limitata in [-0.65, 0.65]
- f''(u) = (6u²-2)/(1+u²)³ → cambia segno → **non convesso**
- La funzione è **limitata** (tra 0 e 1) ma la derivata è **non monotona**
- Con u₀ = 3sin(x), le caratteristiche hanno velocità che dipende dal segno di u: dove u>0 la velocità è negativa (si muovono a sinistra), dove u<0 la velocità è positiva (si muovono a destra) → **convergenza da entrambi i lati** → shock multipli
- La soluzione esatta con entropia di Lax (minimo) è sicuramente sbagliata perché il flusso non è né convesso né concavo
- **Risultato atteso**: la soluzione esatta mostra strutture non fisiche; lo schema numerico produce una soluzione stabile ma diversa dall'esatta; il confronto "esatta vs numerica" mostra divergenze che NON sono errori dello schema ma limiti della teoria dell'entropia di Lax

---

## 🧪 CASO 9: Flusso a legge di potenza frazionaria + IC nulla tranne un picco
### f(u) — flusso personalizzato
```
Math.pow(Math.abs(u),1.5)/1.5
```
### u(x,0) — IC personalizzata
```
x>-0.1&&x<0.1?1:0
```
### Setup consigliato
- **Nx**: 400–800
- **CFL**: 0.2–0.3
- **T**: 0.05–0.2
- **xmin**: -1, **xmax**: 1
- **Schema**: MUSCL o KT

### Cosa testa
- f(u) = |u|¹·⁵/1.5 → f'(u) = |u|⁰·⁵·sign(u) → **derivata non lipschitziana in u=0** (radice quadrata)
- f'(0) = 0, ma f'(u) → ∞/0? No, f'(u) → 0 per u→0, ma la derivata seconda è infinita in 0
- IC a "rettangolo" (discontinua su entrambi i lati): due shock che si propagano
- La velocità dello shock dipende da (f(uₗ)-f(uᵣ))/(uₗ-uᵣ) = (|1|¹·⁵/1.5 - 0)/(1-0) = 1/1.5 ≈ 0.667
- **Risultato atteso**: la soluzione esatta con Riemann dovrebbe funzionare; lo schema numerico potrebbe avere difficoltà con la derivata non lipschitziana vicino a u=0

---

## 🧪 CASO 10: Flusso caotico (mappa logistica) + IC costante (test di instabilità non lineare)
### f(u) — flusso personalizzato
```
4*u*(1-u)
```
### u(x,0) — IC personalizzata
```
0.5+0.01*Math.sin(10*x)
```
### Setup consigliato
- **Nx**: 200–400
- **CFL**: 0.1–0.2
- **T**: 0.05–0.2
- **xmin**: -π, **xmax**: π
- **Schema**: MUSCL-Hancock

### Cosa testa
- f(u) = 4u(1-u) = 4u - 4u² → f'(u) = 4 - 8u
- f(u) è la **mappa logistica** (caos deterministico) → comportamento estremamente sensibile alle condizioni iniziali
- f'(u) varia tra -4 (u=1) e +4 (u=0) → range di velocità molto ampio
- La piccola perturbazione sinusoidale nella IC viene amplificata non linearmente
- **Risultato atteso**: la soluzione esatta potrebbe divergere dalla numerica molto rapidamente a causa della sensibilità alle condizioni iniziali (effetto farfalla); lo schema numerico potrebbe mostrare strutture complesse che sembrano "turbolenza numerica"
