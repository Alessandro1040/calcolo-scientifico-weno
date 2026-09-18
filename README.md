# Calcolo Scientifico — Burgers 1D con WENO

Materiale del progetto di **Calcolo Scientifico (Progetto 14)**:
*High order FD discretization of Burgers' equation in 1D with WENO*.

Il repository raccoglie tre cose che parlano fra loro:

1. una **applicazione web interattiva** (`visualizzazione.html`, un solo file,
   Python nel browser con Pyodide e grafici Plotly) per risolvere
   `u_t + f(u)_x = 0` con schemi di tipo differenze finite e confrontarli con la
   **soluzione esatta** calcolata per caratteristiche;
2. il **codice Python autonomo** del progetto (`codice/burgers_weno.py`):
   WENO5 + SSP-RK3 con flux splitting di Lax-Friedrichs, più la soluzione
   esatta entropica;
3. la **relazione LaTeX** (`relazione/relazione.tex`, compilata in
   `relazione/relazione.pdf`): teoria e dimostrazioni (formazione degli urti,
   Rankine-Hugoniot, perché Taylor diretto fallisce, forma conservativa,
   costruzione del flusso WENO5, soluzione esatta) e il listato del codice.

## La formula della soluzione esatta (corretta in questa versione)

La soluzione entropica di Burgers si ottiene dalla formula di
**Lax–Oleinik / Hopf–Lax**. Per `f(u) = u²/2`:

```
u(x,t) = (x − y*)/t ,     y* = argmin_y [ G(y) + (x − y)² / (2t) ] ,      G(y) = ∫ u₀(s) ds
```

I punti stazionari di `J(y) = G(y) + (x−y)²/(2t)` sono le radici
dell'equazione delle caratteristiche `y + u₀(y) t = x`, quindi la formula si può
calcolare così:

1. si trovano tutte le radici di `y + u₀(y) t = x` (le caratteristiche che
   arrivano in `(x,t)`), con bisezione;
2. fra queste si sceglie quella che **minimizza `J`**;
3. `u(x,t) = (x − y*)/t` (equivalentemente `u₀(y*)` nel punto stazionario).

Nella prima stesura della relazione (e nella prima versione
dell'applicazione) al punto 2 si prendeva invece la radice con **`u₀` minimo**.
Le due scelte coincidono finché la soluzione è a un sol valore o c'è un solo
urto isolato, ma **divergono appena più urti interagiscono**: il minimo di `u₀`
può pescare una caratteristica partita molto lontano (dove `u₀` è basso) e
restituire un valore che non è la soluzione fisica. Un test che lo mostra, con
`u₀ = sin(2πx)` su `[−2,2]` e `t = 0.5`:

| punto `x` | Hopf–Lax–Oleinik (corretta) | vecchia regola (`min u₀`) | riferimento numerico indipendente |
|-----------|-----------------------------|---------------------------|-----------------------------------|
| `0.486`   | **+0.72**                   | −0.77                     | +0.72                             |

(urto in `x = 0.5`; il riferimento è uno schema conservativo del primo ordine su
2000 celle, vedi `test_soluzione_esatta.py`). La relazione contiene la
discussione completa (Sezioni 2.4, 6.1, 6.4, 6.5) e la nota di versione in
Sezione 1.

## Applicazione web

Apri `visualizzazione.html` in un browser moderno (Chrome, Firefox, Safari,
Edge). **Nessuna installazione**: Pyodide (Python in WebAssembly) e Plotly
vengono scaricati da CDN, quindi serve una connessione a internet.

Cosa si può fare:

- scegliere la **legge di conservazione** `f(u)`: Burgers `u²/2`, trasporto
  lineare `a·u`, cubica `u³/3`, sinusoidale `sin(u)` o una funzione
  personalizzata scritta in JavaScript (`u` è la variabile, `a` un parametro);
- scegliere la **condizione iniziale**: `sin(πx)`, salto di Riemann (urto o
  rarefazione, con `u_l`/`u_r`), gaussiana (`amp`, `σ`), gradino smussato o una
  funzione personalizzata `u(x,0)`;
- scegliere lo **schema numerico**: WENO di ordine `2k−1` (`k = 1, 2, 3`, cioè
  upwind, WENO3, WENO5, con `ε` di stabilizzazione), MUSCL-Hancock TVD,
  Kurganov-Tadmor, Lax-Friedrichs, Lax-Wendroff, MacCormack, il
  **Lax-Wendroff di Taylor diretto** (non conservativo: serve a vedere
  l'urto muoversi alla velocità sbagliata) oppure un algoritmo libero scritto
  in Python nel pannello **Algoritmo** (eseguito via Pyodide);
- impostare `Nx`, `CFL` e tempo finale `T`, e usare il cursore del tempo;
- confrontare la **soluzione esatta** (calcolata con la formula di
  Hopf–Lax–Oleinik; il codice Python è visibile e modificabile) con quella
  numerica: profilo 2D, superfici 3D, **piano (x,t) con le caratteristiche**,
  pannello di ispezione di una cella `j` con i pesi WENO `ω` contro quelli
  lineari `d`;
- fare un **test di convergenza** su una sequenza di `Nx` (media o norme
  `L1`, `L2`, `L∞` dell'errore rispetto alla soluzione esatta).

In `altro/funzioni_patologiche.md` ci sono dieci combinazioni di flusso e
condizione iniziale pensate per mettere in crisi gli schemi (flussi non
convessi, derivate discontinue, singolarità, IC a dente di sega...).
`altro/visualizzazione_weno5.html` è la versione precedente dell'app (solo
WENO5, con la vecchia regola di selezione entropica), tenuta come riferimento
storico.

## Codice Python

Script autonomo (numpy + matplotlib):

```bash
python3 codice/burgers_weno.py                        # errore L1 + grafico su figure/
python3 codice/burgers_weno.py --nx 400 --t 1.0       # griglia e tempo finale
python3 codice/burgers_weno.py --show                 # apre anche la finestra
```

Stampa l'errore medio `|u_WENO5 − u_esatta|` e salva
`figure/burgers_weno.png` (profilo iniziale, soluzione esatta, soluzione WENO5).
I parametri di default sono quelli del progetto: `u₀(x) = −sin(2x)` su
`[−π, π]`, `Nx = 200`, `CFL = 0.45`, `T = 0.4`.

Il notebook originale del progetto è in `codice/burgers_weno.ipynb`, mentre
`codice/weno_algoritmo.py` è lo scheletro dell'algoritmo WENO (ordine `k`
selezionabile) che l'applicazione web mostra e permette di modificare nel
pannello "Algoritmo".

> **Nota.** Il notebook è l'artefatto originale del progetto e contiene ancora
> la *vecchia* funzione `exact_burgers` (selezione con il minimo di `u₀`): a
> `T = 0.4`, che è il valore di default, gli urti non si sono ancora formati e
> il risultato coincide con quello corretto, quindi i grafici del notebook
> restano validi. L'implementazione corretta è in `codice/burgers_weno.py`
> (`exact_solution`); la vecchia regola è conservata lì dentro come
> `old_min_rule`, solo per il confronto fatto dai test.

### Test

```bash
python3 -m unittest -v test_soluzione_esatta
```

Sei test (`unittest`, nessuna dipendenza extra oltre a numpy) che coprono:

| test | cosa verifica |
|------|----------------|
| `test_senza_urti_le_due_regole_coincidono` | per `t < T_b` la soluzione è a un sol valore: Hopf–Lax e vecchia regola coincidono e valgono `u₀(x₀)` |
| `test_contro_riferimento_ad_alta_risoluzione` | a `t = 1.0` Hopf–Lax riproduce un riferimento indipendente (Rusanov su griglia fine), la vecchia regola no |
| `test_stato_entropico_accanto_all_urto` | appena a sinistra dell'urto di `sin(2πx)` la soluzione è `≈ +0.72` e non `−0.77` |
| `test_urto_di_riemann_rankine_hugoniot` | l'urto sta in `x = s·t` con `s = (u_l + u_r)/2` |
| `test_ventaglio_di_rarefazione` | nel ventaglio `u(x,t) = x/t`, fuori i valori piatti |
| `test_weno_converge_alla_soluzione_esatta` | WENO5 + SSP-RK3 converge all'esatta e l'errore cala rafffinando |

## Relazione LaTeX

```bash
cd relazione && sh compila.sh        # richiede pdflatex e pygments (minted)
```

Il sorgente è `relazione/relazione.tex` (34 pagine), il PDF compilato è
`relazione/relazione.pdf`. Contenuto:

| sezione | argomento |
|---------|-----------|
| 1 | introduzione, obiettivo del progetto, nota di versione sulla formula corretta |
| 2 | metodo delle caratteristiche, tempo di rottura `T_b`, soluzioni deboli, Rankine–Hugoniot, condizioni di entropia di Lax e Oleinik, **formula di Lax–Oleinik** |
| 3 | approccio di Taylor (lineare e non lineare) e perché fallisce sugli urti |
| 4 | forma conservativa, conservazione/non conservazione della massa discreta, cancellazione telescopica |
| 5 | metodo WENO: flux splitting di Lax–Friedrichs, sotto-stencil, pesi non lineari, proprietà, pseudocodice |
| 6 | **soluzione esatta**: formulazione, caso `t ≈ 0`, estensione del dominio, ricerca delle radici e selezione entropica, ventaglio di rarefazione |
| Appendice | listato completo di `codice/burgers_weno.py` |

Il documento usa il pacchetto `minted` per il listato: la compilazione richiede
`-shell-escape` (già incluso in `compila.sh`) e il pacchetto Python `pygments`
(`pip install pygments`).

## Contenuto del repository

```
.
├── README.md                       questo file
├── visualizzazione.html            applicazione web interattiva (Pyodide + Plotly)
├── test_soluzione_esatta.py        test della soluzione esatta (unittest)
├── codice/
│   ├── burgers_weno.py             script autonomo: WENO5+SSP-RK3 e soluzione esatta
│   ├── burgers_weno.ipynb          notebook originale del progetto
│   └── weno_algoritmo.py           scheletro dell'algoritmo WENO (pannello "Algoritmo")
├── relazione/
│   ├── relazione.tex               sorgente LaTeX (formula di Lax–Oleinik corretta)
│   ├── relazione.pdf               PDF compilato (34 pagine)
│   └── compila.sh                  compilazione (pdflatex -shell-escape, due passate)
├── figure/                         grafici prodotti dallo script
└── altro/
    ├── visualizzazione_weno5.html  versione precedente dell'app (WENO5 + vecchia regola)
    └── funzioni_patologiche.md     dieci flussi/IC patologici da provare nell'app
```

## Requisiti

- **Applicazione web**: solo un browser recente e connessione a internet
  (Pyodide e Plotly da CDN). Nessun server, nessuna installazione.
- **Codice Python**: Python 3 (testato con 3.14.6), `numpy` (testato con 2.4.6)
  e `matplotlib` (3.11.0) per il grafico.
- **Relazione**: una distribuzione TeX Live con `minted`, `tcolorbox` e
  `pygments` (testata con TeX Live 2026).

## Note

- Il progetto nasce come elaborato per il progetto 14 del corso: qui sono
  raccolti il codice, l'applicazione di visualizzazione e la relazione, con la
  correzione della formula della soluzione esatta descritta sopra.
- Il tempo di rottura della condizione iniziale del notebook,
  `u₀(x) = −sin(2x)`, è `T_b = 1/2`: a `T = 0.4` (default) gli urti **non** si
  sono ancora formati, quindi il confronto esatta/numerica cade ancora in un
  regime liscio. Per vedere l'effetto degli urti usare per esempio `--t 1.0`.
- Con flussi non convessi (cubica, sinusoidale, funzioni personalizzate) la
  formula di Hopf–Lax–Oleinik va applicata usando l'**inviluppo convesso** del
  flusso nella trasformata di Legendre–Fenchel `f*`: è quello che fa il codice
  della soluzione esatta dell'applicazione web (`EXACT_TEMPLATES`), mentre
  `codice/burgers_weno.py` implementa il caso convesso di Burgers.
