# Guía de ejecución y depuración — `pfsp`

Guía paso a paso para preparar el entorno, ejecutar de extremo a extremo el *baseline*
NEH y el *Iterated Greedy* (incluido el operador de destrucción guiada que propone el
trabajo), y depurar componentes individuales sobre una instancia.

> El código del paquete está en **inglés** (identificadores, comentarios, docstrings,
> logs), por convención técnica del proyecto. Esta guía va en español, pero **todos los
> comandos son literales** y deben copiarse tal cual.

Todos los comandos asumen que el directorio de trabajo es la **raíz del repositorio** y
que existe el entorno virtual `.venv/` en esa raíz. Las rutas se escriben relativas a la
raíz.

---

## 1. Preparación del entorno (setup)

El paquete `pfsp` vive en `Code/` y se instala en **modo editable**, de modo que los CLI
y los tests importan exactamente el mismo código sin tocar `sys.path`.

### 1.1. Crear y activar el entorno (solo la primera vez)

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)
```

Requiere **Python 3.12** o superior (declarado en `pyproject.toml`).

> Si prefieres no activar el entorno, puedes anteponer `.venv/bin/` a cada comando
> (p. ej. `.venv/bin/python`, `.venv/bin/pfsp-neh`). Esta guía usa la forma activada por
> brevedad.

### 1.2. Instalar dependencias

Dependencias de ejecución (solo NumPy, con versión fija):

```bash
pip install -r Code/requirements.txt
```

Dependencias de desarrollo (pruebas, formateador, linter y `matplotlib` para la capa de
análisis):

```bash
pip install -r Code/requirements-dev.txt
```

### 1.3. Instalar el paquete en modo editable

```bash
pip install -e Code/
```

Esto registra el paquete `pfsp` y los ejecutables de consola `pfsp-neh` y `pfsp-ig`.

### 1.4. Verificar la instalación

```bash
python -c "import pfsp; print('pfsp ok')"
pfsp-neh --help
pfsp-ig --help
```

---

## 2. Ejecutar el *baseline* NEH (`pfsp-neh`)

El *entry point* `pfsp-neh` es la ejecución completa del *baseline*: carga una instancia
por ruta (lector Taillard o VRF según el formato), ejecuta NEH, calcula el RPD frente a
la mejor referencia conocida, imprime el resultado y persiste un *result record* y un
*run manifest* enlazados.

```bash
pfsp-neh --instance Code/data/taillard/tai20_5_0.fsp
```

Salida típica por pantalla:

```text
instance: taillard/tai20_5_0
algorithm: NEH
permutation: 2 16 8 7 14 13 10 15 12 18 5 3 4 17 0 1 9 6 19 11
makespan: 1286
rpd: 0.6260
record: .../Code/results/raw/NEH.csv
manifest: .../Code/results/manifests/run-YYYYMMDD-HHMMSS-xxxx.json
```

### Opciones

| Opción | Descripción |
|---|---|
| `--instance PATH` | **Obligatoria.** Ruta a un fichero Taillard (`.fsp`) o VRF (`VFR*`). El formato se detecta automáticamente. |
| `--seed INT` | Semilla entera, opcional. NEH es determinista y **no** consume aleatoriedad; se registra solo como metadato de procedencia. |
| `--results-dir DIR` | Directorio donde se escriben el *record* crudo y el *manifest*. Por defecto, `Code/results/`. |

### Ejemplos

Instancia VRF, con semilla y directorio de resultados a medida:

```bash
pfsp-neh --instance Code/data/vrf/Small/VFR10_5_1_Gap.txt --seed 42 --results-dir Code/results
```

Procesar varias instancias (cada invocación produce su propio *record* + *manifest*):

```bash
for f in Code/data/taillard/tai20_5_*.fsp; do pfsp-neh --instance "$f"; done
```

### Artefactos que genera

- `Code/results/raw/NEH.csv` — una fila por ejecución (se **añade**, no se sobrescribe).
- `Code/results/manifests/run-*.json` — un *manifest* por ejecución, con el entorno
  capturado y las rutas de salida.

La tabla resumen se **deriva** de los crudos con el agregador (no se edita a mano):

```bash
python -c "from pfsp.results.aggregate import write_summary, default_raw_dir, default_summary_path; print(write_summary(default_raw_dir(), default_summary_path()))"
```

Esto reescribe `Code/results/summary/summary.csv` a partir de las filas crudas
presentes. El detalle de cada artefacto y su relación con la metodología de la memoria
está en `Code/README.md`.

---

## 3. Ejecutar el *Iterated Greedy* (`pfsp-ig`)

`pfsp-ig` ejecuta el *Iterated Greedy* sobre una instancia con una semilla dada: parte de
la solución inicial NEH (con búsqueda local opcional) e itera destrucción →
reconstrucción → (búsqueda local) → aceptación hasta agotar el criterio de parada.

### 3.1. Invocación sobre una instancia

```bash
pfsp-ig --instance Code/data/taillard/tai20_5_0.fsp --seed 42
```

Salida típica:

```text
instance: taillard/tai20_5_0
algorithm: IG
seed: 42
permutation: 2 16 8 ...
makespan: 1260
rpd: 0.0000
record: .../Code/results/raw/IG.csv
manifest: .../Code/results/manifests/run-YYYYMMDD-HHMMSS-xxxx.json
```

### 3.2. Parámetros

| Opción | Descripción |
|---|---|
| `--instance PATH` | **Obligatoria.** Ruta a un fichero Taillard `.fsp` o VRF `VFR*`. |
| `--seed INT` | **Obligatoria.** Semilla entera no negativa para el generador aleatorio. |
| `--d INT` | Tamaño de destrucción (trabajos extraídos por iteración). Por defecto: 4. |
| `--tp FLOAT` | Factor de temperatura `Tp ≥ 0` del criterio de aceptación. Por defecto: 0,4. |
| `--stop KIND:VALUE` | Criterio de parada. Tipos: `evaluations`, `iterations`, `time`. Por defecto: `evaluations:5000`. |
| `--no-local-search` | Desactiva la búsqueda local por inserción (etiqueta la ejecución como `IG-noLS`). |
| `--destruction {random,idle-greedy,idle-rcl}` | Operador de destrucción. Por defecto: `random`. |
| `--alpha FLOAT` | Fracción de tamaño de la lista restringida de candidatos (RCL) del operador `idle-rcl`, con `0 < α ≤ 1`. Ignorado por los otros dos operadores. Por defecto: 0,10. |
| `--results-dir DIR` | Directorio de *records* y *manifests*. Por defecto, `Code/results/`. |

### 3.3. Criterios de parada

- **`evaluations:N`** — para tras `N` evaluaciones del *makespan* (determinista,
  reproducible; es el que se usa en las campañas del trabajo).
- **`iterations:N`** — para tras `N` ciclos de destrucción-reconstrucción (determinista).
- **`time:S`** — para tras `S` segundos de reloj (no exactamente reproducible; útil para
  comparar con el presupuesto `n·m·t` ms habitual en la literatura).

### 3.4. Los tres operadores de destrucción

| Valor | Operador | Papel en el trabajo |
|---|---|---|
| `random` | Extracción de `d` trabajos uniformemente al azar. | *Iterated Greedy* clásico; es la línea de comparación. |
| `idle-greedy` | Extrae las `d` posiciones de mayor *idle time*, de forma puramente voraz (O1). | Variante de ablación: aísla el efecto de guiar la destrucción sin aleatoriedad. |
| `idle-rcl` | Puntúa las posiciones por *idle time* y sortea las `d` extracciones dentro de una lista restringida de candidatos de tamaño `α·n` (O2). | **Aportación del trabajo**: introduce diversificación controlada sobre la señal de *idle time*. |

Cada operador escribe en su **propio** fichero de resultados (`IG.csv`,
`IG-idle-greedy.csv`, `IG-idle-rcl.csv`), de modo que ninguna campaña sobrescribe la
evidencia de otra.

### 3.5. Ejemplos

Operador propuesto con el α calibrado en el trabajo:

```bash
pfsp-ig --instance Code/data/taillard/tai20_5_0.fsp --seed 42 \
        --destruction idle-rcl --alpha 0.30
```

Variante de ablación puramente voraz:

```bash
pfsp-ig --instance Code/data/taillard/tai20_5_0.fsp --seed 42 --destruction idle-greedy
```

Presupuesto mayor y tamaño de destrucción a medida, sobre una instancia VRF:

```bash
pfsp-ig --instance Code/data/vrf/Small/VFR10_5_1_Gap.txt --seed 123 \
        --d 6 --tp 0.5 --stop evaluations:10000
```

Sin búsqueda local (ablación) y parada por tiempo:

```bash
pfsp-ig --instance Code/data/taillard/tai20_5_0.fsp --seed 42 --no-local-search
pfsp-ig --instance Code/data/taillard/tai50_10_0.fsp --seed 7 --stop time:30
```

---

## 4. Resolver una planta propia

Los ejemplos anteriores usan instancias de *benchmark*, pero el objetivo del producto
mínimo viable es resolver un caso propio. Basta describir la planta en un fichero de
texto con extensión `.fsp`: **una línea por máquina**, en el orden en que los trabajos
las recorren, con un entero no negativo por trabajo que indique su tiempo de proceso en
esa máquina. Las dimensiones se deducen de la matriz, así que no hace falta cabecera.

Tres máquinas y cuatro trabajos, guardado como `mi_planta.fsp`:

```text
54 83 15 71
79  3 11 99
16 89 49 15
```

```bash
pfsp-neh --instance mi_planta.fsp
pfsp-ig  --instance mi_planta.fsp --seed 42 --destruction idle-rcl --alpha 0.30
```

Ambas órdenes imprimen la secuencia de trabajos y su *makespan*, que es el dato de
interés operativo: el orden en que conviene introducir los pedidos en la línea. La
columna `rpd` aparece como `n/a`, porque el RPD se calcula frente al mejor *makespan*
publicado de la instancia y ese valor de referencia solo existe para las instancias de
*benchmark*.

---

## 5. Depuración manual: ejecutar un componente aislado

El paquete `pfsp` es la única fuente de verdad y cada componente se puede invocar por
separado, sin pasar por el CLI completo.

### 5.1. *Snippet* rápido con `python -c`

```bash
python -c "
from pfsp.io.loader import load_instance
from pfsp.core.makespan import makespan
from pfsp.core.neh import neh

inst = load_instance('Code/data/taillard/tai20_5_0.fsp')
print('id:', inst.identifier, 'shape (m,n):', inst.processing_times.shape)
print('makespan identidad:', makespan(inst.processing_times, list(range(inst.n))))
perm, ms = neh(inst)
print('NEH perm:', perm)
print('NEH makespan:', ms)
"
```

### 5.2. Sesión interactiva (REPL)

```python
>>> from pfsp.io.loader import load_instance
>>> from pfsp.core.makespan import makespan
>>> from pfsp.core.neh import neh
>>>
>>> inst = load_instance("Code/data/taillard/tai20_5_0.fsp")
>>> inst.n, inst.m, inst.upper_bound, inst.lower_bound
(20, 5, 1278, 1232)
>>> inst.processing_times[:, :3]          # primeras 3 columnas (trabajos)
>>> makespan(inst.processing_times, list(range(inst.n)))   # orden natural
>>> perm, ms = neh(inst)
>>> ms >= inst.lower_bound                # invariante: makespan >= LB
True
```

### 5.3. Inspeccionar el operador de destrucción guiada

Las piezas del operador propuesto también se pueden observar por separado: la
puntuación de *idle time* por posición y la lista restringida de candidatos que se
deriva de ella.

```bash
python -c "
from pfsp.io.loader import load_instance
from pfsp.core.destruction import idle_scores, build_rcl
from pfsp.core.neh import neh

inst = load_instance('Code/data/taillard/tai20_5_0.fsp')
perm, _ = neh(inst)
scores = idle_scores(perm, inst.processing_times)
print('idle score por posicion:', scores)
print('RCL (alpha=0.30, d=4):', build_rcl(scores, alpha=0.30, d=4))
"
```

### 5.4. Depuradores

- **`pdb`** sobre el CLI: insertar `breakpoint()` en el punto de interés (p. ej. en
  `Code/pfsp/core/ig.py`) y ejecutar `pfsp-ig --instance ... --seed 42`.
- **Depurador del IDE:** crear una configuración apuntando al módulo `pfsp.cli_ig` con
  sus argumentos, usando el intérprete del `.venv`.

### 5.5. Sin instancias a mano

Los datos de *benchmark* viajan en el repositorio, pero también se puede construir una
`Instance` sintética desde una matriz NumPy `(m, n)`:

```bash
python -c "
import numpy as np
from pfsp.instance import Instance
from pfsp.core.neh import neh

p = np.array([[3,1,4,1,5],[2,7,1,8,2],[8,1,8,2,4]], dtype=np.int64)  # m=3, n=5
inst = Instance(processing_times=p, n=5, m=3, source_benchmark='synthetic', source_path='synthetic/demo')
print(neh(inst))
"
```

---

## 6. Campañas multi-semilla (reanudables)

Las campañas experimentales del trabajo —un algoritmo sobre todas las instancias de un
*benchmark*, con varias semillas— se ejecutaron con el motor reanudable del paquete,
`pfsp.experiments.runner`. El motor persiste cada resultado **en cuanto la pareja
instancia × semilla termina**, así que una campaña se puede interrumpir (Ctrl-C, cierre
del equipo) y retomar después sin repetir trabajo ni duplicar filas.

Sus piezas públicas son `run_campaign`, `discover_instances` y las estrategias de
comprobación `lower_bound_sanity` (Taillard: `makespan ≥ LB`) y `best_known_sanity`
(VRF: `RPD ≥ 0` frente al mejor conocido). El motor acepta el conjunto de semillas, la
configuración del algoritmo, un `limit` para ejecutar por tramos y un modo `fresh` que
elimina solo las filas de **esa** campaña.

```bash
python -c "
import inspect
from pfsp.experiments import runner
print(inspect.signature(runner.run_campaign))
print(inspect.getdoc(runner.run_campaign))
"
```

Los envoltorios de línea de órdenes que se usaron para lanzar las campañas publicadas no
forman parte de esta distribución (ver la nota final de `Code/README.md`); la evidencia
cruda que produjeron sí, en `Code/results/` y `Code/results-server/`.

---

## 7. Verificación del proyecto

Pruebas, *linter* y formateador (las herramientas se configuran en `pyproject.toml`):

```bash
pytest Code/tests                 # suite de pruebas (unitarias + basadas en propiedades)
ruff check Code                   # linter (sin warnings nuevos)
black --check Code                # formateo
```
