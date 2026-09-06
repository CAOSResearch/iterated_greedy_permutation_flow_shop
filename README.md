# iterated_greedy_permutation_flow_shop

**Optimización de bajo coste computacional para la programación de tareas industriales en
PYMEs: un enfoque basado en Iterated Greedy para el Permutation Flow-Shop.**

Código y resultados experimentales del Trabajo de Fin de Grado del mismo título.

- **Autor:** Augusto De Pinho Gomes
- **Tutor:** David Yagüe Cuevas
- **Titulación:** Grado en Ingeniería Informática, Universidad Carlos III de Madrid
- **Curso:** 2025-2026

## El problema y el objetivo

Este trabajo aborda el problema clásico de secuenciación de tareas, el *Permutation
Flow-Shop Scheduling Problem* (PFSP), minimizando el *makespan* ($C_{max}$).
Tradicionalmente, resolverlo requiere herramientas comerciales o metaheurísticas de gran
coste computacional, lo que supone una barrera para las Pequeñas y Medianas Empresas
(PYMEs).

El objetivo es diseñar, implementar y validar un algoritmo heurístico de alto rendimiento
que funcione como solución accesible y de bajo coste computacional. Sobre el esquema
clásico *Iterated Greedy* se propone sustituir la destrucción aleatoria por un **operador
de destrucción guiada** que concentra la extracción en las secuencias responsables de los
mayores *idle times*. El rendimiento se evalúa estadísticamente frente al *baseline*
constructivo NEH sobre los conjuntos de instancias **Taillard (1993)** y **VRF (2015)**,
este último de alta complejidad, demostrando su viabilidad como motor de optimización
pragmático para el tejido industrial a pequeña escala.

El *stack* es íntegramente software libre (Python + NumPy), sin *solvers* comerciales: la
solución está pensada para ejecutarse en equipos convencionales de una PYME.

> **Sobre la sostenibilidad.** Al reducir el *makespan* y los tiempos muertos de la
> maquinaria, el trabajo apunta a un beneficio en la eficiencia energética de la planta.
> Conviene ser preciso: ese beneficio es un **argumento narrativo**, derivado de acortar
> los tiempos de proceso e inactividad. La **única métrica que el algoritmo modela y
> optimiza es el tiempo** ($C_{max}$); no se modela ni se mide consumo energético en
> ningún punto del trabajo.

## Contenido

```
.
├── Code/
│   ├── pfsp/               Paquete Python: el algoritmo y su infraestructura
│   ├── tests/              Suite de pruebas (unitarias y basadas en propiedades)
│   ├── data/               Instancias de benchmark (Taillard y VRF)
│   ├── results/            Resultados experimentales (equipo portátil)
│   ├── results-server/     Resultados experimentales (servidor del laboratorio)
│   ├── pyproject.toml      Configuración del paquete, black, ruff y pytest
│   ├── requirements.txt    Dependencia de ejecución, con versión fija
│   ├── requirements-dev.txt Dependencias de desarrollo (pruebas, análisis, figuras)
│   ├── README.md           Mapa de artefactos del paquete y su relación con la memoria
│   └── EXECUTION_GUIDE.md  Guía de ejecución y depuración, paso a paso
├── Memoria/                Memoria del TFG (PDF)
└── LICENSE                 BSD 3-Clause (código)
```

## Puesta en marcha

Requisitos: **Python 3.12** o superior y `git`. Nada más.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r Code/requirements.txt
pip install -e Code/
```

Resolver una instancia con el *baseline* y con el algoritmo propuesto:

```bash
pfsp-neh --instance Code/data/taillard/tai20_5_0.fsp
pfsp-ig  --instance Code/data/taillard/tai20_5_0.fsp --seed 42
pfsp-ig  --instance Code/data/taillard/tai20_5_0.fsp --seed 42 \
         --destruction idle-rcl --alpha 0.30
```

Las tres órdenes imprimen la secuencia de trabajos y su *makespan*. Para resolver una
planta propia basta describirla en un fichero `.fsp`: una línea por máquina, en el orden
en que los trabajos la recorren, con un entero no negativo por trabajo. El detalle está
en `Code/EXECUTION_GUIDE.md` y en el Anexo A de la memoria.

Verificar la instalación:

```bash
pip install -r Code/requirements-dev.txt
pytest Code/tests
ruff check Code
black --check Code
```

## Los resultados experimentales

`Code/results/` y `Code/results-server/` contienen la **evidencia cruda** de las campañas
descritas en el capítulo de resultados de la memoria, en los dos entornos de ejecución
empleados (portátil y servidor del laboratorio):

| Carpeta | Dónde | Contenido |
|---|---|---|
| `raw/` | ambos | Una fila por ejecución individual (instancia × semilla), no medias: `NEH.csv`, `IG.csv` (destrucción aleatoria) e `IG-idle-greedy.csv` / `IG-idle-rcl.csv` (las dos variantes del operador propuesto). |
| `calibration/` | ambos | Barridos de calibración: presupuesto de cómputo y parámetro α de la lista restringida de candidatos. |
| `instrumentation/` | solo servidor | Instrumentación del mecanismo (trabajo de la búsqueda local) con su informe de análisis. Se midió en el servidor para tener los tiempos en un único entorno. |
| `summary/` | solo portátil | Tabla resumen, **derivada** del crudo por el agregador del paquete; nunca escrita a mano. Regenerable en cualquier entorno. |
| `figures/` | solo portátil | Figuras en PDF generadas a partir de esos datos. |

Las dos carpetas **no son espejo**: cada una guarda lo que se produjo en su entorno. Los
derivados (`summary/`, `figures/`) se regeneran desde el crudo con el paquete, así que su
ausencia en un entorno no supone pérdida de información.

Las tablas y figuras de la memoria se derivan de estos ficheros. La métrica de comparación
es el **RPD** (*Relative Percentage Deviation*) frente al mejor *makespan* publicado de
cada instancia, que se resuelve desde la cota de Taillard o desde el fichero de mejores
soluciones de VRF, y **nunca se fabrica** si no existe.

## Licencia

El **código** se distribuye bajo licencia **BSD de tres cláusulas** (BSD-3-Clause), según
el fichero [`LICENSE`](LICENSE) de la raíz de este repositorio: permite usar, modificar y
redistribuir el software sin obligaciones de *copyleft*, conservando el aviso de copyright
y sin emplear el nombre del titular para promocionar productos derivados.

La **memoria** se publica bajo Creative Commons **Reconocimiento - No Comercial - Sin Obra
Derivada** (CC BY-NC-ND), según se indica en su portada.

Las instancias de *benchmark* de `Code/data/` son de dominio público y conservan la
atribución de sus autores originales, documentada en los `README.md` de cada subcarpeta.
Las dependencias de ejecución y desarrollo conservan las suyas (PSF para Python, BSD-3
para NumPy, MIT para pytest, ruff y black), todas compatibles entre sí.
