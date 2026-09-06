# Instancias VRF (Vallada, Ruiz & Framinan, 2015)

El benchmark **VRF** de instancias *duras* para el PFSP con makespan: **480
instancias** en dos conjuntos —`Small/` (240) y `Large/` (240)— con nombre
`VFR{n}_{m}_{k}_Gap.txt` (ojo: prefijo **`VFR`**, no `VRF`; `{k}` = réplica 1..10).
Tamaños hasta 800 trabajos × 60 máquinas. Incluye `BestSolutionsAndBounds.xlsx`
(mejores soluciones conocidas y cotas).

> ⚠️ **Los ficheros de datos NO se versionan en git** mientras PA-04 (versionado
> de datasets) no esté decidido — ver `.gitignore`. Solo se conservan este README
> y el `.gitkeep`. Si clonas el repo, regenera las instancias (ver abajo).

## Fuente

- **Oficial:** grupo SOA (`soa.iti.es/problem-instances`), fichero
  `InstancesAndBounds.7z` (contiene `VRF_Instances.7z` + `BestSolutionsAndBounds.xlsx`).
  El sitio está **caído/reorganizado** (2026-06-06): redirige a `iti.es`.
- **Descargado vía Wayback Machine** (Internet Archive), captura del 2024-05-19:
  `https://web.archive.org/web/20240519074859/http://soa.iti.es/problem-instances`
- Paper: Vallada, Ruiz & Framinan (2015), *New hard benchmark for flowshop
  scheduling problems minimising makespan*, EJOR 240(3), 666-677,
  `doi:10.1016/j.ejor.2014.07.033` (ver `KB/02_estado_del_arte/vallada_2015.md`).

## Cómo regenerar la descarga

Requiere `py7zr` (en el `.venv`: `.venv/bin/python -m pip install py7zr`):

```bash
# 1. Descargar el archivo oficial desde la Wayback Machine (binario crudo: sufijo if_)
curl -L -o InstancesAndBounds.7z \
  "https://web.archive.org/web/20240519074859if_/http://soa.iti.es:80/files/InstancesAndBounds.7z"

# 2. Extraer (doble 7z: el externo contiene VRF_Instances.7z + el xlsx)
.venv/bin/python -c "import py7zr; py7zr.SevenZipFile('InstancesAndBounds.7z').extractall('.')"
.venv/bin/python -c "import py7zr; py7zr.SevenZipFile('VRF_Instances.7z').extractall('.')"
# Resultan las carpetas Small/ y Large/ con los .txt
```

## Bounds: del xlsx a CSV (para el código)

El archivo oficial `BestSolutionsAndBounds.xlsx` (mejores soluciones y cotas) se
**convierte una sola vez** a `best_solutions_and_bounds.csv` para que el código lo
lea con la librería estándar (`csv`), **sin** añadir dependencias (pandas/openpyxl)
al proyecto. El `.xlsx` se conserva como fuente; el CSV es el artefacto que consume
el `Reference_Provider`.

- **Columnas del CSV:** `instance,set,n,m,UB,LB`.
- `instance` es el nombre base `VFR{n}_{m}_{k}` (sin sufijo `_Gap`), que coincide
  con el `Instance_Identifier` usado por el código.
- `set` ∈ {`Small`, `Large`} (240 + 240 = 480 filas).

> Ni el `.xlsx` ni el `.csv` se versionan en git (regla de `Code/data/**`,
> ADR-0006). Regenéralos junto con las instancias.

### Cómo regenerar el CSV

La conversión usa solo la stdlib (un `.xlsx` es un zip de XML). Tras descargar el
`.xlsx` (ver pasos de arriba), ejecutar un script puntual que: (1) lea
`xl/sharedStrings.xml` y las hojas `Large`/`Small` con `zipfile` + `xml.etree`,
(2) normalice a las columnas `instance,set,n,m,UB,LB` (derivando `n`/`m` del nombre
en la hoja `Small`), y (3) escriba `best_solutions_and_bounds.csv` con el módulo
`csv`. El script no es parte del runtime del paquete `pfsp`.

## Formato del fichero (verificado 2026-06-06)

Distinto del de Taillard. Ver detalle en `KB/05_datasets/formato-instancias.md`.
Resumen:

- **Cabecera:** una sola línea con dos enteros → `n  m` (trabajos, máquinas). **Sin**
  seed/UB/LB ni rótulos de texto.
- **Cuerpo:** `n` líneas, **una por trabajo** (¡filas = trabajos, no máquinas!).
  Cada línea tiene `2*m` enteros en pares `maquina tiempo`:
  `0 t0 1 t1 2 t2 ... (m-1) t(m-1)`. Los índices de máquina (0..m-1) van explícitos
  e intercalados; el parser debe quedarse con los tiempos (posiciones impares).
- Validado en muestra de 12 instancias (pequeñas y grandes): filas == `n`, cada
  fila con `2m` valores y machine-ids 0..m-1 correctos.
