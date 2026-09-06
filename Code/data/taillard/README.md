# Instancias Taillard (1993)

Las **120 instancias estándar** del benchmark de Taillard para el PFSP (12 grupos
*n*×*m*, con *n* ∈ {20, 50, 100, 200, 500} y *m* ∈ {5, 10, 20}, 10 réplicas por
grupo). Ficheros `tai{n}_{m}_{k}.fsp`.

> ⚠️ **Los ficheros de datos NO se versionan en git** mientras PA-04 (versionado
> de datasets) no esté decidido — ver `.gitignore`. Solo se conservan este README
> y el `.gitkeep`. Si clonas el repo, regenera las instancias con el comando de
> abajo.

## Fuente

- Espejo: **`chneau/go-taillard`** → `https://github.com/chneau/go-taillard`
  (carpeta `pfsp/instances/`). Formato canónico de Taillard (con rótulos de texto).
- Sitio histórico original (Éric Taillard, `mistic.heig-vd.ch`): caído a fecha de
  descarga (2026-06-06).

## Cómo regenerar la descarga

Desde esta carpeta (`Code/data/taillard/`):

```bash
curl -sS -L "https://api.github.com/repos/chneau/go-taillard/git/trees/master?recursive=1" \
  | grep -oE '"path": *"pfsp/instances/[^"]*\.fsp"' \
  | sed 's/"path": *"//;s/"$//' \
  | while read -r p; do
      curl -sS -L -f -o "$(basename "$p")" \
        "https://raw.githubusercontent.com/chneau/go-taillard/master/$p"
    done
```

## Formato

Ver `KB/05_datasets/formato-instancias.md` (verificado contra estos ficheros el
2026-06-06). Resumen: línea de rótulo + línea `n m seed UB LB` + línea de rótulo +
*m* líneas de *n* enteros (filas = máquinas, columnas = trabajos).
