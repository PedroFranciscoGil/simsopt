# GPU-native SIMSOPT documentation

The living implementation report is maintained in
`making_simsopt_gpu_native.tex`. Its length is determined by the material: new
benchmark results, figures, and design decisions should be added rather than
compressed to meet a fixed page count.

Build the PDF from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error making_simsopt_gpu_native.tex
```

Remove generated files with:

```sh
latexmk -C making_simsopt_gpu_native.tex
```
