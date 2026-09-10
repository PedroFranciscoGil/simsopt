# Making SIMSOPT GPU native

The project plan is maintained in
`making_simsopt_gpu_native.tex`.

Build the PDF from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error making_simsopt_gpu_native.tex
```

Remove generated files with:

```sh
latexmk -C making_simsopt_gpu_native.tex
```
