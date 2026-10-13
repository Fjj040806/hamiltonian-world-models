#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pandoc manuscript.md math_appendix.md --standalone --number-sections --pdf-engine=pdflatex -o Hamiltonian_Control_Study.pdf
pandoc manuscript.md math_appendix.md --standalone --number-sections -t latex -o manuscript.tex
