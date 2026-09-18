#!/bin/sh
# Compila la relazione. Il documento usa minted (evidenziazione del codice
# Python), quindi serve -shell-escape e il pacchetto Python "pygments":
#     pip install pygments
set -e
cd "$(dirname "$0")"
pdflatex -shell-escape -interaction=nonstopmode relazione.tex
pdflatex -shell-escape -interaction=nonstopmode relazione.tex
echo "OK -> $(pwd)/relazione.pdf"
