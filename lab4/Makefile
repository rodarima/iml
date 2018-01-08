all: report.pdf

report.pdf: report.tex fig/ex1/* fig/ex2/*
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

clean:
	rm -f report.{pdf,aux,log}
