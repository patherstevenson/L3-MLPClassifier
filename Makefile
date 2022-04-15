PROJECT=MLPClassifier
AUTHOR=Pather Stevenson
PY3=python3
PYTHONPATH=./src
export PYTHONPATH
SPHINXBUILD=sphinx-build
CONFIGPATH=.
SOURCEDOC=sourcedoc
DOC=doc
SCRIPT=install_lib.sh
MAIN=src/main.py

all: main

main:
	mkdir -p data/res/found && mkdir -p data/res/fail
	$(PY3) $(MAIN)

lib:
	chmod +x $(SCRIPT)
	./$(SCRIPT)

gnuplot:
	gnuplot mlp_accuracy.plt

clean:
	rm -f *~ */*~
	rm -rf __pycache__ src/__pycache__
	rm -rf $(DOC)
	rm -f $(PROJECT).zip

doc: author
	$(SPHINXBUILD) -c $(CONFIGPATH) -b html $(SOURCEDOC) $(DOC)

archive: clean
	zip -r $(PROJECT).zip .

author:
	sed -i -e 's/^project =.*/project = "$(PROJECT)"/g' conf.py
	sed -i -e 's/^copyright =.*/copyright = "2022, $(AUTHOR), Univ. Lille"/g' conf.py
	sed -i -e 's/^author =.*/author = "$(AUTHOR)"/g' conf.py

.PHONY: clean doc archive author main install_lib gnuplot
