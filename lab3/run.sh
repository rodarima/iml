#!/bin/bash

DATADIR=data
DATASETS="autos hepatitis"
RESULTS=results.txt

rm -f $RESULTS

for d in $DATASETS; do
	DATAPATH="$DATADIR/$d"
	echo "Dataset $d"
	# Without weight and selection
	python lazy.py $DATAPATH >> results.txt
	# With weight and wihout selection
	python lazy.py -w $DATAPATH >> results.txt
	# With weight and selection
	python lazy.py -w -s $DATAPATH >> results.txt
done
