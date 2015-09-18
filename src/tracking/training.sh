#!/bin/bash
indici=(062.oni 063.oni 064.oni 065.oni 066.oni 067.oni)
for i in 062.oni 063.oni 064.oni 065.oni 066.oni 067.oni; do
	python features.py --v $i
done

python knn.py
