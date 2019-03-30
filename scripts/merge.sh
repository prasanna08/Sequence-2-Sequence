#!bin/bash

echo "Merging EN-HI model files into single model."
cat ./models/ENHI-0* > ./models/ENHI.pt

echo "Merging EN-DE model files into single model."
cat ./models/ENDE-0* > ./models/ENDE.pt

echo "Merging EN-HI vocab file into single file."
cat ./data/EN-HI-2-30-0* > ./data/EN-HI-2-30.json
