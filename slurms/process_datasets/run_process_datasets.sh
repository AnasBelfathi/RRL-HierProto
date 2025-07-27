#!/bin/bash

# Chemin vers le dossier contenant les datasets
DATASETS_DIR="ssc-datasets"
OUTPUT_DIR="processed-datasets"

python process_datasets.py "$DATASETS_DIR" "$OUTPUT_DIR"
