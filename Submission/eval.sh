#!/bin/bash

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

# Run the Python script with the input file as an argument
python3 script.py "$input_file"
