#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <n> <neural_network>"
    exit 1
fi

# Parse input arguments
n=$1
neural_network=$2

# Ensure n is a positive integer
if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -le 0 ]; then
    echo "Error: <n> must be a positive integer."
    exit 1
fi

# Loop from 0 to n-1
for ((i=0; i<n; i++))
do
    echo "Running: python3 main.py -n $neural_network -l $i"
    python3 main.py -n "$neural_network" -l "$i"
    python3 pt_to_csv.py -n "$neural_network" -l "$i"
    if [ $? -ne 0 ]; then
        echo "Error: Command failed at iteration $i"
        exit 1
    fi
done

echo "All $n iterations completed successfully."
