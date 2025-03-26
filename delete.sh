#!/bin/bash
# delete_dirs.sh
# WARNING: This script will permanently remove directories listed in unmatched_dirs.csv.
# Double-check the CSV before running.

CSV_FILE="unmatched_dirs.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "CSV file $CSV_FILE not found!"
    exit 1
fi

# Skip the header (first line) and process each subsequent line.
tail -n +2 "$CSV_FILE" | while IFS=, read -r dir mod_time; do
    # Remove any surrounding quotes (if present)
    dir=$(echo "$dir" | tr -d '"')
    if [ -d "$dir" ]; then
        echo "Deleting directory: $dir"
        rm -rf "$dir"
    else
        echo "Directory $dir does not exist"
    fi
done