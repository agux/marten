#!/bin/bash

# Source the .env file from the parent directory
if [ -f ../.env ]; then
  source ../.env
else
  echo ".env file not found in the parent directory!"
  exit 1
fi


SOURCE="postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
# Get the number of CPU cores
# NUM_CORES=$(nproc)
# Calculate 90% of the number of CPU cores
# DEFAULT_NWORKERS=$(echo "$NUM_CORES * 0.9 / 1" | bc)

# get a .sql backup file name with timestamp as suffix
SCHEMA_FILE="schema_$(date +"%Y%m%d%H%M%S").sql"
DATA_FILE="data_$(date +"%Y%m%d%H%M%S").sql"

# backup schema
pg_dump -d "$SOURCE" \
    --quote-all-identifiers \
    --create \
    --schema-only \
    --file=$SCHEMA_FILE

# backup data
pg_dump -d "$SOURCE" \
    --quote-all-identifiers \
    --large-objects \
    --data-only \
    --file=$DATA_FILE