#!/bin/bash

# Name of the log file
LOG_FILE="backup_db_$(date +%Y%m%d%H%M%S).log"

backup() {
  # Source the .env file from the parent directory
  if [ -f ../.env ]; then
    source ../.env
  else
    echo ".env file not found in the parent directory!"
    exit 1
  fi

  SOURCE="postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

  export PGPASSWORD="$DB_PASSWORD"
  
  echo "executing VACUUM..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "VACUUM;"
  
  echo "executing REINDEX..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "REINDEX DATABASE $DB_NAME;"
  
  echo "executing ANALYZE..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "ANALYZE;"

  unset PGPASSWORD

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
}

# Main script logic
if [[ "$1" != "background" ]]; then
  # Relaunch the script in the background using nohup
  nohup "$0" background > "$LOG_FILE" 2>&1 &
  echo "Maintenance script is running in the background. Logs are in $LOG_FILE"
  exit 0
else
  # Call the function to perform maintenance tasks
  backup
fi