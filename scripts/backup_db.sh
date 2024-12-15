#!/bin/bash

# Name of the log file with timestamp
LOG_FILE="backup_db_$(date +%Y%m%d%H%M%S).log"

# Function to output log messages with timestamp and color
log_message() {
  echo -e "\e[32m$(date '+%Y-%m-%d %H:%M:%S %Z')\e[0m - $1"
}

backup() {
  # Source the .env file from the parent directory
  if [ -f ../.env ]; then
    source ../.env
  else
    echo ".env file not found in the parent directory!"
    exit 1
  fi

  # Database connection string
  SOURCE="postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

  # Export password for psql and pg_dump
  export PGPASSWORD="$DB_PASSWORD"
  
  log_message "Executing VACUUM..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "VACUUM;"
  
  log_message "Executing REINDEX..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "REINDEX DATABASE $DB_NAME;"
  
  log_message "Executing ANALYZE..."
  psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" \
    -c "ANALYZE;"

  # Unset the password variable
  unset PGPASSWORD

  # Get a timestamp for file names
  TIMESTAMP=$(date +"%Y%m%d%H%M%S")

  # Define backup file names with timestamp
  SCHEMA_FILE="schema_${TIMESTAMP}.sql"
  DATA_FILE="data_${TIMESTAMP}.sql"

  log_message "Backing up schema to ${SCHEMA_FILE}..."
  pg_dump -d "$SOURCE" \
      --quote-all-identifiers \
      --create \
      --schema-only \
      --file="$SCHEMA_FILE"

  log_message "Backing up data to ${DATA_FILE}..."
  pg_dump -d "$SOURCE" \
      --quote-all-identifiers \
      --large-objects \
      --data-only \
      --file="$DATA_FILE"
  
  # Define the tar.gz file name
  TAR_FILE="backup_${TIMESTAMP}.tar.gz"
  
  # Tar and gzip the backup files
  log_message "Archiving backup files into ${TAR_FILE}..."
  tar -czf "$TAR_FILE" "$SCHEMA_FILE" "$DATA_FILE"
  # tar with bzip2: for better compression
  # tar -cjf "$TAR_FILE" "$SCHEMA_FILE" "$DATA_FILE"
  # tar with xz: highest compression ratio but slowest
  # tar -cJf "$TAR_FILE" "$SCHEMA_FILE" "$DATA_FILE"

  if [ $? -eq 0 ]; then
    log_message "Backup files archived successfully."
    # Optionally, remove the .sql files after archiving
    rm "$SCHEMA_FILE" "$DATA_FILE"
    log_message "Temporary .sql files removed."
  else
    log_message "Error archiving backup files."
    # Handle the error as needed
    # You may choose to keep the .sql files for troubleshooting
  fi

  log_message "Database maintenance and backup complete."
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
