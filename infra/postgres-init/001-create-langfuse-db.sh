#!/bin/bash
# Runs once on first Postgres init (docker-entrypoint-initdb.d). Creates a
# second database for Langfuse so it doesn't share docuagent's tables.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE langfuse;
EOSQL
