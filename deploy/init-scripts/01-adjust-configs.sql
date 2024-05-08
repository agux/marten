-- Increase max_locks_per_transaction
ALTER SYSTEM SET max_locks_per_transaction = '2048';
SELECT pg_reload_conf();