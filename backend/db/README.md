# Database Models & Migrations

PostgreSQL relational database schema management for metadata, user profiles, personalization vectors, and leaderboard evaluations.

## Data Schemas
- `users`: User profiles, accessibility preferences, UI language defaults.
- `enrolled_signs`: Sign embeddings for few-shot personal gesture recognition (vector indexed via `pgvector`).
- `benchmark_submissions`: Leaderboard model submissions, BLEU-4 metrics, signer-independent accuracy scores.
- `audit_logs`: Governance verification logs, consent withdrawal audit trails.
