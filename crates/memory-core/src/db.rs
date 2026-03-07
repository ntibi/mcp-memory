use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;
use tokio_rusqlite::Connection;
use tracing::info;

use crate::error::{Error, Result};

pub fn init_sqlite_vec() {
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }
}

fn get_schema_version(conn: &rusqlite::Connection) -> std::result::Result<i64, rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )",
    )?;
    let version: Option<i64> = conn
        .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
            row.get(0)
        })
        .unwrap_or(None);
    Ok(version.unwrap_or(-1))
}

fn set_schema_version(
    conn: &rusqlite::Connection,
    version: i64,
) -> std::result::Result<(), rusqlite::Error> {
    conn.execute_batch("DELETE FROM schema_version")?;
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (?1)",
        rusqlite::params![version],
    )?;
    Ok(())
}

fn migrate_v0(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            tag         TEXT NOT NULL,
            PRIMARY KEY (memory_id, tag)
        );

        CREATE TABLE IF NOT EXISTS memory_votes (
            id          TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            vote        TEXT NOT NULL CHECK (vote IN ('helpful', 'harmful')),
            voted_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_access_log (
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            accessed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS curation_suggestions (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL CHECK (type IN ('merge', 'prune', 'rewrite')),
            memory_ids  TEXT NOT NULL,
            suggestion  TEXT NOT NULL,
            source      TEXT NOT NULL CHECK (source IN ('auto', 'llm')),
            status      TEXT NOT NULL CHECK (status IN ('pending', 'applied', 'dismissed')),
            created_at  TEXT NOT NULL
        );",
    )?;

    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
            memory_id   TEXT PRIMARY KEY,
            embedding   float[384] distance_metric=cosine
        )",
        [],
    )?;

    Ok(())
}

fn column_exists(
    conn: &rusqlite::Connection,
    table: &str,
    column: &str,
) -> std::result::Result<bool, rusqlite::Error> {
    assert!(
        table.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
        "invalid table name: {table}"
    );
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let names: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(names.iter().any(|n| n == column))
}

fn migrate_v1(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id          TEXT PRIMARY KEY,
            key_hash    TEXT NOT NULL UNIQUE,
            user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name        TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            revoked_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS admins (
            user_id     TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE
        );",
    )?;

    if !column_exists(conn, "memories", "user_id")? {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")?;
    }
    if !column_exists(conn, "curation_suggestions", "user_id")? {
        conn.execute_batch(
            "ALTER TABLE curation_suggestions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''",
        )?;
    }

    let has_empty_user: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM memories WHERE user_id = '')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if has_empty_user {
        let migration_user_id = std::env::var("MEMORY__MIGRATION_USER_ID")
            .unwrap_or_else(|_| ulid::Ulid::new().to_string());
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT OR IGNORE INTO users (id, name, created_at) VALUES (?1, 'migration', ?2)",
            rusqlite::params![migration_user_id, now],
        )?;

        conn.execute(
            "UPDATE memories SET user_id = ?1 WHERE user_id = ''",
            rusqlite::params![migration_user_id],
        )?;
        conn.execute(
            "UPDATE curation_suggestions SET user_id = ?1 WHERE user_id = ''",
            rusqlite::params![migration_user_id],
        )?;

        conn.execute(
            "INSERT OR IGNORE INTO admins (user_id) VALUES (?1)",
            rusqlite::params![migration_user_id],
        )?;

        info!("backfilled empty user_id rows with migration user {migration_user_id}");
    }

    Ok(())
}

type Migration = fn(&rusqlite::Connection) -> std::result::Result<(), rusqlite::Error>;

const MIGRATIONS: &[Migration] = &[migrate_v0, migrate_v1];

fn run_migrations(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    let current = get_schema_version(conn)?;

    for (i, migration) in MIGRATIONS.iter().enumerate() {
        let version = i as i64;
        if version <= current {
            continue;
        }
        migration(conn)?;
        set_schema_version(conn, version)?;
    }

    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;

    Ok(())
}

pub async fn open(path: &str) -> Result<Connection> {
    init_sqlite_vec();
    let path = path.to_owned();
    let conn = Connection::open(&path).await.map_err(Error::Database)?;
    conn.call(|conn| {
        run_migrations(conn)?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;
    info!("database opened at {path}");
    Ok(conn)
}

#[cfg(test)]
pub async fn open_in_memory() -> Result<Connection> {
    init_sqlite_vec();
    let conn = Connection::open_in_memory()
        .await
        .map_err(Error::Database)?;
    conn.call(|conn| {
        run_migrations(conn)?;
        Ok(())
    })
    .await
    .map_err(Error::Database)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn should_create_all_tables_when_migrated() {
        let conn = open_in_memory().await.unwrap();

        let tables: Vec<String> = conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'memory_embeddings_%' ORDER BY name",
                )?;
                let rows = stmt.query_map([], |row| row.get(0))?;
                let mut tables = Vec::new();
                for row in rows {
                    tables.push(row?);
                }
                Ok(tables)
            })
            .await
            .unwrap();

        let expected = [
            "admins",
            "api_keys",
            "curation_suggestions",
            "memories",
            "memory_access_log",
            "memory_embeddings",
            "memory_tags",
            "memory_votes",
            "schema_version",
            "users",
        ];

        assert_eq!(tables, expected);
    }

    #[tokio::test]
    async fn should_verify_sqlite_vec_loaded() {
        let conn = open_in_memory().await.unwrap();

        let version: String = conn
            .call(|conn| {
                let v: String = conn.query_row("SELECT vec_version()", [], |row| row.get(0))?;
                Ok(v)
            })
            .await
            .unwrap();

        assert!(!version.is_empty());
    }
}
