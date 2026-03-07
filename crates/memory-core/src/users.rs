use chrono::{DateTime, Utc};
use rand::RngCore;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_rusqlite::Connection;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyWithRaw {
    pub api_key: ApiKey,
    pub raw_key: String,
}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub is_admin: bool,
}

pub fn hash_key(raw: &str) -> String {
    let digest = Sha256::digest(raw.as_bytes());
    hex::encode(digest)
}

pub fn generate_raw_key() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!("mem_{}", hex::encode(bytes))
}

pub struct UserStore {
    conn: Connection,
}

impl UserStore {
    pub fn new(conn: Connection) -> Self {
        Self { conn }
    }

    pub async fn create_user(&self, name: &str) -> Result<User> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let created_at_str = now.to_rfc3339();
        let name = name.to_string();
        let id_clone = id.clone();
        let name_clone = name.clone();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO users (id, name, created_at) VALUES (?1, ?2, ?3)",
                    rusqlite::params![id_clone, name_clone, created_at_str],
                )?;
                Ok(())
            })
            .await
            .map_err(Error::Database)?;

        Ok(User {
            id,
            name,
            created_at: now,
        })
    }

    pub async fn get_user(&self, id: &str) -> Result<User> {
        let id = id.to_string();
        let id_for_err = id.clone();

        self.conn
            .call(move |conn| {
                let row = conn
                    .query_row(
                        "SELECT id, name, created_at FROM users WHERE id = ?1",
                        rusqlite::params![id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                            ))
                        },
                    )
                    .optional()?;

                let (id, name, created_at_str) =
                    row.ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)?;

                let created_at: DateTime<Utc> =
                    created_at_str
                        .parse()
                        .map_err(|e: chrono::ParseError| {
                            rusqlite::Error::FromSqlConversionFailure(
                                2,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?;

                Ok(User {
                    id,
                    name,
                    created_at,
                })
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound(format!("user not found: {id_for_err}"))
                }
                other => Error::Database(other),
            })
    }

    pub async fn list_users(&self) -> Result<Vec<User>> {
        self.conn
            .call(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, name, created_at FROM users ORDER BY created_at DESC")?;
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?;

                let mut users = Vec::new();
                for row in rows {
                    let (id, name, created_at_str) = row?;
                    let created_at: DateTime<Utc> =
                        created_at_str
                            .parse()
                            .map_err(|e: chrono::ParseError| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    2,
                                    rusqlite::types::Type::Text,
                                    Box::new(e),
                                )
                            })?;
                    users.push(User {
                        id,
                        name,
                        created_at,
                    });
                }
                Ok(users)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn delete_user(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        let id_for_err = id.clone();

        let changed = self
            .conn
            .call(move |conn| {
                let changed = conn.execute(
                    "DELETE FROM users WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(changed)
            })
            .await
            .map_err(Error::Database)?;

        if changed == 0 {
            return Err(Error::NotFound(format!("user not found: {id_for_err}")));
        }

        Ok(())
    }

    pub async fn create_api_key(&self, user_id: &str, name: &str) -> Result<ApiKeyWithRaw> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let raw_key = generate_raw_key();
        let key_hash = hash_key(&raw_key);
        let created_at_str = now.to_rfc3339();
        let user_id = user_id.to_string();
        let name = name.to_string();
        let id_clone = id.clone();
        let user_id_clone = user_id.clone();
        let name_clone = name.clone();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO api_keys (id, key_hash, user_id, name, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![id_clone, key_hash, user_id_clone, name_clone, created_at_str],
                )?;
                Ok(())
            })
            .await
            .map_err(Error::Database)?;

        Ok(ApiKeyWithRaw {
            api_key: ApiKey {
                id,
                user_id,
                name,
                created_at: now,
                revoked_at: None,
            },
            raw_key,
        })
    }

    pub async fn list_api_keys(&self, user_id: &str) -> Result<Vec<ApiKey>> {
        let user_id = user_id.to_string();

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, user_id, name, created_at, revoked_at FROM api_keys WHERE user_id = ?1 ORDER BY created_at DESC",
                )?;
                let rows = stmt.query_map(rusqlite::params![user_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                })?;

                let mut keys = Vec::new();
                for row in rows {
                    let (id, user_id, name, created_at_str, revoked_at_str) = row?;
                    let created_at: DateTime<Utc> =
                        created_at_str.parse().map_err(|e: chrono::ParseError| {
                            rusqlite::Error::FromSqlConversionFailure(
                                3,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?;
                    let revoked_at: Option<DateTime<Utc>> = revoked_at_str
                        .map(|s| {
                            s.parse().map_err(|e: chrono::ParseError| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    4,
                                    rusqlite::types::Type::Text,
                                    Box::new(e),
                                )
                            })
                        })
                        .transpose()?;
                    keys.push(ApiKey {
                        id,
                        user_id,
                        name,
                        created_at,
                        revoked_at,
                    });
                }
                Ok(keys)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn revoke_api_key(&self, key_id: &str) -> Result<()> {
        let key_id = key_id.to_string();
        let key_id_for_err = key_id.clone();
        let now = Utc::now().to_rfc3339();

        let changed = self
            .conn
            .call(move |conn| {
                let changed = conn.execute(
                    "UPDATE api_keys SET revoked_at = ?1 WHERE id = ?2 AND revoked_at IS NULL",
                    rusqlite::params![now, key_id],
                )?;
                Ok(changed)
            })
            .await
            .map_err(Error::Database)?;

        if changed == 0 {
            return Err(Error::NotFound(format!(
                "api key not found or already revoked: {key_id_for_err}"
            )));
        }

        Ok(())
    }

    pub async fn authenticate(&self, raw_key: &str) -> Result<AuthContext> {
        let key_hash = hash_key(raw_key);

        self.conn
            .call(move |conn| {
                let row = conn
                    .query_row(
                        "SELECT ak.user_id, CASE WHEN a.user_id IS NOT NULL THEN 1 ELSE 0 END \
                         FROM api_keys ak \
                         LEFT JOIN admins a ON ak.user_id = a.user_id \
                         WHERE ak.key_hash = ?1 AND ak.revoked_at IS NULL",
                        rusqlite::params![key_hash],
                        |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, bool>(1)?))
                        },
                    )
                    .optional()?;

                let (user_id, is_admin) =
                    row.ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)?;

                Ok(AuthContext { user_id, is_admin })
            })
            .await
            .map_err(|e| match e {
                tokio_rusqlite::Error::Rusqlite(rusqlite::Error::QueryReturnedNoRows) => {
                    Error::NotFound("invalid api key".to_string())
                }
                other => Error::Database(other),
            })
    }

    pub async fn add_admin(&self, user_id: &str) -> Result<()> {
        let user_id = user_id.to_string();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT OR IGNORE INTO admins (user_id) VALUES (?1)",
                    rusqlite::params![user_id],
                )?;
                Ok(())
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn remove_admin(&self, user_id: &str) -> Result<()> {
        let user_id = user_id.to_string();
        let user_id_for_err = user_id.clone();

        let changed = self
            .conn
            .call(move |conn| {
                let changed = conn.execute(
                    "DELETE FROM admins WHERE user_id = ?1",
                    rusqlite::params![user_id],
                )?;
                Ok(changed)
            })
            .await
            .map_err(Error::Database)?;

        if changed == 0 {
            return Err(Error::NotFound(format!(
                "admin not found: {user_id_for_err}"
            )));
        }

        Ok(())
    }

    pub async fn list_admins(&self) -> Result<Vec<User>> {
        self.conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT u.id, u.name, u.created_at \
                     FROM users u \
                     JOIN admins a ON u.id = a.user_id \
                     ORDER BY u.created_at DESC",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?;

                let mut users = Vec::new();
                for row in rows {
                    let (id, name, created_at_str) = row?;
                    let created_at: DateTime<Utc> =
                        created_at_str.parse().map_err(|e: chrono::ParseError| {
                            rusqlite::Error::FromSqlConversionFailure(
                                2,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?;
                    users.push(User {
                        id,
                        name,
                        created_at,
                    });
                }
                Ok(users)
            })
            .await
            .map_err(Error::Database)
    }

    pub async fn bootstrap(&self, user_name: &str) -> Result<Option<ApiKeyWithRaw>> {
        let user_name = user_name.to_string();

        let result = self
            .conn
            .call(move |conn| {
                let tx = conn.transaction()?;

                let key_count: i64 =
                    tx.query_row("SELECT COUNT(*) FROM api_keys", [], |row| row.get(0))?;
                if key_count > 0 {
                    return Ok(None);
                }

                let now = chrono::Utc::now();
                let now_str = now.to_rfc3339();

                let user_id: Option<String> = tx
                    .query_row(
                        "SELECT u.id FROM users u JOIN admins a ON u.id = a.user_id WHERE u.name = ?1 LIMIT 1",
                        rusqlite::params![user_name],
                        |row| row.get(0),
                    )
                    .optional()?;

                let user_id = match user_id {
                    Some(id) => id,
                    None => {
                        let id = ulid::Ulid::new().to_string();
                        tx.execute(
                            "INSERT INTO users (id, name, created_at) VALUES (?1, ?2, ?3)",
                            rusqlite::params![id, user_name, now_str],
                        )?;
                        tx.execute(
                            "INSERT INTO admins (user_id) VALUES (?1)",
                            rusqlite::params![id],
                        )?;
                        id
                    }
                };

                let raw_key = generate_raw_key();
                let key_hash = hash_key(&raw_key);
                let key_id = ulid::Ulid::new().to_string();

                tx.execute(
                    "INSERT INTO api_keys (id, key_hash, user_id, name, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![key_id, key_hash, user_id, "bootstrap", now_str],
                )?;

                tx.commit()?;

                Ok(Some(ApiKeyWithRaw {
                    api_key: ApiKey {
                        id: key_id,
                        user_id,
                        name: "bootstrap".to_string(),
                        created_at: now,
                        revoked_at: None,
                    },
                    raw_key,
                }))
            })
            .await
            .map_err(Error::Database)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup() -> UserStore {
        let conn = crate::db::open_in_memory().await.unwrap();
        UserStore::new(conn)
    }

    #[tokio::test]
    async fn should_create_and_get_user() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        assert_eq!(user.name, "alice");
        assert!(!user.id.is_empty());

        let fetched = store.get_user(&user.id).await.unwrap();
        assert_eq!(fetched.id, user.id);
        assert_eq!(fetched.name, "alice");
    }

    #[tokio::test]
    async fn should_list_users() {
        let store = setup().await;

        store.create_user("alice").await.unwrap();
        store.create_user("bob").await.unwrap();

        let users = store.list_users().await.unwrap();
        assert_eq!(users.len(), 2);
        assert_eq!(users[0].name, "bob");
        assert_eq!(users[1].name, "alice");
    }

    #[tokio::test]
    async fn should_delete_user() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        store.delete_user(&user.id).await.unwrap();

        let result = store.get_user(&user.id).await;
        assert!(matches!(result, Err(Error::NotFound(_))));

        let result = store.delete_user("nonexistent").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_create_and_authenticate_api_key() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        let key = store.create_api_key(&user.id, "test key").await.unwrap();

        assert!(key.raw_key.starts_with("mem_"));
        assert_eq!(key.api_key.user_id, user.id);
        assert_eq!(key.api_key.name, "test key");

        let auth = store.authenticate(&key.raw_key).await.unwrap();
        assert_eq!(auth.user_id, user.id);
        assert!(!auth.is_admin);
    }

    #[tokio::test]
    async fn should_reject_invalid_key() {
        let store = setup().await;

        let result = store.authenticate("mem_invalid").await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_reject_revoked_key() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        let key = store.create_api_key(&user.id, "revokable").await.unwrap();

        store.revoke_api_key(&key.api_key.id).await.unwrap();

        let result = store.authenticate(&key.raw_key).await;
        assert!(matches!(result, Err(Error::NotFound(_))));

        let result = store.revoke_api_key(&key.api_key.id).await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn should_manage_admins() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        let key = store.create_api_key(&user.id, "key").await.unwrap();

        store.add_admin(&user.id).await.unwrap();
        let auth = store.authenticate(&key.raw_key).await.unwrap();
        assert!(auth.is_admin);

        let admins = store.list_admins().await.unwrap();
        assert_eq!(admins.len(), 1);
        assert_eq!(admins[0].id, user.id);

        store.remove_admin(&user.id).await.unwrap();
        let auth = store.authenticate(&key.raw_key).await.unwrap();
        assert!(!auth.is_admin);

        let admins = store.list_admins().await.unwrap();
        assert!(admins.is_empty());
    }

    #[tokio::test]
    async fn should_bootstrap_first_user() {
        let store = setup().await;

        let result = store.bootstrap("admin").await.unwrap();
        assert!(result.is_some());
        let key = result.unwrap();
        assert!(key.raw_key.starts_with("mem_"));

        let auth = store.authenticate(&key.raw_key).await.unwrap();
        assert!(auth.is_admin);

        let result = store.bootstrap("admin2").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn should_delete_user_cascades_keys() {
        let store = setup().await;

        let user = store.create_user("alice").await.unwrap();
        let key = store.create_api_key(&user.id, "key").await.unwrap();

        store.delete_user(&user.id).await.unwrap();

        let keys = store.list_api_keys(&user.id).await.unwrap();
        assert!(keys.is_empty());

        let result = store.authenticate(&key.raw_key).await;
        assert!(matches!(result, Err(Error::NotFound(_))));
    }
}
