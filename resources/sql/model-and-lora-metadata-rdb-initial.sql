-- Initial schema for model/lora metadata cache database (SQLite)
-- Source spec: docs/spec/model-and-lora-metadata-rdb.md

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS content (
    content_id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE CHECK (LENGTH(sha256) = 64),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_identity (
    file_id INTEGER PRIMARY KEY,
    identity_kind TEXT NOT NULL CHECK (identity_kind IN ('win_file_id', 'posix_inode')),
    identity_key TEXT NOT NULL,
    volume_hint TEXT,
    last_file_size INTEGER NOT NULL,
    last_mtime_ns INTEGER NOT NULL,
    last_ctime_ns INTEGER,
    content_id INTEGER,
    content_link_kind TEXT CHECK (content_link_kind IN ('identity', 'sha256') OR content_link_kind IS NULL),
    content_linked_at TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (identity_kind, identity_key),
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_file_identity_content
    ON file_identity (content_id);

CREATE TABLE IF NOT EXISTS local_asset_path (
    asset_path_id INTEGER PRIMARY KEY,
    folder_name TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    basename TEXT NOT NULL,
    stem TEXT NOT NULL,
    file_id INTEGER,
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (folder_name, relative_path),
    FOREIGN KEY (file_id) REFERENCES file_identity(file_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_asset_path_file
    ON local_asset_path (file_id);

CREATE INDEX IF NOT EXISTS idx_asset_path_stem
    ON local_asset_path (stem);

-- Supplemental hashes only. Canonical SHA256 is stored in content.sha256.
CREATE TABLE IF NOT EXISTS content_hash (
    content_hash_id INTEGER PRIMARY KEY,
    content_id INTEGER NOT NULL,
    hash_algo TEXT NOT NULL CHECK (
        hash_algo IN (
            'sha1',
            'md5',
            'crc32',
            'a1111_legacy',
            'autov1',
            'autov2',
            'autov3',
            'blake3'
        )
    ),
    hash_value TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('local_compute', 'civitai_api')),
    computed_at TEXT NOT NULL,
    verified_at TEXT,
    UNIQUE (content_id, hash_algo),
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_content_hash_lookup
    ON content_hash (hash_algo, hash_value);

CREATE TABLE IF NOT EXISTS civitai_model (
    model_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    nsfw INTEGER CHECK (nsfw IN (0, 1) OR nsfw IS NULL),
    poi INTEGER CHECK (poi IN (0, 1) OR poi IS NULL),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_civitai_model_type
    ON civitai_model (type);

CREATE TABLE IF NOT EXISTS civitai_model_version (
    model_version_id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL,
    name TEXT,
    air TEXT,
    base_model TEXT,
    base_model_type TEXT,
    status TEXT,
    published_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    early_access_ends_at TEXT,
    usage_control TEXT,
    upload_type TEXT,
    description_html TEXT,
    trained_words_json TEXT,
    version_download_url TEXT,
    raw_json TEXT,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES civitai_model(model_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_civitai_version_model
    ON civitai_model_version (model_id);

CREATE INDEX IF NOT EXISTS idx_civitai_version_air
    ON civitai_model_version (air);

CREATE TABLE IF NOT EXISTS civitai_file (
    civitai_file_id INTEGER PRIMARY KEY,
    model_version_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    size_kb REAL,
    is_primary INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    download_url TEXT,
    pickle_scan_result TEXT,
    pickle_scan_message TEXT,
    virus_scan_result TEXT,
    virus_scan_message TEXT,
    scanned_at TEXT,
    metadata_format TEXT,
    metadata_size TEXT,
    metadata_fp TEXT,
    raw_json TEXT,
    FOREIGN KEY (model_version_id) REFERENCES civitai_model_version(model_version_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_civitai_file_version
    ON civitai_file (model_version_id);

CREATE INDEX IF NOT EXISTS idx_civitai_file_primary
    ON civitai_file (model_version_id, is_primary);

CREATE TABLE IF NOT EXISTS civitai_file_hash (
    civitai_file_id INTEGER NOT NULL,
    hash_algo TEXT NOT NULL CHECK (hash_algo IN ('sha256', 'autov1', 'autov2', 'autov3', 'crc32', 'blake3')),
    hash_value TEXT NOT NULL,
    PRIMARY KEY (civitai_file_id, hash_algo),
    FOREIGN KEY (civitai_file_id) REFERENCES civitai_file(civitai_file_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_civitai_hash_lookup
    ON civitai_file_hash (hash_algo, hash_value);

CREATE TABLE IF NOT EXISTS civitai_lookup_state (
    sha256 TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('found', 'not_found', 'unknown_auth_or_blocked', 'temporary_error')),
    http_status INTEGER,
    checked_at TEXT NOT NULL,
    next_retry_at TEXT,
    fail_count INTEGER NOT NULL DEFAULT 0 CHECK (fail_count >= 0),
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_civitai_lookup_retry
    ON civitai_lookup_state (status, next_retry_at);

CREATE TABLE IF NOT EXISTS content_civitai_match (
    content_id INTEGER NOT NULL,
    civitai_file_id INTEGER NOT NULL,
    match_algo TEXT NOT NULL,
    match_hash_value TEXT NOT NULL,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    matched_at TEXT NOT NULL,
    last_confirmed_at TEXT NOT NULL,
    PRIMARY KEY (content_id, civitai_file_id),
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE,
    FOREIGN KEY (civitai_file_id) REFERENCES civitai_file(civitai_file_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_match_content_active
    ON content_civitai_match (content_id, is_active);

CREATE INDEX IF NOT EXISTS idx_match_civitai_file
    ON content_civitai_match (civitai_file_id);

CREATE TABLE IF NOT EXISTS lora_metadata (
    content_id INTEGER PRIMARY KEY,
    ss_output_name TEXT,
    ss_sd_model_name TEXT,
    ss_clip_skip TEXT,
    ss_resolution TEXT,
    ss_bucket_info_json TEXT,
    ss_tag_frequency_json TEXT,
    modelspec_trigger_phrase TEXT,
    modelspec_description TEXT,
    modelspec_tags TEXT,
    modelspec_usage_hint TEXT,
    raw_metadata_json TEXT NOT NULL,
    parsed_at TEXT NOT NULL,
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lora_base_model
    ON lora_metadata (ss_sd_model_name);

CREATE TABLE IF NOT EXISTS lora_tag_frequency (
    content_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    tag TEXT NOT NULL,
    frequency INTEGER NOT NULL CHECK (frequency >= 0),
    PRIMARY KEY (content_id, dataset_name, tag),
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lora_tag_total
    ON lora_tag_frequency (content_id, tag);

CREATE INDEX IF NOT EXISTS idx_lora_tag_freq
    ON lora_tag_frequency (content_id, frequency);

CREATE TABLE IF NOT EXISTS model_runtime_settings (
    content_id INTEGER PRIMARY KEY,
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);
