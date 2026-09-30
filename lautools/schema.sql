-- lautools database schema, version 1.
-- The database is opened with PRAGMA foreign_keys = ON (set in db.py).

-- ---------------------------------------------------------------------
-- Beamtimes: facts from beamtime-metadata-<id>.json + user description
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS beamtime (
    id INTEGER PRIMARY KEY,
    beamtime_id TEXT NOT NULL UNIQUE,
    label TEXT,

    beamline TEXT,
    beamline_alias TEXT,
    beamline_setup TEXT,
    facility TEXT,

    proposal_id TEXT,
    proposal_type TEXT,

    event_start TEXT,
    event_end TEXT,
    generated TEXT,

    core_path TEXT,

    applicant_username TEXT,
    applicant_lastname TEXT,
    applicant_institute TEXT,
    applicant_email TEXT,
    applicant_user_id TEXT,

    contact TEXT,

    leader_username TEXT,
    leader_lastname TEXT,
    leader_institute TEXT,
    leader_email TEXT,
    leader_user_id TEXT,

    pi_username TEXT,
    pi_lastname TEXT,
    pi_institute TEXT,
    pi_email TEXT,
    pi_user_id TEXT,

    retention_period TEXT,
    title TEXT,
    description TEXT,
    unix_id TEXT,

    users_door_db TEXT,         -- JSON list
    users_special TEXT,         -- JSON list
    users_unknown TEXT,         -- JSON list

    metadata_json TEXT,         -- raw metadata file content

    created_at TEXT NOT NULL,
    updated_at TEXT
);

-- ---------------------------------------------------------------------
-- Storage state of a beamtime (1:1 with beamtime)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS beamtime_storage (
    beamtime_id INTEGER PRIMARY KEY
        REFERENCES beamtime(id) ON DELETE CASCADE,

    on_gpfs INTEGER,
    on_tape INTEGER,
    last_on_gpfs TEXT,

    raw_exists INTEGER,
    raw_subdir_count INTEGER,
    raw_subdir_samples TEXT,    -- JSON list
    raw_size_bytes INTEGER,
    raw_size_bytes_timestamp TEXT,

    processed_exists INTEGER,
    processed_size_bytes INTEGER,
    processed_size_bytes_timestamp TEXT,

    scratch_cc_exists INTEGER,
    scratch_cc_writable INTEGER,
    scratch_cc_size_bytes INTEGER,
    scratch_cc_size_bytes_timestamp TEXT,

    shared_exists INTEGER,

    last_inspected TEXT
);

-- ---------------------------------------------------------------------
-- Laupy projects: directory (usually in scratch_cc) containing workspaces
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS laupy_project (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL,

    project_size_bytes INTEGER,
    project_size_bytes_timestamp TEXT,
    last_inspected TEXT
);

-- ---------------------------------------------------------------------
-- Project workspaces: individual working directories inside a project
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS laupy_project_workspace (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL
        REFERENCES laupy_project(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL,

    workspace_size_bytes INTEGER,
    workspace_size_bytes_timestamp TEXT,
    last_inspected TEXT,

    UNIQUE (project_id, name),
    UNIQUE (id, project_id)     -- target of composite FK in open_history
);

CREATE INDEX IF NOT EXISTS laupy_project_workspace_project_id_idx
    ON laupy_project_workspace(project_id);

-- ---------------------------------------------------------------------
-- M:N link beamtime <-> laupy project
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS beamtime_project_link (
    beamtime_id INTEGER NOT NULL
        REFERENCES beamtime(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL
        REFERENCES laupy_project(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (beamtime_id, project_id)
);

CREATE INDEX IF NOT EXISTS beamtime_project_link_project_id_idx
    ON beamtime_project_link(project_id);

-- ---------------------------------------------------------------------
-- Lists shown in the program (recent / pinned)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lautools_app_listed_beamtime (
    beamtime_id INTEGER PRIMARY KEY
        REFERENCES beamtime(id) ON DELETE CASCADE,
    listed_at TEXT NOT NULL,
    last_access TEXT,
    pinned INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lautools_app_listed_project (
    project_id INTEGER PRIMARY KEY
        REFERENCES laupy_project(id) ON DELETE CASCADE,
    listed_at TEXT NOT NULL,
    last_access TEXT,
    pinned INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------
-- Program history: what was opened and when
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lautools_app_history (
    id INTEGER PRIMARY KEY,
    opened_at TEXT NOT NULL,
    project_id INTEGER
        REFERENCES laupy_project(id) ON DELETE CASCADE,
    workspace_id INTEGER,
    action TEXT,

    FOREIGN KEY (workspace_id, project_id)
        REFERENCES laupy_project_workspace(id, project_id) ON DELETE CASCADE,
    CHECK (workspace_id IS NULL OR project_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS lautools_app_history_opened_at_idx
    ON lautools_app_history(opened_at);

CREATE INDEX IF NOT EXISTS lautools_app_history_project_opened_at_idx
    ON lautools_app_history(project_id, opened_at);
