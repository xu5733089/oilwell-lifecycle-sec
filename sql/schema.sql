-- 统一数据模型（方案 §3.2）
-- DDL 刻意只用 TEXT / REAL / INTEGER，SQLite 与 PostgreSQL 通用；
-- 换 PostgreSQL 时可再按需把日期列改为 DATE、给 prod_daily 加分区。

CREATE TABLE IF NOT EXISTS well_master (
    well_id           TEXT PRIMARY KEY,
    well_code_anon    TEXT NOT NULL,
    block             TEXT,
    layer             TEXT,
    well_type         TEXT,            -- 直井 / 定向井 / 水平井
    spud_date         TEXT,
    completion_date   TEXT,
    first_prod_date   TEXT,
    x_off             REAL,            -- 偏移后坐标，不可反推真实井位
    y_off             REAL,
    tvd               REAL,
    md                REAL,
    lateral_length    REAL,
    stage_count       INTEGER,
    proppant_t        REAL,
    frac_fluid_m3     REAL,
    status            TEXT,            -- producing / shut_in / completed_not_producing / planned
    data_source       TEXT NOT NULL    -- SYNTHETIC / PUBLIC / REAL
);

CREATE TABLE IF NOT EXISTS prod_daily (
    well_id    TEXT NOT NULL,
    dt         TEXT NOT NULL,
    day_index  INTEGER NOT NULL,       -- 自投产日起的天数，便于对齐
    oil_t      REAL,
    water_m3   REAL,
    gas_m3     REAL,
    whp_mpa    REAL,
    bhp_mpa    REAL,
    choke_mm   REAL,
    hours_on   REAL,
    water_cut  REAL,
    gor        REAL,
    PRIMARY KEY (well_id, dt)
);
CREATE INDEX IF NOT EXISTS ix_prod_daily_well ON prod_daily (well_id, day_index);

CREATE TABLE IF NOT EXISTS geo_static (
    well_id        TEXT NOT NULL,
    layer          TEXT NOT NULL,
    toc_pct        REAL,
    porosity_pct   REAL,
    perm_md        REAL,
    so_pct         REAL,
    sw_pct         REAL,
    net_pay_m      REAL,
    sweet_spot_idx REAL,
    brittleness    REAL,
    pressure_coef  REAL,
    temp_c         REAL,
    PRIMARY KEY (well_id, layer)
);

CREATE TABLE IF NOT EXISTS well_event (
    well_id    TEXT NOT NULL,
    dt         TEXT NOT NULL,
    day_index  INTEGER,
    event_type TEXT NOT NULL,          -- frac / acid / pump_change / shut_in / convert_injection
    note       TEXT,
    PRIMARY KEY (well_id, dt, event_type)
);

CREATE TABLE IF NOT EXISTS lifecycle_label (
    well_id           TEXT NOT NULL,
    label_def_version TEXT NOT NULL,
    t_oil_break       REAL,            -- d
    t_peak            REAL,            -- d
    q_peak            REAL,            -- t/d
    p_peak            REAL,            -- MPa
    cum_360           REAL,            -- t
    eur               REAL,            -- t
    dca_model         TEXT,
    di                REAL,
    b                 REAL,
    d_min             REAL,
    label_quality     TEXT,            -- ok / suspect / rejected
    reject_reason     TEXT,
    PRIMARY KEY (well_id, label_def_version)
);

CREATE TABLE IF NOT EXISTS reserves_record (
    target_type   TEXT NOT NULL,       -- well / block
    target_code   TEXT NOT NULL,
    as_of_date    TEXT NOT NULL,
    ooip_t        REAL,
    rf_pct        REAL,
    proved_t      REAL,
    category      TEXT,                -- PDP / PDNP / PUD / NOT_PROVED
    price_deck_id TEXT,
    method        TEXT,
    reviewer      TEXT,
    PRIMARY KEY (target_type, target_code, as_of_date)
);

CREATE TABLE IF NOT EXISTS model_run (
    run_id            TEXT PRIMARY KEY,
    model_version     TEXT NOT NULL,
    label_def_version TEXT NOT NULL,
    data_source       TEXT,
    split_method      TEXT,
    n_train           INTEGER,
    n_calib           INTEGER,
    n_test            INTEGER,
    metrics_json      TEXT,
    created_at        TEXT
);

CREATE TABLE IF NOT EXISTS prediction (
    trace_id          TEXT PRIMARY KEY,
    well_id           TEXT,
    model_version     TEXT,
    label_def_version TEXT,
    data_source       TEXT,
    obs_days          INTEGER,
    input_hash        TEXT,
    output_json       TEXT,
    created_at        TEXT
);

-- 全链路留痕：任何一次工具调用、任何一次智能体应答都能顺 trace_id 回溯
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id   TEXT,
    actor      TEXT,                   -- user / agent / kernel
    action     TEXT,
    payload    TEXT,
    status     TEXT,                   -- ok / failed / denied
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_trace ON audit_log (trace_id);
