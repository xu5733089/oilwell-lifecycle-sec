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
    event_type TEXT NOT NULL,          -- perforation / frac / acid / workover / sand_control / waterflood
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

-- SEC 评估单元层级：油田公司 -> 采油厂 -> SEC 单元
CREATE TABLE IF NOT EXISTS sec_unit (
    unit_id       TEXT PRIMARY KEY,
    unit_name     TEXT NOT NULL,
    plant_id      TEXT NOT NULL,
    plant_name    TEXT NOT NULL,
    company_id    TEXT NOT NULL,
    company_name  TEXT NOT NULL,
    area_type     TEXT,                -- 老区 / 新区
    opex_factor   REAL DEFAULT 1.0,    -- 单元操作成本系数
    data_source   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unit_well (
    unit_id  TEXT NOT NULL,
    well_id  TEXT NOT NULL,
    PRIMARY KEY (unit_id, well_id)
);

-- 产量与工作量计划（规划部门下达），用于计划与实际对标
CREATE TABLE IF NOT EXISTS unit_plan_monthly (
    unit_id             TEXT NOT NULL,
    ym                  TEXT NOT NULL,     -- YYYY-MM
    plan_new_wells      INTEGER,
    plan_new_oil_t      REAL,
    plan_measure_wells  INTEGER,
    plan_measure_inc_t  REAL,
    plan_old_oil_t      REAL,
    data_source         TEXT NOT NULL,
    PRIMARY KEY (unit_id, ym)
);

-- 部署井位（开发方案中的井位）：PUD 入账、五年规则与转化跟踪
CREATE TABLE IF NOT EXISTS unit_location (
    location_id        TEXT PRIMARY KEY,
    unit_id            TEXT NOT NULL,
    x_off              REAL NOT NULL,
    y_off              REAL NOT NULL,
    planned_drill_ym   TEXT NOT NULL,     -- 计划钻井年月 YYYY-MM
    first_booked_as_of TEXT,              -- 首次作为 PUD 入账的评估基准日；空 = 尚未入账
    drilled_well_id    TEXT,              -- 已钻：对应 well_master.well_id
    status             TEXT NOT NULL,     -- planned / drilled / cancelled
    capex_wan          REAL,              -- 钻完井投资（万元），空则取配置默认值
    data_source        TEXT NOT NULL
);

-- 单元资产账面价值（财务台账）：产量法折耗与减值测试
CREATE TABLE IF NOT EXISTS unit_asset_book (
    unit_id              TEXT NOT NULL,
    as_of                TEXT NOT NULL,   -- 期末评估基准日
    opening_nbv_wan      REAL,            -- 期初资产净值（万元）；空 = 由上期期末滚动
    capex_additions_wan  REAL NOT NULL,   -- 本期资本化投入（万元）
    data_source          TEXT NOT NULL,
    PRIMARY KEY (unit_id, as_of)
);

-- 开发与经营指标评分锚点方案（内置方案来自 conf/indicators.yaml，可另存为自定义方案）
CREATE TABLE IF NOT EXISTS indicator_profile (
    profile_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    spec_json    TEXT NOT NULL,
    is_builtin   INTEGER NOT NULL DEFAULT 0,
    is_default   INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT
);

-- 数据导入批次：记录本批写入的主键与被替换的旧行，支持撤销
CREATE TABLE IF NOT EXISTS import_batch (
    batch_id      TEXT PRIMARY KEY,
    import_type   TEXT NOT NULL,
    filename      TEXT,
    n_rows        INTEGER,
    summary_json  TEXT,
    keys_json     TEXT,
    replaced_json TEXT,
    status        TEXT NOT NULL,      -- active / reverted
    created_at    TEXT,
    reverted_at   TEXT
);

-- 历史评估成果：每期、每单元、每价格情景、每个构成一行。对账时直接取上期入库结果
CREATE TABLE IF NOT EXISTS sec_eval_record (
    as_of          TEXT NOT NULL,
    unit_id        TEXT NOT NULL,
    scenario       TEXT NOT NULL,      -- sec / assessment / impairment
    component      TEXT NOT NULL,      -- old_base / measure / new_infill / extension / total
    reserves_t     REAL,
    n_wells        INTEGER,
    method         TEXT,
    detail_json    TEXT,
    model_version  TEXT,
    trace_id       TEXT,
    created_at     TEXT,
    PRIMARY KEY (as_of, unit_id, scenario, component)
);

-- 单元评估快照：内核逐单元评估的完整中间结果（逐井递减拟合、措施效果、新井取值）。
-- 服务重启后直接读快照、不必重新逐井拟合；指纹覆盖数据、模型、口径、价格册与评估算法源码，
-- 任一变化旧快照自动作废。它是可随时丢弃的计算缓存，历史评估成果以 sec_eval_record 为准。
CREATE TABLE IF NOT EXISTS sec_eval_snapshot (
    unit_id        TEXT NOT NULL,
    as_of          TEXT NOT NULL,
    price_deck_id  TEXT NOT NULL,
    scenario       TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    model_version  TEXT,
    created_at     TEXT,
    PRIMARY KEY (unit_id, as_of, price_deck_id, scenario)
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
