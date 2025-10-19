# ---------------- 파라미터 ----------------
NIGHT            = range(1, 7)    # 00~05
MIN_TOTAL        = 50
NIGHT_SHARE      = 0.80
REQUIRE_MIN_DAYS = 3

SUSPECT_SIGS     = 2
CONFIRM_SIGS     = 3

MDA_SCOPE_TH     = 0.60
PUB_SCOPE_TH     = 0.60

# IP 가드(탐지 신호용)
MIN_ROWS_FOR_IP_CHECK = 100
IP_TOP_SHARE_TH       = 0.95
IP_UNIQUE_RATIO_TH    = 0.10

# --------------- 유틸 ---------------
def _to_dt_any(s):
    dt  = pd.to_datetime(s, errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        dt = dt.where(~num.between(1e9,1e11),  pd.to_datetime(num, unit="s",  errors="coerce"))
        dt = dt.where(~num.between(1e12,1e14), pd.to_datetime(num, unit="ms", errors="coerce"))
    return dt

def _to_dt_date_time(df, dcol, tcol):
    t  = df[tcol].astype(str)
    dt = pd.to_datetime(df[dcol].astype(str) + " " + t, errors="coerce")
    h  = pd.to_numeric(t, errors="coerce")
    if h.notna().any():
        dt2 = pd.to_datetime(df[dcol], errors="coerce") + pd.to_timedelta(h.fillna(0), unit="h")
        dt  = dt.fillna(dt2)
    return dt

# -------- 최종 함수: abuse_4 (스코프=mda/pub, dvc는 메타) --------
def detect_abuse4_night_scoped(df_rpt, df_join, df_settle):
    # 1) 정규화
    r = df_rpt.copy()
    r["dt"]   = _to_dt_date_time(r, "rpt_time_date", "rpt_time_time")
    r         = r[r["dt"].notna()]
    r["date"] = r["dt"].dt.date
    r["hour"] = r["dt"].dt.hour
    r["clk"]  = pd.to_numeric(r["rpt_time_clk"], errors="coerce").fillna(0).astype("int64")

    j = df_join.copy()
    j["dt"]   = _to_dt_any(j["click_date"])
    j         = j[j["dt"].notna()]
    j["date"] = j["dt"].dt.date
    j["hour"] = j["dt"].dt.hour
    j["dvc_idx"]        = pd.to_numeric(j.get("dvc_idx", 0), errors="coerce").fillna(0).astype("Int64")
    j["pub_sub_rel_id"] = pd.to_numeric(j.get("pub_sub_rel_id", 0), errors="coerce").astype("Int64")
    j["mda_idx"]        = pd.to_numeric(j.get("mda_idx", 0), errors="coerce").astype("Int64")
    j["user_ip"]        = j.get("user_ip", "").astype(str).str.strip()

    s = df_settle.copy()
    s["dt"]   = _to_dt_any(s["click_date"])
    s         = s[s["dt"].notna()]
    s["date"] = s["dt"].dt.date
    s["hour"] = s["dt"].dt.hour
    s["ctit"] = pd.to_numeric(s["ctit"], errors="coerce")

    # 2) 후보(ads×date)
    day   = r.groupby(["ads_idx","date"], as_index=False).agg(total_clk=("clk","sum")).sort_values(["ads_idx","date"])
    night = r[r["hour"].isin(NIGHT)].groupby(["ads_idx","date"], as_index=False).agg(night_clk=("clk","sum"))
    base  = day.merge(night, on=["ads_idx","date"], how="left").fillna({"night_clk":0})
    base["night_share"] = np.where(base["total_clk"]>0, base["night_clk"]/base["total_clk"], np.nan)
    base["base_days"]   = base.groupby("ads_idx").cumcount() + 1

    cand = base[(base["total_clk"]>=MIN_TOTAL) & (base["night_share"]>=NIGHT_SHARE) & (base["base_days"]>=REQUIRE_MIN_DAYS)].copy()
    if cand.empty:
        cols = ["ads_idx","date","total_clk","night_clk","night_share","base_days","signals","severity",
                "scope_type","scope_id","scope_share",
                "dvc_top_id","dvc_top_share","dvc_top_cnt","dvc_in_pub_share","dvc_top_in_pub_id","dvc_top_in_pub_cnt",
                "pub_mda_idx"]
        return pd.DataFrame(columns=cols)
    cand = cand.reset_index(drop=True)

    # 3) 신호 5개
    # (a) 전환률 급락
    settles_d = s.groupby(["ads_idx","date"], as_index=False).agg(settles=("click_key","nunique"))
    day_conv  = day.merge(settles_d, on=["ads_idx","date"], how="left").fillna({"settles":0})
    day_conv["cr"] = np.where(day_conv["total_clk"]>0, day_conv["settles"]/day_conv["total_clk"], np.nan)
    day_conv = day_conv.sort_values(["ads_idx","date"])
    cr_med7  = (day_conv.groupby("ads_idx")["cr"].rolling(7, min_periods=3).median()
                .reset_index(level=0, drop=True))
    day_conv["cr_med7"] = cr_med7.values

    s_n  = s[s["hour"].isin(NIGHT)].copy()
    n_set = s_n.groupby(["ads_idx","date"], as_index=False).agg(night_settles=("click_key","nunique"))
    ncv   = night.merge(n_set, on=["ads_idx","date"], how="left").fillna({"night_settles":0})
    ncv["cr_night"] = np.where(ncv["night_clk"]>0, ncv["night_settles"]/ncv["night_clk"], np.nan)

    conv   = (cand.merge(day_conv[["ads_idx","date","cr_med7"]], on=["ads_idx","date"], how="left")
                   .merge(ncv[["ads_idx","date","cr_night"]],      on=["ads_idx","date"], how="left"))
    f_conv = ((conv["cr_med7"].notna()) & (conv["cr_night"].notna()) & (conv["cr_night"] <= 0.5*conv["cr_med7"])).reset_index(drop=True)

    # (b) 퍼블리셔 지배
    jn = j[j["hour"].isin(NIGHT)]
    pub_cnt = jn.groupby(["ads_idx","date","pub_sub_rel_id"]).size().reset_index(name="cnt")
    pub_top = pd.DataFrame(columns=["ads_idx","date","pub_sub_rel_id","pub_share"])
    if not pub_cnt.empty:
        pub_tot = pub_cnt.groupby(["ads_idx","date"])["cnt"].sum().reset_index(name="tot")
        pub_top = (pub_cnt.sort_values(["ads_idx","date","cnt"], ascending=[True,True,False])
                          .drop_duplicates(["ads_idx","date"])
                          .merge(pub_tot, on=["ads_idx","date"], how="left"))
        pub_top["pub_share"] = np.where(pub_top["tot"]>0, pub_top["cnt"]/pub_top["tot"], 0.0)
    f_pub = cand.merge(pub_top[["ads_idx","date","pub_share"]], on=["ads_idx","date"], how="left")["pub_share"].fillna(0).ge(PUB_SCOPE_TH).reset_index(drop=True)

    # (c) dvc 다양성 저하(탐지 신호)
    uv = jn.groupby(["ads_idx","date"]).agg(clicks=("click_key","nunique"), uniq=("dvc_idx","nunique")).reset_index()
    uv["uniq_share"] = np.where(uv["clicks"]>0, uv["uniq"]/uv["clicks"], np.nan)
    f_uniq = cand.merge(uv[["ads_idx","date","uniq_share"]], on=["ads_idx","date"], how="left")["uniq_share"].fillna(1).le(0.3).reset_index(drop=True)

    # (d) CTIT 왜곡
    s_n["short"] = s_n["ctit"].le(3)
    s_n["long"]  = s_n["ctit"].ge(24*3600)
    ct = s_n.groupby(["ads_idx","date"]).agg(n=("click_key","nunique"), s=("short","sum"), l=("long","sum")).reset_index()
    ct["sh_short"] = np.where(ct["n"]>0, ct["s"]/ct["n"], np.nan)
    ct["sh_long"]  = np.where(ct["n"]>0, ct["l"]/ct["n"], np.nan)
    _tmp = cand.merge(ct[["ads_idx","date","sh_short","sh_long"]], on=["ads_idx","date"], how="left")
    f_ctit = (_tmp["sh_short"].fillna(0).ge(0.6) | _tmp["sh_long"].fillna(0).ge(0.2)).reset_index(drop=True)

    # (e) IP 팬아웃(탐지 신호)
    ipfan  = jn.groupby(["ads_idx","date","user_ip"])["dvc_idx"].nunique().reset_index(name="n_dvc")
    ipflag = ipfan.groupby(["ads_idx","date"])["n_dvc"].max().reset_index(name="max_dvc")
    f_ip = cand.merge(ipflag, on=["ads_idx","date"], how="left")["max_dvc"].fillna(0).ge(5).reset_index(drop=True)

    # 4) signals & severity
    signals = (
        f_conv.values.astype("int8")
        + f_pub.values.astype("int8")
        + f_uniq.values.astype("int8")
        + f_ctit.values.astype("int8")
        + f_ip.values.astype("int8")
    ).astype("int8")
    sev = np.where(signals >= CONFIRM_SIGS, "확정",
           np.where(signals >= SUSPECT_SIGS, "의심", "후보"))
    cases = cand.assign(signals=signals, severity=sev)
    cases = cases[cases["severity"].isin(["의심","확정"])].reset_index(drop=True)
    if cases.empty:
        cols = ["ads_idx","date","total_clk","night_clk","night_share","base_days","signals","severity",
                "scope_type","scope_id","scope_share",
                "dvc_top_id","dvc_top_share","dvc_top_cnt","dvc_in_pub_share","dvc_top_in_pub_id","dvc_top_in_pub_cnt",
                "pub_mda_idx"]
        return pd.DataFrame(columns=cols)

    # 5) 스코프: mda → pub (끝)
    # mda
    mda_cnt = jn.groupby(["ads_idx","date","mda_idx"]).size().reset_index(name="cnt")
    mda_top = pd.DataFrame(columns=["ads_idx","date","mda_idx","mda_share"])
    if not mda_cnt.empty:
        mda_tot = mda_cnt.groupby(["ads_idx","date"])["cnt"].sum().reset_index(name="tot")
        mda_top = (mda_cnt.sort_values(["ads_idx","date","cnt"], ascending=[True,True,False])
                           .drop_duplicates(["ads_idx","date"])
                           .merge(mda_tot, on=["ads_idx","date"], how="left"))
        mda_top["mda_share"] = np.where(mda_top["tot"]>0, mda_top["cnt"]/mda_top["tot"], 0.0)
    cases = cases.merge(mda_top[["ads_idx","date","mda_idx","mda_share"]], on=["ads_idx","date"], how="left").fillna({"mda_share":0.0})

    # pub
    cases = cases.merge(pub_top[["ads_idx","date","pub_sub_rel_id","pub_share"]], on=["ads_idx","date"], how="left").fillna({"pub_share":0.0})

    # pub별 mda_idx 매핑 준비 (pub_mda_map)
    # 각 (ads_idx, date, pub_sub_rel_id)별로 가장 많은 mda_idx를 pub_mda_idx로 지정
    pub_mda_map = pd.DataFrame()
    if not jn.empty:
        pub_mda_map = (
            jn.groupby(["ads_idx", "date", "pub_sub_rel_id", "mda_idx"])
              .size().reset_index(name="cnt")
        )
        # 각 pub_sub_rel_id별로 cnt가 가장 큰 mda_idx를 선택
        pub_mda_map = (
            pub_mda_map.sort_values(["ads_idx", "date", "pub_sub_rel_id", "cnt"], ascending=[True, True, True, False])
                       .drop_duplicates(["ads_idx", "date", "pub_sub_rel_id"])
                       .rename(columns={"mda_idx": "pub_mda_idx"})
                       [["ads_idx", "date", "pub_sub_rel_id", "pub_mda_idx"]]
        )

    # 스코프 결정
    scope_type = np.where(cases["mda_share"] >= MDA_SCOPE_TH, "mda",
                   np.where(cases["pub_share"] >= PUB_SCOPE_TH, "pub", "pub"))  # 최소 pub로 귀속
    scope_id = np.where(scope_type=="mda", cases["mda_idx"].astype("Int64"),
                 cases["pub_sub_rel_id"].astype("Int64"))
    scope_share = np.where(scope_type=="mda", cases["mda_share"], cases["pub_share"])

    # 6) dvc 참고 메타(리포트 컬럼)
    # 전체 top dvc (웹=0 제외)
    jn_d = jn[jn["dvc_idx"].fillna(0).astype(int) != 0]
    dvc_top = pd.DataFrame(columns=["ads_idx","date","dvc_top_id","dvc_top_share","dvc_top_cnt"])
    if not jn_d.empty:
        dvc_cnt = jn_d.groupby(["ads_idx","date","dvc_idx"]).size().reset_index(name="cnt")
        dvc_tot = dvc_cnt.groupby(["ads_idx","date"])["cnt"].sum().reset_index(name="tot")
        dvc_top = (dvc_cnt.sort_values(["ads_idx","date","cnt"], ascending=[True,True,False])
                           .drop_duplicates(["ads_idx","date"])
                           .merge(dvc_tot, on=["ads_idx","date"], how="left"))
        dvc_top["dvc_top_share"] = np.where(dvc_top["tot"]>0, dvc_top["cnt"]/dvc_top["tot"], 0.0)
        dvc_top = dvc_top.rename(columns={"dvc_idx":"dvc_top_id","cnt":"dvc_top_cnt"})
        dvc_top = dvc_top[["ads_idx","date","dvc_top_id","dvc_top_share","dvc_top_cnt"]]

    # dominant pub 내부 top dvc
    dvc_in_pub_top = pd.DataFrame(columns=["ads_idx","date","pub_sub_rel_id","dvc_top_in_pub_id","dvc_in_pub_share","dvc_top_in_pub_cnt"])
    if not jn_d.empty and not pub_cnt.empty:
        dvc_in_pub = jn_d.groupby(["ads_idx","date","pub_sub_rel_id","dvc_idx"]).size().reset_index(name="cnt")
        pub_tot2   = dvc_in_pub.groupby(["ads_idx","date","pub_sub_rel_id"])["cnt"].sum().reset_index(name="tot")
        dvc_in_pub_top = (dvc_in_pub.sort_values(["ads_idx","date","pub_sub_rel_id","cnt"],
                                                 ascending=[True,True,True,False])
                                     .drop_duplicates(["ads_idx","date","pub_sub_rel_id"])
                                     .merge(pub_tot2, on=["ads_idx","date","pub_sub_rel_id"], how="left"))
        dvc_in_pub_top["dvc_in_pub_share"] = np.where(dvc_in_pub_top["tot"]>0, dvc_in_pub_top["cnt"]/dvc_in_pub_top["tot"], 0.0)
        dvc_in_pub_top = dvc_in_pub_top.rename(columns={"dvc_idx":"dvc_top_in_pub_id","cnt":"dvc_top_in_pub_cnt"})
        dvc_in_pub_top = dvc_in_pub_top[["ads_idx","date","pub_sub_rel_id","dvc_top_in_pub_id","dvc_in_pub_share","dvc_top_in_pub_cnt"]]

    out = (cases.assign(scope_type=scope_type, scope_id=scope_id, scope_share=scope_share)
                 .merge(dvc_top, on=["ads_idx","date"], how="left")
                 .merge(dvc_in_pub_top, on=["ads_idx","date","pub_sub_rel_id"], how="left")
                 .fillna({"dvc_top_share":0.0, "dvc_in_pub_share":0.0,
                          "dvc_top_cnt":0, "dvc_top_in_pub_cnt":0})
                 .drop(columns=["mda_idx","mda_share"], errors="ignore")
                 .sort_values(["severity","signals","night_share"], ascending=[False,False,False])
                 .reset_index(drop=True))

    # pub_mda_idx 컬럼 추가: 스코프가 pub인 경우에만 pub_mda_idx를 표시, 아니면 NaN
    if not pub_mda_map.empty:
        out = out.merge(pub_mda_map, on=["ads_idx", "date", "pub_sub_rel_id"], how="left")
    else:
        out["pub_mda_idx"] = pd.NA

    # 스코프가 pub이 아닌 경우 pub_mda_idx를 NaN으로
    out["pub_mda_idx"] = np.where(out["scope_type"] == "pub", out["pub_mda_idx"], pd.NA)

    cols = ["ads_idx","date","total_clk","night_clk","night_share","base_days","signals","severity",
            "scope_type","scope_id","scope_share",
            "dvc_top_id","dvc_top_share","dvc_top_cnt","dvc_in_pub_share","dvc_top_in_pub_id","dvc_top_in_pub_cnt",
            "pub_mda_idx"]
    return out[cols]

abuse_4 = detect_abuse4_night_scoped(df_rpt, df_join, df_settle)

SEV2CODE = {"확정":2, "의심":1}
NIGHT = range(1,7)  # 00~05시

# -- 공통: abuse_4 결과(ads_idx,date,scope_type,scope_id,severity) → 키 정규화
def _prep_abuse4_keys(abuse4: pd.DataFrame) -> pd.DataFrame:
    a = abuse4[["ads_idx","date","scope_type","scope_id","severity"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.date
    a = a[a["severity"].isin(SEV2CODE.keys())]
    a["abuse_4"] = a["severity"].map(SEV2CODE).astype(np.int8)
    return a

# ======================== 1) df_join 라벨 주입 ========================
def add_abuse4_to_join(df_join: pd.DataFrame, abuse4: pd.DataFrame) -> pd.DataFrame:
    j = df_join.copy()
    dt = pd.to_datetime(j["click_date"], errors="coerce")
    date_vals = dt.dt.date
    hour_vals = dt.dt.hour

    # 라벨 키 준비(후보 제외)
    a = abuse4[["ads_idx","date","scope_type","scope_id","severity"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.date
    a = a[a["severity"].isin(SEV2CODE)]
    a["abuse_4"] = a["severity"].map(SEV2CODE).astype("int8")

    lab = np.zeros(len(j), dtype="int8")
    scopes = {"mda":"mda_idx", "pub":"pub_sub_rel_id", "dvc":"dvc_idx", "ip":"user_ip"}

    def apply_scope(scope_name: str):
        col = scopes[scope_name]
        if col not in j.columns: 
            return
        key = a[a["scope_type"]==scope_name].rename(columns={"scope_id": col})
        if key.empty: 
            return

        left = pd.DataFrame({"ads_idx": j["ads_idx"].values, "date": date_vals})
        if col == "user_ip":
            left[col] = j[col].astype(str).values
            key[col]  = key[col].astype(str)
        else:
            left[col] = pd.to_numeric(j[col], errors="coerce")
            key[col]  = pd.to_numeric(key[col], errors="coerce")

        m = left.merge(key[["ads_idx","date",col,"abuse_4"]],
                       on=["ads_idx","date",col], how="left")
        np.maximum(lab, m["abuse_4"].fillna(0).astype("int8").to_numpy(), out=lab)

    for s in ("mda","pub","dvc","ip"):
        apply_scope(s)

    j["abuse_4"] = np.where(pd.Series(hour_vals).isin(NIGHT), lab, 0).astype(np.int8)
    return j

# ======================== 2) df_rpt 라벨 주입 ========================
def add_abuse4_to_rpt(df_rpt: pd.DataFrame, abuse4: pd.DataFrame) -> pd.DataFrame:
    # 라벨 키 준비(확정=2, 의심=1)
    a = abuse4[["ads_idx","date","severity"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.date
    a = a[a["severity"].isin({"확정","의심"})]
    a["abuse_4"] = a["severity"].map({"확정":2, "의심":1}).astype(np.int8)
    key = (a.groupby(["ads_idx","date"])["abuse_4"]
             .max().reset_index().rename(columns={"abuse_4":"abuse_4_case"}))

    # 원본 보존
    r = df_rpt.copy()

    # 견고한 시간 파싱(숫자 시 보정)
    t  = r["rpt_time_time"].astype(str)
    dt = pd.to_datetime(r["rpt_time_date"].astype(str) + " " + t, errors="coerce")
    h  = pd.to_numeric(r["rpt_time_time"], errors="coerce")
    if h.notna().any():
        dt2 = pd.to_datetime(r["rpt_time_date"], errors="coerce") + pd.to_timedelta(h.fillna(0), unit="h")
        dt  = dt.fillna(dt2)

    # 임시 키/시각(컬럼로 붙이지 않음)
    date_vals = dt.dt.date
    hour_vals = dt.dt.hour

    # ads_idx + date로 abuse_4_case 매핑 (임시 df에서 매핑 후 다시 정렬)
    tmp = pd.DataFrame({"ads_idx": r["ads_idx"].values, "date": date_vals})
    tmp = tmp.merge(key, on=["ads_idx","date"], how="left")
    abuse_case = tmp["abuse_4_case"]  # Series

    # 야간(01~06시)만 라벨 주입, 나머지는 0
    mask_night = pd.Series(hour_vals).isin(NIGHT) & pd.Series(date_vals).notna()
    r["abuse_4"] = np.where(mask_night, abuse_case.fillna(0), 0).astype(np.int8)

    return r

# ======================== 3) df_list 비율 주입 ========================
def add_abuse4_rate_to_list(df_list: pd.DataFrame, df_join_v1: pd.DataFrame) -> pd.DataFrame:
    # 야간행 기준: 분모=야간조인 전체, 분자=야간조인 중 abuse_4>0
    j = df_join_v1.copy()
    j["dt"]   = pd.to_datetime(j["click_date"], errors="coerce")
    j         = j[j["dt"].notna()]
    j["hour"] = j["dt"].dt.hour

    den = j[j["hour"].isin(NIGHT)].groupby("ads_idx").size().reset_index(name="den")
    num = j[(j["hour"].isin(NIGHT)) & (j["abuse_4"]>0)].groupby("ads_idx").size().reset_index(name="num")
    rate = den.merge(num, on="ads_idx", how="left").fillna({"num":0})
    rate["abuse_4"] = (rate["num"]/rate["den"]).clip(0,1).astype("float32")

    out = df_list.merge(rate[["ads_idx","abuse_4"]], on="ads_idx", how="left").fillna({"abuse_4":0.0})
    return out

# ======================== 2) df_settle 라벨 주입 ========================
def add_abuse4_to_settle(df_settle: pd.DataFrame, abuse4: pd.DataFrame) -> pd.DataFrame:
    """
    abuse_4 결과(ads_idx, date, severity) 기반으로 df_settle 각 행에 abuse_4(0/1/2) 붙이기.
    - 확정=2, 의심=1, 그 외(후보/미해당)=0
    - 야간(01~06시)만 라벨 적용, 나머지 시간대는 0
    - 원본 컬럼 보존: date/hour 등 임시 계산만 사용
    """
    # 라벨 키 준비(확정/의심만)
    a = abuse4[["ads_idx","date","severity"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.date
    a = a[a["severity"].isin({"확정","의심"})]
    a["abuse_4"] = a["severity"].map({"확정":2, "의심":1}).astype(np.int8)

    key = (a.groupby(["ads_idx","date"])["abuse_4"]
             .max().reset_index().rename(columns={"abuse_4":"abuse_4_case"}))

    # 원본 보존
    s = df_settle.copy()

    # 견고한 시각 파싱 (초/밀리초 숫자도 허용)
    dt = _to_dt_any(s["click_date"])
    date_vals = dt.dt.date
    hour_vals = dt.dt.hour

    # ads_idx + date로 abuse_4_case 매핑 (임시 DF로 매핑 후 값만 사용)
    tmp = pd.DataFrame({"ads_idx": s["ads_idx"].values, "date": date_vals})
    tmp = tmp.merge(key, on=["ads_idx","date"], how="left")

    # 야간(01~06시)만 라벨 주입, 그 외 시간대는 0
    NIGHT = range(1,7)  # 01~06시
    s["abuse_4"] = np.where(pd.Series(hour_vals).isin(NIGHT), tmp["abuse_4_case"].fillna(0), 0).astype(np.int8)

    return s

# 2) 원본 라벨 주입
df_join_v1 = add_abuse4_to_join(df_join_v1, abuse_4)   # 행단위 0/1/2 (야간만)
df_rpt_v1  = add_abuse4_to_rpt(df_rpt_v1, abuse_4)     # 시간행 0/1/2 (야간만)
df_list_v1 = add_abuse4_rate_to_list(df_list_v1, df_join_v1)  # 광고별 0~1 비율
df_settle_v1 = add_abuse4_to_settle(df_settle_v1, abuse_4)  # settle 추가

# 저장
df_join_v1.to_csv("df_join_v1.csv", index=False)
df_rpt_v1.to_csv("df_rpt_v1.csv", index=False)
df_list_v1.to_csv("df_list_v1.csv", index=False)
df_settle_v1.to_csv("df_settle_v1.csv", index=False)