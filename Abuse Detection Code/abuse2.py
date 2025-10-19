# ========== 파라미터 ==========
REJOIN_UNIT = "ads_idx"            # 'ads_idx'(권장) | 'ads_code'
CODE_FALLBACK_STRATEGY = "latest"  # 'latest'|'mode' (REJOIN_UNIT = 'ads_code'일 때만 사용)
USE_CLICK_KEY = True               # True: click_key 고유개수 / False: 행수

# IP 가드 정책
WEB_IP_POLICY = "guarded"          # 'guarded' | 'use' | 'ignore'
IP_TOP_SHARE_TH = 0.95
IP_UNIQUE_RATIO_TH = 0.001
MIN_ROWS_FOR_IP_CHECK = 200
GROUP_KEY_ORDER = ("mda_idx", "pub_sub_rel_id")

# ---------- 유틸: IP 가드 ----------
def _pick_group_key(df: pd.DataFrame) -> str | None:
    for k in GROUP_KEY_ORDER:
        if k in df.columns:
            return k
    return None

def _ip_bad_groups(df: pd.DataFrame, key: str | None) -> set:
    """IP 품질 나쁜 그룹 set 반환. key=None이면 전체를 하나의 그룹으로 보고 판정."""
    if "user_ip" not in df.columns:
        # user_ip 자체가 없으면 웹은 전부 배제 대상(guarded에서)
        return set([-1]) if key is None else set(df[key].unique())

    if key is None:
        vc = df["user_ip"].astype(str).value_counts(dropna=False)
        rows = len(df); uniq = int(vc.size)
        top1_share = float(vc.iloc[0]/rows) if rows else 1.0
        unique_ratio = max(1, uniq)/max(1, rows)
        bad = (rows >= MIN_ROWS_FOR_IP_CHECK) and ((top1_share >= IP_TOP_SHARE_TH) or (unique_ratio <= IP_UNIQUE_RATIO_TH))
        return set([-1]) if bad else set()

    g = df.groupby(key, observed=True)
    rows = g.size()
    uniq = g["user_ip"].apply(lambda s: s.astype(str).nunique())
    top1 = g["user_ip"].apply(lambda s: s.astype(str).value_counts(dropna=False).max())
    top1_share = (top1 / rows).astype(float)
    unique_ratio = uniq.clip(lower=1) / rows.clip(lower=1)
    bad_idx = rows.index[(rows >= MIN_ROWS_FOR_IP_CHECK) & ((top1_share >= IP_TOP_SHARE_TH) | (unique_ratio <= IP_UNIQUE_RATIO_TH))]
    return set(bad_idx)

def _build_user_id_with_guard(df: pd.DataFrame) -> pd.Series:
    """dvc 우선, 웹은 정책/품질에 따라 포함/배제."""
    dvc = pd.to_numeric(df.get("dvc_idx", 0), errors="coerce").fillna(0).astype("Int64")
    ip  = df.get("user_ip", pd.Series("", index=df.index)).astype(str).str.strip()

    if WEB_IP_POLICY == "ignore":
        return np.where(dvc.ne(0), "dvc:"+dvc.astype(str), np.nan)

    key = _pick_group_key(df)
    bad = _ip_bad_groups(df, key)

    if key is None:
        web_ok = ("user_ip" in df.columns) and ((-1) not in bad)
        return np.where(
            dvc.ne(0), "dvc:"+dvc.astype(str),
            np.where((WEB_IP_POLICY=="guarded") & (~web_ok), np.nan, "ip:"+ip)
        )

    keyvals = df[key]
    in_bad = keyvals.astype("Int64").isin(bad) if pd.api.types.is_integer_dtype(keyvals) else keyvals.astype(str).isin(bad)
    return np.where(
        dvc.ne(0), "dvc:"+dvc.astype(str),
        np.where((WEB_IP_POLICY=="guarded") & in_bad, np.nan, "ip:"+ip)
    )

# ---------- 유틸: 코드 정규화(ads_code 단위 사용 시) ----------
def _safe_strategy(s: str) -> str:
    s = (s or "").lower()
    return "latest" if s in ("latest","lastest") else ("mode" if s=="mode" else "latest")

def _rep_code_by_idx(df_any: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    ads_idx별 대표코드(정산 기준) 선택:
      - latest: 가장 최근(click_date→regdate 순)
      - mode  : 최빈 코드
    """
    d = df_any.copy()
    if "ads_code" in d.columns:
        d = d.rename(columns={"ads_code":"ads_code_settle"})
    if "click_date" in d.columns:
        d["_dt"] = pd.to_datetime(d["click_date"], errors="coerce")
    elif "regdate" in d.columns:
        d["_dt"] = pd.to_datetime(d["regdate"], errors="coerce")
    else:
        d["_dt"] = pd.NaT

    if strategy == "latest":
        rep = (d.dropna(subset=["ads_code_settle"])
                 .sort_values(["ads_idx","_dt"])
                 .groupby("ads_idx", as_index=False).tail(1)[["ads_idx","ads_code_settle"]]
                 .rename(columns={"ads_code_settle":"ads_code_settle_rep"}))
    else:
        def _mode(x):
            x = x.dropna()
            if x.empty: return np.nan
            m = x.mode()
            return m.iloc[0] if not m.empty else x.iloc[0]
        rep = (d.groupby("ads_idx")["ads_code_settle"]
                 .apply(_mode).reset_index()
                 .rename(columns={"ads_code_settle":"ads_code_settle_rep"}))
    return rep

# ---------- 공통 전처리 ----------
def _prepare_data(df_list: pd.DataFrame, df_settle: pd.DataFrame, df_join: pd.DataFrame):
    """
    df_settle(적립) 기준으로 df_join에서 user_ip를 click_key로 부착 → 날짜/키/유저 정규화
    반환: s (분석용 테이블), ts(타임스탬프 컬럼명)
    """
    # 1) user_ip 붙이기 (INNER: 정산에 없는 click_key는 제외)
    ipmap = df_join[["click_key","user_ip"]].dropna().drop_duplicates()
    s = df_settle.merge(ipmap, on="click_key", how="inner", validate="m:1").copy()

    # 2) 메타 결합(ads_rejoin_type, 목록 코드)
    meta = df_list[["ads_idx","ads_rejoin_type","ads_code"]].drop_duplicates().rename(columns={"ads_code":"ads_code_list"})
    s = s.merge(meta, on="ads_idx", how="left", validate="m:1")

    # 3) 날짜 통일
    ts = None
    if "click_date" in s.columns:
        s["click_date"] = pd.to_datetime(s["click_date"], errors="coerce"); ts = "click_date"
    elif "regdate" in s.columns:
        s["regdate"] = pd.to_datetime(s["regdate"], errors="coerce"); ts = "regdate"
    if ts is None:
        raise KeyError("df_settle에 click_date/regdate 중 하나는 있어야 합니다.")
    s = s[s[ts].notna()].copy()
    s["click_day"] = s[ts].dt.date

    # 4) 판단 키(grp) 구성
    if REJOIN_UNIT == "ads_idx":
        s["grp"] = s["ads_idx"]
    else:
        strat = _safe_strategy(CODE_FALLBACK_STRATEGY)
        rep = _rep_code_by_idx(s.rename(columns={"ads_code_list":"ads_code"}), strat)
        cmap = meta[["ads_idx","ads_code_list"]].merge(rep, on="ads_idx", how="outer")
        cmap["ad_code_key"] = cmap["ads_code_list"].fillna(cmap["ads_code_settle_rep"])
        s = s.merge(cmap[["ads_idx","ad_code_key"]], on="ads_idx", how="left", validate="m:1")
        s["grp"] = s["ad_code_key"]

    # 5) user_id (IP 가드)
    s["user_id"] = _build_user_id_with_guard(s)
    s = s[s["user_id"].notna() & s["grp"].notna()].copy()

    return s, ts

# ---------- 집계 유틸(부분프레임을 반드시 넣는다!) ----------
def _agg(df: pd.DataFrame, by_cols: list[str], use_click_key: bool, ts: str):
    if df.empty:
        return pd.DataFrame(columns=by_cols + ["cnt","first_click","last_click"])
    if use_click_key and ("click_key" in df.columns):
        return (df.groupby(by_cols, as_index=False)
                  .agg(cnt=("click_key","nunique"),
                       first_click=(ts,"min"),
                       last_click=(ts,"max")))
    else:
        return (df.groupby(by_cols, as_index=False)
                  .agg(cnt=(ts,"size"),
                       first_click=(ts,"min"),
                       last_click=(ts,"max")))

# ---------- (1) 탐지 ----------
def detect_abuse(df_list: pd.DataFrame, df_settle: pd.DataFrame, df_join: pd.DataFrame):
    """
    반환: (rejoin_violation, daily_dup)
      - 재참여_위반(NONE): user_id × grp 2회 이상
      - 일일중복_위반(DAILY_UPDATE): user_id × grp × click_day 2회 이상
      - grp = REJOIN_UNIT ('ads_idx' 또는 정규화된 'ad_code_key')
      - first/last는 타임스탬프(시:분:초) 유지
    """
    s, ts = _prepare_data(df_list, df_settle, df_join)

    # 재참여 불가(NONE)만 집계
    s_none = s[s["ads_rejoin_type"] == "NONE"].copy()
    if len(s_none):
        g_none = _agg(s_none, ["user_id","grp"], USE_CLICK_KEY, ts)
        rejoin_violation = g_none[g_none["cnt"] >= 2].copy()
        rejoin_violation["abuse_type"] = "재참여_위반(NONE)"
    else:
        rejoin_violation = pd.DataFrame(columns=["abuse_type","user_id","grp","cnt","first_click","last_click"])

    # 일일중복(DAILY_UPDATE)만 집계
    s_daily = s[s["ads_rejoin_type"] == "ADS_CODE_DAILY_UPDATE"].copy()
    if len(s_daily):
        g_daily = _agg(s_daily, ["user_id","grp","click_day"], USE_CLICK_KEY, ts)
        daily_dup = g_daily[g_daily["cnt"] >= 2].copy()
        daily_dup["abuse_type"] = "일일중복_위반(DAILY_UPDATE)"
    else:
        daily_dup = pd.DataFrame(columns=["abuse_type","user_id","grp","click_day","cnt","first_click","last_click"])

    # 열 이름 정리 및 정렬
    key_name = "ads_idx" if REJOIN_UNIT == "ads_idx" else "ad_code_key"
    rejoin_violation = (rejoin_violation.rename(columns={"grp": key_name})
                        .loc[:, ["abuse_type","user_id",key_name,"cnt","first_click","last_click"]]
                        .sort_values(["cnt","last_click"], ascending=[False, True]).reset_index(drop=True))
    daily_dup = (daily_dup.rename(columns={"grp": key_name})
                 .loc[:, ["abuse_type","user_id",key_name,"click_day","cnt","first_click","last_click"]]
                 .sort_values(["cnt","last_click"], ascending=[False, True]).reset_index(drop=True))
    return rejoin_violation, daily_dup

# 결과
rejoin_violation, daily_dup = detect_abuse(df_list, df_settle, df_join)

def _prepare_join_for_label_v1(df_list_v1: pd.DataFrame,
                               df_join_v1: pd.DataFrame,
                               df_settle: pd.DataFrame):
    """df_join_v1을 라벨 전파용으로 정규화 (grp, click_day, user_id)."""
    j = df_join_v1.copy()

    # meta 붙이기 (ads_rejoin_type, ads_code_list)
    meta = (df_list_v1[["ads_idx","ads_rejoin_type","ads_code"]]
            .drop_duplicates().rename(columns={"ads_code":"ads_code_list"}))
    j = j.merge(meta, on="ads_idx", how="left", validate="m:1")

    # 날짜/일자
    if "click_date" not in j.columns:
        raise KeyError("df_join_v1에 click_date 필요")
    j["click_date"] = pd.to_datetime(j["click_date"], errors="coerce")
    j = j[j["click_date"].notna()].copy()
    j["click_day"] = j["click_date"].dt.date

    # grp 결정
    if REJOIN_UNIT == "ads_idx":
        j["grp"] = j["ads_idx"]
    else:
        # ads_code 단위 → 대표코드(ad_code_key) 정규화
        strat = _safe_strategy(CODE_FALLBACK_STRATEGY)
        rep = _rep_code_by_idx(df_settle.rename(columns={"ads_code":"ads_code_settle"}), strat)
        cmap = meta[["ads_idx","ads_code_list"]].merge(rep, on="ads_idx", how="outer")
        cmap["ad_code_key"] = cmap["ads_code_list"].fillna(cmap["ads_code_settle_rep"])
        j = j.merge(cmap[["ads_idx","ad_code_key"]], on="ads_idx", how="left", validate="m:1")
        j["grp"] = j["ad_code_key"]

    # user_id (IP 가드 동일 적용)
    j["user_id"] = _build_user_id_with_guard(j)
    j = j[j["user_id"].notna() & j["grp"].notna()].copy()
    return j


def _prepare_settle_for_label(df_list_v1: pd.DataFrame,
                              df_settle: pd.DataFrame,
                              df_join_v1: pd.DataFrame):
    """
    df_settle 기준으로 user_ip/메타를 붙여 라벨링용 보조 컬럼(click_day, user_id, grp)을 만든다.
    ✅ 행수 보존: 입력 df_settle과 동일한 행 수 유지
    - user_ip 매칭 실패, 날짜 파싱 실패, grp/user_id 결측 모두 유지 (라벨링 단계에서 0으로 처리)
    """
    import pandas as pd
    import numpy as np

    # --- 0) 시작 행수 기록
    n0 = len(df_settle)

    # --- 1) click_key→user_ip 매핑 (LEFT 조인, 드랍 금지)
    ipmap = (df_join_v1[["click_key","user_ip"]]
                .drop_duplicates(subset=["click_key"]))  # click_key 만-일 보장
    s = df_settle.merge(ipmap, on="click_key", how="left", validate="m:1").copy()

    # --- 2) 광고 메타 (LEFT 조인, 드랍 금지)
    meta = (df_list_v1[["ads_idx","ads_rejoin_type","ads_code"]]
            .drop_duplicates()
            .rename(columns={"ads_code":"ads_code_list"}))
    s = s.merge(meta, on="ads_idx", how="left", validate="m:1")

    # --- 3) 날짜 파싱 (드랍 금지)
    ts = "click_date" if "click_date" in s.columns else ("regdate" if "regdate" in s.columns else None)
    if ts is None:
        raise KeyError("df_settle에 click_date 또는 regdate 필요")
    dt = pd.to_datetime(s[ts], errors="coerce")
    s["click_day"] = dt.dt.date  # NaT는 그대로 남김

    # --- 4) 그룹키(grp) 생성: ads_idx 기본, 코드 전략 있으면 사용. 결측은 ads_idx로 폴백
    def _fallback_grp(df):
        # ads_idx가 숫자면 그대로, 아니면 문자열 변환
        return df["ads_idx"]

    try:
        # 외부 전역 설정이 있으면 사용
        if "REJOIN_UNIT" in globals() and REJOIN_UNIT == "ads_idx":
            s["grp"] = s["ads_idx"]
        else:
            # 코드 기반 전략이 설정돼 있으면 시도, 실패 시 폴백
            try:
                strat = _safe_strategy(CODE_FALLBACK_STRATEGY)
                rep = _rep_code_by_idx(s.rename(columns={"ads_code_list":"ads_code"}), strat)
                cmap = meta[["ads_idx","ads_code_list"]].merge(rep, on="ads_idx", how="outer")
                cmap["ad_code_key"] = cmap["ads_code_list"].fillna(cmap.get("ads_code_settle_rep"))
                s = s.merge(cmap[["ads_idx","ad_code_key"]], on="ads_idx", how="left", validate="m:1")
                s["grp"] = s["ad_code_key"].where(s["ad_code_key"].notna(), _fallback_grp(s))
            except Exception:
                s["grp"] = _fallback_grp(s)
    except NameError:
        s["grp"] = _fallback_grp(s)

    # --- 5) user_id 생성: 실패/결측도 보존. 폴백 계층: dvc→ip→click_key
    def _build_user_id_safe(df):
        # dvc_idx 사용 가능하면 dvc:xxx
        if "dvc_idx" in df.columns:
            dvc = pd.to_numeric(df["dvc_idx"], errors="coerce").fillna(0).astype("int64")
        else:
            dvc = pd.Series(0, index=df.index, dtype="int64")
        has_dvc = dvc.ne(0)

        ip = df.get("user_ip")
        has_ip = ip.notna() & (ip.astype(str).str.len() > 0) if ip is not None else pd.Series(False, index=df.index)

        uid = pd.Series(pd.NA, index=df.index, dtype="object")
        uid = np.where(has_dvc, "dvc:" + dvc.astype(str), uid)
        uid = np.where(~has_dvc & has_ip, "ip:" + ip.astype(str), uid)
        uid = np.where(pd.isna(uid), "ck:" + df["click_key"].astype(str), uid)
        return pd.Series(uid, index=df.index, dtype="object")

    try:
        s["user_id"] = _build_user_id_with_guard(s)  # 네가 가지고 있는 헬퍼 우선 사용
        # 혹시 반환에 결측 있으면 안전 폴백 적용
        miss = s["user_id"].isna() | (s["user_id"].astype(str).str.len()==0)
        if miss.any():
            s.loc[miss, "user_id"] = _build_user_id_safe(s.loc[miss])
    except Exception:
        s["user_id"] = _build_user_id_safe(s)

    # --- 6) 절대 드랍 금지: 결측은 그대로 둔다 (라벨링 단계에서 0 처리)
    #     s = s[s["user_id"].notna() & s["grp"].notna()]  # ❌ 삭제

    # --- 7) 행수 보존 확인
    assert len(s) == n0, f"행수 변동 감지: before={n0}, after={len(s)}"

    return s


def update_abuse2_with_v1(
    df_list_v1: pd.DataFrame,
    df_join_v1: pd.DataFrame,
    df_settle: pd.DataFrame,
    rejoin_violation: pd.DataFrame,
    daily_dup: pd.DataFrame
):
    """
    - df_settle_v1  : 새로 생성 (abuse_2=0/2)
    - df_join_v1    : 입력 df_join_v1에 abuse_2 컬럼을 '추가/갱신'(반환)
    - df_list_v1    : 입력 df_list_v1에 광고별 어뷰징 비율 abuse_2(0~1) '추가/갱신'(반환)
    """
    key_name = "ads_idx" if REJOIN_UNIT == "ads_idx" else "ad_code_key"

    # 1) df_settle_v1 생성 (정산기준 라벨)
    s_settle = _prepare_settle_for_label(df_list_v1, df_settle, df_join_v1)
    s_settle["abuse_2"] = np.int8(0)

    if len(rejoin_violation):
        v1 = rejoin_violation[["user_id", key_name]].drop_duplicates().rename(columns={key_name: "grp"})
        s_settle = s_settle.merge(v1.assign(_a2=np.int8(2)), on=["user_id","grp"], how="left")
        s_settle["abuse_2"] = s_settle[["abuse_2","_a2"]].max(axis=1).fillna(0).astype("int8")
        s_settle.drop(columns=["_a2"], inplace=True)

    if len(daily_dup):
        v2 = daily_dup[["user_id", key_name, "click_day"]].drop_duplicates().rename(columns={key_name: "grp"})
        s_settle = s_settle.merge(v2.assign(_b2=np.int8(2)), on=["user_id","grp","click_day"], how="left")
        s_settle["abuse_2"] = s_settle[["abuse_2","_b2"]].max(axis=1).fillna(0).astype("int8")
        s_settle.drop(columns=["_b2"], inplace=True)

    df_settle_v1 = s_settle  # 원본 스키마 유지가 필요하면 reindex로 맞춰도 됨

    # 2) df_join_v1에 abuse_2 추가 (원시참여도 동일 조건으로 색칠)
    j_norm = _prepare_join_for_label_v1(df_list_v1, df_join_v1, df_settle)
    j_norm["abuse_2"] = np.int8(0)

    if len(rejoin_violation):
        j_norm = j_norm.merge(v1.assign(_a2=np.int8(2)), on=["user_id","grp"], how="left")
        j_norm["abuse_2"] = j_norm[["abuse_2","_a2"]].max(axis=1).fillna(0).astype("int8")
        j_norm.drop(columns=["_a2"], inplace=True)

    if len(daily_dup):
        j_norm = j_norm.merge(v2.assign(_b2=np.int8(2)), on=["user_id","grp","click_day"], how="left")
        j_norm["abuse_2"] = j_norm[["abuse_2","_b2"]].max(axis=1).fillna(0).astype("int8")
        j_norm.drop(columns=["_b2"], inplace=True)

    # df_join_v1 스키마 유지 + abuse_2 붙이기 (index 정렬 보호를 위해 merge 사용)
    df_join_v1 = df_join_v1.merge(
        j_norm[["click_key","abuse_2"]],
        on="click_key", how="left", suffixes=("","")
    )
    df_join_v1["abuse_2"] = df_join_v1["abuse_2"].fillna(0).astype("int8")

    # 3) df_list_v1에 광고별 어뷰징 비율(정산기준)
    base = s_settle.copy()
    base["is_abuse"] = (base["abuse_2"] > 0).astype("int8")
    rate = base.groupby("grp", observed=True)["is_abuse"].mean().rename("abuse_2").astype("float32")

    if REJOIN_UNIT == "ads_idx":
        df_list_v1 = df_list_v1.merge(rate.rename_axis("ads_idx").reset_index(), on="ads_idx", how="left")
    else:
        df_list_v1 = df_list_v1.merge(rate.rename_axis("ad_code_key").reset_index(),
                                      left_on="ads_code", right_on="ad_code_key", how="left")
        df_list_v1.drop(columns=["ad_code_key"], inplace=True, errors="ignore")

    df_list_v1["abuse_2"] = df_list_v1["abuse_2"].fillna(0).astype("float32")

    return df_settle_v1, df_join_v1, df_list_v1

# abuse_2 추가
df_settle_v1, df_join_v1, df_list_v1 = update_abuse2_with_v1(
    df_list_v1, df_join_v1, df_settle, rejoin_violation, daily_dup
)

# 저장
df_join_v1.to_csv("df_join_v1.csv", index=False, encoding="utf-8-sig")
df_list_v1.to_csv("df_list_v1.csv", index=False, encoding="utf-8-sig")
df_settle_v1.to_csv("df_settle_v1.csv", index=False, encoding="utf-8-sig")