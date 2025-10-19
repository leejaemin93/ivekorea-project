# ===== CTIT 탐지 상단 파라미터(최소) =====
MIN_N_PER_DAY    = 30           # 광고×날짜(또는 퍼블리셔×날짜) 최소 표본 수
LONG_SEC_DEFAULT = 24 * 3600    # '과도하게 김' 기준(초)
LONG_SHARE_TH    = 0.20         # '김' 비율 임계 (ex. 20%)

# 타입/카테고리별 '너무 빠름'(초) 기준
SHORT_SEC_BY_TYPE = {1: 15, 2: 10, 3: 3, 4: 2}   # 1:설치, 2:실행, 3:참여, 4:클릭
GAME_CATS   = {2, 5, 6}                           # 게임형(CPI/CPA/멀티보상)
FAST_CATS   = {4, 13, 8, 10}                      # 퀴즈/간편/무료/유료 참여
FINANCE_CATS= {7}
SHORT_SEC_BY_CAT  = {
    **{c: 30 for c in GAME_CATS},
    **{c: 3  for c in FAST_CATS},
    **{c: 15 for c in FINANCE_CATS},
}

# 타입별 '너무 빠름' 비율 임계
SHORT_SHARE_TH_BY_TYPE = {1: 0.60, 2: 0.60, 3: 0.80, 4: 0.90}

# 퍼블리셔 쏠림 시 확정 승격 조건
PUB_TOP_SHORT_SH_TH = 0.80  # top 퍼블리셔의 '짧음' 비율
PUB_TOP_CONTRIB_TH  = 0.50  # 해당 퍼블리셔 기여도(그 날 점유)

# 헬퍼: CTIT 단위 자동 정규화(초)
def _autoscale_ctit_seconds(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").where(lambda x: x >= 0)
    cand = [1.0, 1e3, 1e6, 1e9]  # sec, ms, μs, ns 가정
    def _score(arr):
        a = arr.dropna()
        if a.empty: return np.inf
        r_long = (a >= LONG_SEC_DEFAULT).mean()
        med, p95 = a.median(), a.quantile(0.95)
        pen = 0.0
        if not (1 <= med <= 6*3600): pen += 1.0
        if not (3 <= p95 <= 7*24*3600): pen += 1.0
        return r_long * 100 + pen
    best_div = min(cand, key=lambda d: _score(s / d))
    out = (s / best_div).clip(lower=0, upper=7*24*3600)  # 최대 7일
    return out

# 헬퍼: 행 단위 '짧음' 기준(초)
def _ctit_short_sec_for_rows(df: pd.DataFrame) -> pd.Series:
    sec = pd.Series(np.nan, index=df.index, dtype=float)
    if "ads_category" in df.columns:
        sec = df["ads_category"].map(SHORT_SEC_BY_CAT)
    if "ads_type" in df.columns:
        sec = sec.fillna(df["ads_type"].map(SHORT_SEC_BY_TYPE))
    return sec.fillna(10.0)

# 메인: CTIT 의심/확정만 반환(정상 제외)
def detect_ctit_severity(
    df_settle: pd.DataFrame,
    df_list:   pd.DataFrame,
    by: str = "ad_date",             # "ad_date" | "publisher_date"
    *,
    device_only: bool = False,
    sus_mult: float = 0.85,
    conf_long_mult: float = 1.5
) -> pd.DataFrame:
    # ---- 메타 최소 ----
    meta_cols = [c for c in ["ads_idx","ads_type","ads_category","ads_code","ads_name"] if c in df_list.columns]
    meta = df_list[meta_cols].drop_duplicates("ads_idx")

    # ---- 원본/전처리 ----
    need_cols = ["mda_idx","ads_idx","pub_sub_rel_id","click_key","click_date","ctit","dvc_idx"]  # ← mda_idx 추가
    st = df_settle[[c for c in need_cols if c in df_settle.columns]].copy()
    st["dt"]      = pd.to_datetime(st["click_date"], errors="coerce")
    st            = st[st["dt"].notna()].copy()
    st["date"]    = st["dt"].dt.date
    st["mda_idx"] = pd.to_numeric(st.get("mda_idx"), errors="coerce")
    st["dvc_idx"] = pd.to_numeric(st.get("dvc_idx", 0), errors="coerce").fillna(0).astype("int64")
    if device_only:
        st = st[st["dvc_idx"] != 0]
    # mda_idx 없는 행 제거
    st = st.dropna(subset=["mda_idx"]).copy()
    if st.empty:
        base_cols = ["mda_idx","ads_idx","date","n","short_sh","long_sh","severity"]
        if by=="publisher_date": base_cols.insert(3,"pub_sub_rel_id")
        return pd.DataFrame(columns=base_cols)

    # ---- CTIT(초) 정규화 + 라벨 원재료 ----
    st["ctit"]      = _autoscale_ctit_seconds(st["ctit"])
    st              = st.merge(meta, on="ads_idx", how="left", validate="m:1")
    st["short_sec"] = _ctit_short_sec_for_rows(st)
    st["is_short"]  = st["ctit"].le(st["short_sec"])
    st["is_long"]   = st["ctit"].ge(LONG_SEC_DEFAULT)

    # ---- 집계키 (mda_idx 포함) ----
    if by == "ad_date":
        keys = ["mda_idx","ads_idx","date"]
    else:  # publisher_date
        keys = ["mda_idx","ads_idx","date","pub_sub_rel_id"]

    g = (st.groupby(keys, as_index=False)
           .agg(n=("click_key","nunique"),
                short_n=("is_short","sum"),
                long_n=("is_long","sum")))

    # 비율 + 타입별 기준
    g = g.merge(meta[["ads_idx","ads_type"]], on="ads_idx", how="left")
    g["short_sh"] = np.where(g["n"]>0, g["short_n"]/g["n"], np.nan)
    g["long_sh"]  = np.where(g["n"]>0, g["long_n"]/g["n"], np.nan)
    g["short_th"] = g["ads_type"].map(SHORT_SHARE_TH_BY_TYPE).fillna(0.8)
    long_th       = LONG_SHARE_TH

    base_ok   = g["n"] >= MIN_N_PER_DAY
    short_hit = base_ok & (g["short_sh"] >= g["short_th"])
    long_hit  = base_ok & (g["long_sh"]  >= long_th)
    short_near= base_ok & (g["short_sh"] >= g["short_th"]*sus_mult)
    long_near = base_ok & (g["long_sh"]  >= long_th*sus_mult)

    # 퍼블리셔 쏠림(광고×날짜에서만) — mda_idx 포함해서 계산
    if by == "ad_date":
        sp = (st.groupby(["mda_idx","ads_idx","date","pub_sub_rel_id"], as_index=False)
                .agg(n=("click_key","nunique"),
                     short_n=("is_short","sum")))
        if len(sp):
            sp["short_sh"] = np.where(sp["n"]>0, sp["short_n"]/sp["n"], np.nan)
            tot = (sp.groupby(["mda_idx","ads_idx","date"], as_index=False)["n"]
                      .sum().rename(columns={"n":"n_total"}))
            top = (sp.merge(tot, on=["mda_idx","ads_idx","date"])
                     .sort_values(["mda_idx","ads_idx","date","short_sh"], ascending=[True,True,True,False])
                     .drop_duplicates(["mda_idx","ads_idx","date"]))
            top["pub_top_contrib"] = np.where(top["n_total"]>0, top["n"]/top["n_total"], np.nan)
            g = g.merge(
                top[["mda_idx","ads_idx","date","short_sh","pub_top_contrib"]]
                    .rename(columns={"short_sh":"pub_top_short_sh"}),
                on=["mda_idx","ads_idx","date"], how="left"
            )
        else:
            g["pub_top_short_sh"] = np.nan
            g["pub_top_contrib"]  = np.nan
    else:
        g["pub_top_short_sh"] = np.nan
        g["pub_top_contrib"]  = np.nan

    # ---- severity 분류(정상 제외) ----
    conf = (short_hit & long_hit) \
           | (short_hit & (g["pub_top_short_sh"].fillna(0)>=PUB_TOP_SHORT_SH_TH)
                       & (g["pub_top_contrib"].fillna(0) >=PUB_TOP_CONTRIB_TH)) \
           | (base_ok & (g["long_sh"] >= long_th*conf_long_mult))
    sus  = (~conf) & (short_near | long_near)

    g["severity"] = np.select([conf, sus], ["확정","의심"], default="정상")
    g = g[g["severity"]!="정상"].copy()
    g["severity"] = pd.Categorical(g["severity"], categories=["의심","확정"], ordered=True)

    # ✅ ads_code/ads_name 붙이기(있을 때만)
    add_meta = [c for c in ["ads_code","ads_name"] if c in meta.columns]
    if add_meta:
        g = g.merge(meta[["ads_idx"]+add_meta], on="ads_idx", how="left")

    # ---- 출력 슬림(존재하는 컬럼만 안전 선택) ----
    preferred = ["mda_idx","ads_idx"] + [c for c in ["ads_code","ads_name"] if c in g.columns]
    if by=="publisher_date":
        preferred += ["pub_sub_rel_id"]
    preferred += ["date","n","short_sh","long_sh","severity"]

    out_cols = [c for c in preferred if c in g.columns]
    out = g[out_cols].sort_values(
        ["severity","mda_idx","ads_idx","short_sh","long_sh"] if {"short_sh","long_sh"}.issubset(g.columns)
        else ["severity","mda_idx","ads_idx"],
        ascending=[True,True,True,False,False] if {"short_sh","long_sh"}.issubset(g.columns) else [True,True,True]
    ).reset_index(drop=True)
    return out

# 결과
# 광고×날짜 (퍼블리셔 쏠림 승격 로직 포함)
ctit_ad_flags = detect_ctit_severity(
    df_settle, df_list, by="ad_date",
    device_only=False,      # 웹 포함하려면 False, 웹 제외하려면 True
    sus_mult=0.85,          # 의심선(낮출수록 의심↑)  예: 0.80 ~ 0.95
    conf_long_mult=1.5      # 긴 쪽 확정선(낮출수록 확정↑) 예: 1.3 ~ 2.0
)

# 퍼블리셔×날짜
ctit_pub_flags = detect_ctit_severity(
    df_settle, df_list, by="publisher_date",
    sus_mult=0.85, conf_long_mult=1.5
)

SEV2NUM = {"정상":0, "의심":1, "확정":2}

def _prep_flags_unique(flag_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    detect_ctit_severity 출력표를 병합-안전형으로 축약:
    - date 통일
    - severity -> sev_code(Int8)
    - 키별 max(sev_code)로 1행화 (중복 제거)
    - 반환 키:
        ad_date:        [mda_idx, ads_idx, date, sev_code]
        publisher_date: [mda_idx, ads_idx, date, pub_sub_rel_id, sev_code]
    """
    if flag_df is None or flag_df.empty:
        return pd.DataFrame()

    f = flag_df.copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.date

    if by == "ad_date":
        keys = ["mda_idx","ads_idx","date"]
    else:
        keys = ["mda_idx","ads_idx","date","pub_sub_rel_id"]

    keep = [c for c in keys + ["severity"] if c in f.columns]
    f = f[keep].copy()

    # 타입 정규화
    for col in [c for c in ["mda_idx","ads_idx","pub_sub_rel_id"] if c in f.columns]:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    f["sev_code"] = f["severity"].map(SEV2NUM).astype("Int8")

    # 키별 1행화 (최댓값)
    f = (f.groupby(keys, observed=True, as_index=False)["sev_code"].max())
    return f


def enrich_settle_with_ctit_flags(df_settle, ctit_ad_flags, ctit_pub_flags, *, strict=True):
    """
    df_settle에 abuse_6(0/1/2) '하나'만 추가. 행 수는 반드시 보존.
    - ad_date, publisher_date 플래그는 키별 유일화 후 병합(m:1)
    - 내부 보조 컬럼은 모두 제거
    """
    s = df_settle.copy()
    n0 = len(s)

    # 키/날짜 정규화
    s["dt"]   = pd.to_datetime(s["click_date"], errors="coerce")
    s["date"] = s["dt"].dt.date
    for col in [c for c in ["mda_idx","ads_idx","pub_sub_rel_id"] if c in s.columns]:
        s[col] = pd.to_numeric(s[col], errors="coerce")

    # 1) 광고×날짜 플래그 (mda 포함)
    ad_u = _prep_flags_unique(ctit_ad_flags, by="ad_date")
    if not ad_u.empty:
        try:
            s = s.merge(ad_u.rename(columns={"sev_code":"_ad_sev"}),
                        on=[c for c in ["mda_idx","ads_idx","date"] if c in s.columns],
                        how="left",
                        validate="m:1")
        except Exception as e:
            if strict:
                raise
            # 혹시 모를 중복을 한 번 더 방어
            ad_u = (ad_u.groupby([c for c in ad_u.columns if c not in ["sev_code"]], observed=True)
                        .agg(sev_code=("sev_code","max")).reset_index()) \
                   .rename(columns={"sev_code":"_ad_sev"})
            s = s.merge(ad_u, on=[c for c in ["mda_idx","ads_idx","date"] if c in s.columns], how="left")
    else:
        s["_ad_sev"] = pd.Series(pd.NA, index=s.index, dtype="Int8")

    # 2) 퍼블리셔×날짜 플래그 (mda 포함)
    pub_u = _prep_flags_unique(ctit_pub_flags, by="publisher_date")
    if not pub_u.empty:
        keys = [c for c in ["mda_idx","ads_idx","date","pub_sub_rel_id"] if c in s.columns]
        try:
            s = s.merge(pub_u.rename(columns={"sev_code":"_pub_sev"}),
                        on=keys, how="left", validate="m:1")
        except Exception as e:
            if strict:
                raise
            pub_u = (pub_u.groupby(keys, observed=True)
                        .agg(sev_code=("sev_code","max")).reset_index()) \
                   .rename(columns={"sev_code":"_pub_sev"})
            s = s.merge(pub_u, on=keys, how="left")
    else:
        s["_pub_sev"] = pd.Series(pd.NA, index=s.index, dtype="Int8")

    # 3) 최종 abuse_6 = max(ad, pub)  (결측→0)
    s["_ad_sev"]  = pd.to_numeric(s["_ad_sev"], errors="coerce").fillna(0).astype("int8")
    s["_pub_sev"] = pd.to_numeric(s["_pub_sev"], errors="coerce").fillna(0).astype("int8")
    s["abuse_6"]  = s[["_ad_sev","_pub_sev"]].max(axis=1).astype("int8")

    # 보조 컬럼 제거 → abuse_6 하나만 남김
    s.drop(columns=["dt","date","_ad_sev","_pub_sev"], inplace=True, errors="ignore")

    # 행수 검증
    n1 = len(s)
    if n1 != n0:
        raise RuntimeError(f"행수 불일치: before={n0}, after={n1}. (우측 테이블 중복 키로 인한 확장 가능성)")

    # 새 컬럼 확인 (디버그 원하면 주석 해제)
    # new_cols = [c for c in s.columns if c not in df_settle.columns or c=="abuse_6"]
    # print({"new_columns": new_cols})

    return s

def build_list_with_ctit_rate(df_list: pd.DataFrame, df_settle_v1: pd.DataFrame) -> pd.DataFrame:
    """
    df_list에 광고별 'CTIT 어뷰징 전환 비율(0~1)'을 abuse_6 컬럼으로 추가.
    - 분자: df_settle_v1에서 abuse_6 > 0 인 건수
    - 분모: df_settle_v1 전체 건수
    """
    # 안전 변환
    tmp = df_settle_v1[["ads_idx","abuse_6"]].copy()
    tmp["ads_idx"] = pd.to_numeric(tmp["ads_idx"], errors="coerce")
    tmp["is_abuse"] = (pd.to_numeric(tmp["abuse_6"], errors="coerce").fillna(0) > 0).astype("int8")

    # 광고별 오염도(비율)
    rate = (tmp.groupby("ads_idx", observed=True)["is_abuse"]
               .mean()
               .rename("abuse_6")
               .reset_index())

    # df_list에도 ads_idx 정규화
    out = df_list.copy()
    out["ads_idx"] = pd.to_numeric(out["ads_idx"], errors="coerce")

    # m:1 병합 보장
    rate = rate.drop_duplicates(subset=["ads_idx"])
    out = out.merge(rate, on="ads_idx", how="left", validate="m:1")

    # 결측은 0
    out["abuse_6"] = out["abuse_6"].fillna(0.0)

    return out

# ===== 3) df_settle 라벨 전파 (행수 보존 + abuse_6 하나만 추가) =====
n0 = len(df_settle_v1)
df_settle_v1 = enrich_settle_with_ctit_flags(df_settle_v1, ctit_ad_flags, ctit_pub_flags)  # 행수 동일 유지
assert len(df_settle_v1) == n0, "행수 변하면 안 됨!"
assert "abuse_6" in df_settle_v1.columns and df_settle_v1.filter(regex="^_ad_sev|_pub_sev$").empty, "abuse_6만 남아야 함"

# ===== 4) df_list 오염도(%) 계산 =====
df_list_v1 = build_list_with_ctit_rate(df_list_v1, df_settle_v1)

# 확인 후 저장
df_settle_v1.to_csv("df_settle_v1.csv", index=False)
df_list_v1.to_csv("df_list_v1.csv", index=False)