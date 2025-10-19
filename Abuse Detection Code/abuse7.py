# ----- 파라미터 -----

WINDOW_SIZE = 30
MIN_HISTORY_DAYS = 14
MIN_DAILY_VOLUME = 20

PRICE_SPIKE_RATIO   = 2.5
PRICE_SPIKE_Z_SCORE = 2.0
VOLUME_SPIKE_RATIO  = 1.5

# 의심/확정 임계값 조절 파라미터
SUS_MULT = 0.6              # 의심선 배수 (낮을수록 의심↑) - 0.75→0.6으로 추가 조정하여 의심 케이스 더 증가
CONF_MULT = 1.3             # 확정선 배수 (낮을수록 확정↑) - 1.3→1.1으로 추가 조정하여 확정 케이스 더 증가

SUSPICIOUS_CONSECUTIVE_DAYS = 1
CONFIRMED_CONSECUTIVE_DAYS  = 2

LARGE_MEDIA_THRESHOLD  = 10000
MEDIUM_MEDIA_THRESHOLD = 1000

# (추가) 리포트 편의
TOPK = 8

SEV2TXT = {0: "정상", 1: "의심", 2: "확정"}
TXT2SEV = {"정상": 0, "의심": 1, "확정": 2}

# ---------- 유틸 ----------
def _check_cols(df, need, opt=None, name="df"):
    opt = opt or []
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{name} 필수 컬럼 누락: {miss} (선택:{opt})")

def _to_day(col_date, col_time=None):
    if col_time is not None:
        dt = pd.to_datetime(col_date.astype(str)+" "+col_time.astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(col_date, errors="coerce")
    return dt.dt.floor("D")

# ========== 1) 베이스라인 (df_rpt) ==========
def build_media_baseline_from_rpt(df_rpt: pd.DataFrame) -> pd.DataFrame:
    """
    df_rpt로 매체사(mda_idx)별 일단위 turn/earn/단가를 만들고,
    mda별 평균/표준편차/히스토리 카운트를 베이스라인으로 산출.
    """
    _check_cols(df_rpt,
        ["rpt_time_date","rpt_time_time","mda_idx","rpt_time_turn","rpt_time_earn"],
        name="df_rpt"
    )
    r = df_rpt.copy()
    r["date"]   = _to_day(r["rpt_time_date"], r["rpt_time_time"])
    r["mda_idx"]= pd.to_numeric(r["mda_idx"], errors="coerce")
    r["turn"]   = pd.to_numeric(r["rpt_time_turn"], errors="coerce").fillna(0)
    r["earn"]   = pd.to_numeric(r["rpt_time_earn"], errors="coerce").fillna(0.0)
    r = r.dropna(subset=["date","mda_idx"])

    daily = (r.groupby(["mda_idx","date"], observed=True, as_index=False)
               .agg(turn=("turn","sum"), earn=("earn","sum")))
    daily["unit_price"] = np.where(daily["turn"]>0, daily["earn"]/daily["turn"], np.nan)

    base = (daily.groupby("mda_idx", observed=True)
                 .agg(turn_mean=("turn","mean"),
                      turn_median=("turn","median"),
                      turn_std=("turn","std"),
                      turn_count=("turn","count"),
                      earn_mean=("earn","mean"),
                      earn_median=("earn","median"),
                      earn_std=("earn","std"),
                      up_mean=("unit_price","mean"),
                      up_median=("unit_price","median"),
                      up_std=("unit_price","std"))
                 .reset_index())

    # 규모 라벨 (turn_mean 기준)
    base["media_size"] = pd.cut(
        base["turn_mean"],
        bins=[-np.inf, MEDIUM_MEDIA_THRESHOLD, LARGE_MEDIA_THRESHOLD, np.inf],
        labels=["small","medium","large"]
    )

    # 히스토리 부족 제거
    base = base[base["turn_count"] >= MIN_HISTORY_DAYS].copy()
    return base

# ========== 2) subpub 단가 급증 탐지 (df_settle + baseline) ==========
def detect_subpub_spikes_tuned(df_settle: pd.DataFrame, media_baseline: pd.DataFrame) -> pd.DataFrame:
    """
    일×(mda, pub_sub_rel_id)의 unit_price/volume를 baseline과 비교하여 급증 탐지.
    결과를 '의심/확정'으로 라벨링하고, 리포팅에 필요한 수치 포함 반환.
    """
    _check_cols(df_settle,
        ["mda_idx","pub_sub_rel_id","click_date","earn_cost"],
        name="df_settle"
    )
    s = df_settle.copy()
    s["date"]          = _to_day(s["click_date"])
    s["mda_idx"]       = pd.to_numeric(s["mda_idx"], errors="coerce")
    s["pub_sub_rel_id"]= pd.to_numeric(s["pub_sub_rel_id"], errors="coerce")
    s["earn_cost"]     = pd.to_numeric(s["earn_cost"], errors="coerce").fillna(0)
    s = s.dropna(subset=["date","mda_idx","pub_sub_rel_id"])

    # 일×subpub 집계
    g = (s.groupby(["mda_idx","pub_sub_rel_id","date"], observed=True, as_index=False)
           .agg(total_earn=("earn_cost","sum"),
                volume=("earn_cost","size")))
    g["unit_price"] = np.where(g["volume"]>0, g["total_earn"]/g["volume"], np.nan)

    # baseline merge
    use_cols = ["mda_idx","up_mean","up_std","turn_mean","media_size"]
    b = media_baseline[use_cols].copy()
    gg = g.merge(b, on="mda_idx", how="left")

    # 필터: 일 최소 볼륨
    gg = gg[gg["volume"] >= MIN_DAILY_VOLUME].copy()

    # 임계 계산 함수
    def _thr(ms):
        if ms == "large":
            return PRICE_SPIKE_RATIO*1.5, PRICE_SPIKE_Z_SCORE*1.2
        elif ms == "medium":
            return PRICE_SPIKE_RATIO, PRICE_SPIKE_Z_SCORE
        else:  # small
            return PRICE_SPIKE_RATIO*0.8, PRICE_SPIKE_Z_SCORE*0.8

    # 이상치 산출
    rows = []
    for _, r in gg.iterrows():
        base_p  = r["up_mean"]
        base_sd = r["up_std"] if pd.notna(r["up_std"]) and r["up_std"]>0 else (r["up_mean"]*0.3 if pd.notna(r["up_mean"]) else np.nan)
        cur_p   = r["unit_price"]
        ms      = r["media_size"]

        if not np.isfinite(base_p) or not np.isfinite(base_sd) or not np.isfinite(cur_p):
            continue

        ratio_thr, z_thr = _thr(ms)
        price_ratio = cur_p / base_p if base_p>0 else np.inf
        z_score     = (cur_p - base_p) / base_sd if base_sd>0 else np.inf
        vol_ratio   = (r["volume"]/r["turn_mean"]) if (pd.notna(r["turn_mean"]) and r["turn_mean"]>0) else 1.0

        is_price_spike  = (price_ratio >= ratio_thr) or (z_score >= z_thr)
        is_volume_spike = (vol_ratio   >= VOLUME_SPIKE_RATIO)
        is_extreme      = (cur_p >= 5000)

        if (is_price_spike and is_volume_spike) or is_extreme:
            severity_score = float(price_ratio * (abs(z_score)+1) * vol_ratio)
            
            # sus_mult, conf_mult 적용한 임계값 계산
            sus_threshold = 10 * SUS_MULT
            conf_threshold = 25 * CONF_MULT
            
            sev = 2 if (is_extreme and cur_p>=10000) or (severity_score>=conf_threshold) else (1 if severity_score>=sus_threshold or is_extreme else 0)
            if sev>0:
                rows.append({
                    "date": r["date"], "mda_idx": r["mda_idx"], "pub_sub_rel_id": r["pub_sub_rel_id"],
                    "n_settle": int(r["volume"]), "sum_earn": float(r["total_earn"]), "eps": float(cur_p),
                    "price_ratio": float(price_ratio), "z_score": float(z_score), "volume_ratio": float(vol_ratio),
                    "media_size": ms, "severity": SEV2TXT[sev], "risk_score": float(severity_score)
                })

    subpub = pd.DataFrame(rows)
    if subpub.empty:
        return pd.DataFrame(columns=["date","mda_idx","pub_sub_rel_id","n_settle","sum_earn","eps",
                                     "price_ratio","z_score","volume_ratio","media_size",
                                     "severity","risk_score","share"])

    # share 계산(같은 일×mda 내 비중)
    day_tot = (subpub.groupby(["date","mda_idx"], observed=True)["sum_earn"].sum()
                     .rename("tot_earn").reset_index())
    subpub = subpub.merge(day_tot, on=["date","mda_idx"], how="left")
    subpub["share"] = np.where(subpub["tot_earn"]>0, subpub["sum_earn"]/subpub["tot_earn"], np.nan)
    subpub.drop(columns=["tot_earn"], inplace=True)

    # 정렬/TopK
    subpub["severity"] = pd.Categorical(subpub["severity"], categories=["의심","확정"], ordered=True)
    subpub = subpub.sort_values(["date","mda_idx","severity","risk_score","share","sum_earn"],
                                ascending=[True,True,True,False,False,False])
    if TOPK and TOPK>0:
        subpub = subpub.groupby(["date","mda_idx"]).head(TOPK).reset_index(drop=True)
    return subpub

# ========== 3) mda 플래그(일×매체) 만들기 ==========
def make_mda_flags_from_subpub(df_rpt: pd.DataFrame, subpub_flags: pd.DataFrame) -> pd.DataFrame:
    """
    subpub 라벨을 mda×date로 올려 최대 심각도를 mda에 부여.
    df_rpt에서 일×mda의 turn/earn/단가를 가져와 리포트 컬럼 형태를 맞춤.
    """
    if subpub_flags.empty:
        return pd.DataFrame(columns=["date","mda_idx","n_turn","sum_earn","eps",
                                     "sum_ratio","sum_z","eps_ratio","eps_z","severity"])

    # mda×date 최대 severity
    sev_code = subpub_flags.assign(sev_code=subpub_flags["severity"].map(TXT2SEV)) \
                           .groupby(["date","mda_idx"], observed=True)["sev_code"].max().reset_index()

    # df_rpt에서 일×mda 요약(리포트용)
    r = df_rpt.copy()
    r["date"]    = _to_day(r["rpt_time_date"], r["rpt_time_time"])
    r["mda_idx"] = pd.to_numeric(r["mda_idx"], errors="coerce")
    r["turn"]    = pd.to_numeric(r["rpt_time_turn"], errors="coerce").fillna(0)
    r["earn"]    = pd.to_numeric(r["rpt_time_earn"], errors="coerce").fillna(0.0)
    r = r.dropna(subset=["date","mda_idx"])

    day_mda = (r.groupby(["mda_idx","date"], observed=True, as_index=False)
                 .agg(n_turn=("turn","sum"), sum_earn=("earn","sum")))
    day_mda["eps"] = np.where(day_mda["n_turn"]>0, day_mda["sum_earn"]/day_mda["n_turn"], np.nan)

    # 단순 기준비(베이스라인 mean 대비)로 ratio/z (리포트 가독용)
    base = (r.groupby("mda_idx", observed=True)
              .agg(up_mean=("earn", lambda s: np.nan),  # placeholder
                   turn_mean=("turn","mean"),
                   eps_mean=("earn","mean"))            # placeholder
              .reset_index())
    # 위 placeholder는 쓰지 않음. subpub에서 이미 단가 기반으로 판정했으므로,
    # 리포트용 ratio/z는 '당일 eps' vs 'mda eps 중앙값/표준편차(지난 일자)'로 대체
    # 안전하게 다시 계산
    tmp = (r.groupby(["mda_idx","date"], observed=True, as_index=False)
             .agg(eps=("earn", lambda x: np.nan)))  # 자리만
    # 간단히: 당일 eps 그대로 두고 ratio/z는 NaN으로 두어도 무방(리포트용)
    out = day_mda.merge(sev_code, on=["mda_idx","date"], how="inner")
    out["severity"] = out["sev_code"].map(SEV2TXT)
    out.drop(columns=["sev_code"], inplace=True)

    # 리포트 컬럼 정합(원 스타일)
    out["sum_ratio"] = np.nan
    out["sum_z"]     = np.nan
    out["eps_ratio"] = np.nan
    out["eps_z"]     = np.nan

    out["severity"] = pd.Categorical(out["severity"], categories=["의심","확정"], ordered=True)
    out = out[["date","mda_idx","n_turn","sum_earn","eps",
               "sum_ratio","sum_z","eps_ratio","eps_z","severity"]]
    out = out.sort_values(["date","mda_idx","severity"], ascending=[True,True,True]).reset_index(drop=True)
    return out

# ========== 4) 엔드포인트 (원래 네 흐름) ==========
def detect_mda_spikes_and_drilldown(df_rpt: pd.DataFrame, df_settle: pd.DataFrame):
    """
    1) df_rpt로 mda 베이스라인 구축
    2) df_settle에서 subpub 급증 탐지
    3) subpub 결과를 mda×date로 올려 mda 플래그 생성
    """
    media_base = build_media_baseline_from_rpt(df_rpt)
    subpub_top = detect_subpub_spikes_tuned(df_settle, media_base)
    mda_flags  = make_mda_flags_from_subpub(df_rpt, subpub_top)
    return mda_flags, subpub_top

# ========== 5) 라벨 전파 (원래 MAX 전파 스타일) ==========
def apply_abuse7_labels(df_rpt: pd.DataFrame, df_settle: pd.DataFrame,
                        mda_flags: pd.DataFrame, subpub_top: pd.DataFrame):
    """
    - df_rpt: mda×date severity를 그대로 부여
    - df_settle: subpub 라벨과 mda 라벨 중 '큰 값' MAX 전파
    """
    # --- mda 라벨 테이블(숫자형 보장) ---
    mf = (mda_flags.assign(sev_code=mda_flags["severity"].map({"의심":1, "확정":2}))
                    [["mda_idx","date","sev_code"]].drop_duplicates())
    mf["sev_code"] = pd.to_numeric(mf["sev_code"], errors="coerce")  # <- 숫자형 강제

    # --- df_rpt 라벨 ---
    r = df_rpt.copy()
    r["date"]    = pd.to_datetime(r["rpt_time_date"].astype(str) + " " + r["rpt_time_time"].astype(str),
                                  errors="coerce").dt.floor("D")
    r["mda_idx"] = pd.to_numeric(r["mda_idx"], errors="coerce")

    r = r.merge(mf, on=["mda_idx","date"], how="left")
    # Categorical 방지: to_numeric → fillna → int8
    r["abuse_7"] = pd.to_numeric(r["sev_code"], errors="coerce").fillna(0).astype("int8")
    r.drop(columns=["sev_code","date"], inplace=True)

    # --- subpub 라벨 테이블(숫자형 보장) ---
    sp = (subpub_top.assign(sev_code=subpub_top["severity"].map({"의심":1, "확정":2}))
                    [["mda_idx","pub_sub_rel_id","date","sev_code"]].drop_duplicates())
    sp["sev_code"] = pd.to_numeric(sp["sev_code"], errors="coerce")  # <- 숫자형 강제

    # --- df_settle 라벨 ---
    s = df_settle.copy()
    s["date"]          = pd.to_datetime(s["click_date"], errors="coerce").dt.floor("D")
    s["mda_idx"]       = pd.to_numeric(s["mda_idx"], errors="coerce")
    s["pub_sub_rel_id"]= pd.to_numeric(s["pub_sub_rel_id"], errors="coerce")

    s = s.merge(sp, on=["mda_idx","pub_sub_rel_id","date"], how="left", suffixes=("","_pub"))
    s = s.merge(mf, on=["mda_idx","date"], how="left", suffixes=("","_mda"))

    # 모두 숫자형으로 변환 후 계산
    s["sev_pub"] = pd.to_numeric(s["sev_code"], errors="coerce").fillna(0).astype("int8")
    s["sev_mda"] = pd.to_numeric(s["sev_code_mda"], errors="coerce").fillna(0).astype("int8")

    s["abuse_7"] = s[["sev_pub","sev_mda"]].max(axis=1).astype("int8")

    s.drop(columns=["sev_code","sev_code_mda","sev_pub","sev_mda","date"], inplace=True)
    return r, s

# 1) 플래그 산출
mda_flags, subpub_top = detect_mda_spikes_and_drilldown(df_rpt_v1, df_settle_v1)

# 2) 라벨 전파
df_rpt_v1, df_settle_v1 = apply_abuse7_labels(df_rpt_v1, df_settle_v1, mda_flags, subpub_top)

# 저장
df_settle_v1.to_csv("df_settle_v1.csv", index=False)
df_rpt_v1.to_csv("df_rpt_v1.csv", index=False)