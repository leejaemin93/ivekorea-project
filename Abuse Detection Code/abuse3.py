# ================== 파라미터 ==================
# 공통(롤링 기준)
ROLL_WIN = 12        # 기준 구간(행) = 24시간 가정
MIN_P    = 6        # 최소 유효 샘플 수
MULT_TH  = 3.0       # 중앙값 대비 배수 임계(하드컷)
Z_TH     = 3.0       # z-score 임계(하드컷)
BASE_MIN = 1000       # 중앙값 최소(너무 작은 분모 방지)
WARMUP_H = 6         # (ad 전용) 캠페인 시작 후 워밍업 시간(시간)

# 설치/실행형만 평가(퀴즈·참여 등 빠른 타입은 폭증 허용 가능성↑)
INSTALL_TYPES = {1, 2}  # 1:설치형, 2:실행형

# 라벨 상수
LABEL_CONFIRM = "확정"
LABEL_SUSPECT = "의심"

# 소프트 컷(의심) 마진 — 하드컷보다 조금 느슨
MULT_SOFT = 1.0   # 의심: mult ≥ MULT_TH - MULT_SOFT
Z_SOFT    = 1.0   # 의심: z    ≥ Z_TH    - Z_SOFT

# 그룹별 Top-K(과다 탐지 방지)
TOPK_CONFIRM_AD = 3   # 광고단위 확정 최대 개수
TOPK_SUSPECT_AD = 5   # 광고단위 의심 최대 개수
TOPK_CONFIRM_PB = 3   # 퍼블리셔단위 확정 최대 개수
TOPK_SUSPECT_PB = 5   # 퍼블리셔단위 의심 최대 개수

# ================== 1) 광고 단위 클릭 폭증 ==================
def detect_click_spike_ad(df_rpt: pd.DataFrame, df_list: pd.DataFrame) -> pd.DataFrame:
    # --- 시계열 준비
    r = df_rpt.copy()
    r["rpt_dt"] = pd.to_datetime(
        r["rpt_time_date"].astype(str) + " " + r["rpt_time_time"].astype(str),
        errors="coerce"
    )
    r["rpt_time_clk"] = pd.to_numeric(r["rpt_time_clk"], errors="coerce").fillna(0)

    # --- 메타(타입/시작시간)
    meta = df_list[["ads_idx","ads_code","ads_name","ads_type","ads_sdate","ads_category"]].copy()
    meta["ads_sdate"] = pd.to_datetime(meta["ads_sdate"], errors="coerce")
    meta["is_install_like"] = meta["ads_type"].isin(INSTALL_TYPES)

    x = (r.merge(meta, on="ads_idx", how="left")
           .query("is_install_like == True")
           .sort_values(["ads_idx","rpt_dt"]))

    # --- 롤링 통계(ads_idx별)
    def _roll(gr):
        gr = gr.copy()
        gr["roll_med"] = gr["rpt_time_clk"].rolling(ROLL_WIN, min_periods=MIN_P).median()
        gr["roll_std"] = gr["rpt_time_clk"].rolling(ROLL_WIN, min_periods=MIN_P).std(ddof=0)
        return gr
    x = x.groupby("ads_idx", group_keys=False).apply(_roll)

    # --- 지표
    x["mult"]  = x["rpt_time_clk"] / x["roll_med"]
    x["z"]     = (x["rpt_time_clk"] - x["roll_med"]) / x["roll_std"]
    x["hours_since_start"] = (x["rpt_dt"] - x["ads_sdate"]).dt.total_seconds()/3600

    # --- 하드컷(확정) & 소프트컷(의심)
    base_guard = x["roll_med"].ge(BASE_MIN) & x["hours_since_start"].ge(WARMUP_H)
    hard = base_guard & x["mult"].ge(MULT_TH) & x["z"].ge(Z_TH)
    soft = base_guard & ~hard & (
        x["mult"].ge(MULT_TH - MULT_SOFT) | x["z"].ge(Z_TH - Z_SOFT)
    )

    cols = ["ads_idx","ads_code","ads_name","rpt_dt","rpt_time_clk","roll_med","mult","z"]
    cand_h = x.loc[hard, cols].copy(); cand_h["severity"] = LABEL_CONFIRM
    cand_s = x.loc[soft, cols].copy(); cand_s["severity"] = LABEL_SUSPECT

    out = pd.concat([cand_h, cand_s], ignore_index=True)
    if out.empty: 
        return out

    # --- 그룹별 Top-K 제한
    out = out.sort_values(["ads_idx","severity","mult","z","rpt_dt"],
                          ascending=[True, False, False, False, True]).reset_index(drop=True)
    rk = out.groupby(["ads_idx","severity"]).cumcount()
    out = out[
        ((out["severity"]==LABEL_CONFIRM) & (rk < TOPK_CONFIRM_AD)) |
        ((out["severity"]==LABEL_SUSPECT) & (rk < TOPK_SUSPECT_AD))
    ].copy()

    # 최종 정렬 및 출력
    out = out.sort_values(["ads_idx","rpt_dt"]).reset_index(drop=True)
    return out  # severity만 포함, abuse_type 제거

# ================== 2) 퍼블리셔 단위 클릭 폭증(+상위 mda 라벨) ==================
def detect_click_spike_publisher_labeled(df_join: pd.DataFrame) -> pd.DataFrame:
    j = df_join[["pub_sub_rel_id","mda_idx","click_date"]].copy()
    j["hour"] = pd.to_datetime(j["click_date"], errors="coerce").dt.floor("H")

    # --- 퍼블리셔×시간 집계
    grp = (j.groupby(["pub_sub_rel_id","hour"], as_index=False)
             .size().rename(columns={"size":"clk"})
             .sort_values(["pub_sub_rel_id","hour"]))

    # --- 롤링
    def _roll(g):
        g = g.copy()
        g["roll_med"] = g["clk"].rolling(ROLL_WIN, min_periods=MIN_P).median()
        g["roll_std"] = g["clk"].rolling(ROLL_WIN, min_periods=MIN_P).std(ddof=0)
        g["mult"]     = g["clk"] / g["roll_med"]
        g["z"]        = (g["clk"] - g["roll_med"]) / g["roll_std"]
        return g
    y = grp.groupby("pub_sub_rel_id", group_keys=False).apply(_roll)

    # --- 하드/소프트 컷 (퍼블리셔는 워밍업 미적용)
    base_guard = y["roll_med"].ge(BASE_MIN)
    hard = base_guard & y["mult"].ge(MULT_TH) & y["z"].ge(Z_TH)
    soft = base_guard & ~hard & (
        y["mult"].ge(MULT_TH - MULT_SOFT) | y["z"].ge(Z_TH - Z_SOFT)
    )

    cols = ["pub_sub_rel_id","hour","clk","roll_med","mult","z"]
    cand_h = y.loc[hard, cols].copy(); cand_h["severity"] = LABEL_CONFIRM
    cand_s = y.loc[soft, cols].copy(); cand_s["severity"] = LABEL_SUSPECT
    out = pd.concat([cand_h, cand_s], ignore_index=True)
    if out.empty:
        return out

    # --- 해당 시각의 상위 mda 라벨링
    mix = (j.groupby(["pub_sub_rel_id","hour","mda_idx"]).size()
             .reset_index(name="cnt"))
    top = (mix.sort_values(["pub_sub_rel_id","hour","cnt"], ascending=[True,True,False])
              .drop_duplicates(["pub_sub_rel_id","hour"]))
    tot = mix.groupby(["pub_sub_rel_id","hour"])["cnt"].sum().reset_index(name="tot_cnt")
    top = top.merge(tot, on=["pub_sub_rel_id","hour"], how="left")
    top["mda_top_share"] = top["cnt"] / top["tot_cnt"]

    out = (out.merge(top[["pub_sub_rel_id","hour","mda_idx","mda_top_share"]],
                     on=["pub_sub_rel_id","hour"], how="left"))

    # --- 퍼블리셔별 Top-K 제한
    out = out.sort_values(["pub_sub_rel_id","severity","mult","z","hour"],
                          ascending=[True, False, False, False, True]).reset_index(drop=True)
    rk = out.groupby(["pub_sub_rel_id","severity"]).cumcount()
    out = out[
        ((out["severity"]==LABEL_CONFIRM) & (rk < TOPK_CONFIRM_PB)) |
        ((out["severity"]==LABEL_SUSPECT) & (rk < TOPK_SUSPECT_PB))
    ].copy()

    out = out.sort_values(["pub_sub_rel_id","hour"]).reset_index(drop=True)
    return out  # severity만 포함, abuse_type 제거

spikes_ad  = detect_click_spike_ad(df_rpt, df_list)              # 광고 단위 (설치/실행형만)
spikes_pub = detect_click_spike_publisher_labeled(df_join)       # 퍼블리셔 단위

SEV2CODE = {"확정":2, "의심":1}

# ========= 라벨 주입 함수만 =========
def build_df_rpt_v1(df_rpt, spikes_ad):
    r = df_rpt.copy()
    # 1) df_rpt의 시간 키(시간 단위) 표준화
    r["rpt_dt"] = pd.to_datetime(
        r["rpt_time_date"].astype(str) + " " + r["rpt_time_time"].astype(str),
        errors="coerce"
    ).dt.floor("H")

    # 2) spikes_ad 표준화 + 중복 제거(가장 높은 severity 선택)
    sa = spikes_ad.copy()

    # spikes_ad의 rpt_dt 타입/단위 정규화
    if "rpt_dt" in sa.columns:
        sa["rpt_dt"] = pd.to_datetime(sa["rpt_dt"], errors="coerce").dt.floor("H")
    else:
        # fallback: 날짜/시간 컬럼이 따로 있다면 동일 방식으로 생성
        if {"rpt_time_date","rpt_time_time"}.issubset(sa.columns):
            sa["rpt_dt"] = pd.to_datetime(
                sa["rpt_time_date"].astype(str)+" "+sa["rpt_time_time"].astype(str),
                errors="coerce"
            ).dt.floor("H")
        elif "date" in sa.columns:
            sa["rpt_dt"] = pd.to_datetime(sa["date"], errors="coerce").dt.floor("H")
        else:
            raise KeyError("spikes_ad에 rpt_dt를 만들 수 있는 컬럼이 필요합니다.")

    sa["ads_idx"] = pd.to_numeric(sa["ads_idx"], errors="coerce")
    # severity → 코드(2:확정, 1:의심, 0:정상)
    sa["abuse_3"] = sa["severity"].map(SEV2CODE).fillna(0).astype("Int8")

    # ▶ 핵심: 키 중복을 합치기(가장 엄한 라벨로)
    lab = (sa.loc[sa["ads_idx"].notna() & sa["rpt_dt"].notna(), ["ads_idx","rpt_dt","abuse_3"]]
             .groupby(["ads_idx","rpt_dt"], as_index=False)["abuse_3"].max())

    # 3) 안전 머지 (오른쪽은 m:1이어야 함)
    out = r.merge(lab, on=["ads_idx","rpt_dt"], how="left", validate="m:1")
    out["abuse_3"] = out["abuse_3"].fillna(0).astype("Int8")

    # 4) 임시 키 제거 + 행수 보존 검증
    out = out.drop(columns=["rpt_dt"])
    assert len(out) == len(r), f"row mismatch: before={len(r)}, after={len(out)}"
    return out

def build_df_join_v1(df_join, spikes_pub):
    j = df_join.copy()
    j["hour"] = pd.to_datetime(j["click_date"], errors="coerce").dt.floor("H")
    lab = spikes_pub[["pub_sub_rel_id","hour","severity"]].copy()
    lab["abuse_3"] = lab["severity"].map(SEV2CODE).astype("Int8")
    out = j.merge(lab.drop(columns="severity"), on=["pub_sub_rel_id","hour"], how="left")
    out["abuse_3"] = out["abuse_3"].fillna(0).astype("Int8")
    return out.drop(columns="hour")

def build_df_list_v1(df_list, df_rpt_v1, install_types={1,2}, weighted=False):
    meta = df_list[["ads_idx","ads_type"]]
    rpt  = df_rpt_v1.merge(meta, on="ads_idx", how="left")

    den  = (rpt[rpt["ads_type"].isin(install_types)]
            .groupby("ads_idx").size().reset_index(name="tot_rows"))

    x = rpt.loc[rpt["abuse_3"]>0, ["ads_idx","abuse_3"]]
    if weighted:
        x = x.assign(w=x["abuse_3"].map({1:1,2:2}).astype("int8"))
        num = x.groupby("ads_idx")["w"].sum().reset_index(name="flag_rows")
    else:
        num = x.groupby("ads_idx").size().reset_index(name="flag_rows")

    rate = den.merge(num, on="ads_idx", how="left").fillna({"flag_rows":0})
    rate["abuse_3"] = (rate["flag_rows"]/rate["tot_rows"]).clip(0,1).astype("float32")
    return df_list.merge(rate[["ads_idx","abuse_3"]], on="ads_idx", how="left").fillna({"abuse_3":0.0})

# ========= 실행 =========
# 필요시에만 로드
df_list_v1 = pd.read_csv("df_list_v1.csv")
df_join_v1 = pd.read_csv("df_join_v1.csv")

df_rpt_v1  = build_df_rpt_v1(df_rpt, spikes_ad)
df_join_v1 = build_df_join_v1(df_join_v1, spikes_pub)   # 기존 v1에 abuse_3 추가
df_list_v1 = build_df_list_v1(df_list_v1, df_rpt_v1, weighted=False)  # 기존 v1에 abuse_3 추가

# 확인 후 저장
df_rpt_v1.to_csv("df_rpt_v1.csv", index=False)
df_join_v1.to_csv("df_join_v1.csv", index=False)
df_list_v1.to_csv("df_list_v1.csv", index=False)