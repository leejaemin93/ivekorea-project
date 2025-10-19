# ===== 파라미터 =====
WINDOW_MIN = 10          # 창 길이(분)
K_THRESH   = 4           # 창 내 서로 다른 광고 개수 기준
MIN_INSTALL_IN_WIN = 2   # 창 내 설치/실행형 최소 개수

USER_LEVEL = 1           # 0=느슨, 1=보통(권장), 2=엄격
PUB_LEVEL  = 1           # 0=느슨, 1=보통(권장), 2=엄격

CHUNK_DAYS        = 3    # 날짜 청크(일)
CHUNK_OVERLAP_MIN = 10   # 청크 경계 오버랩(분)

INSTALL_TYPES = {1, 2}   # 1:설치형, 2:실행형

def detect_multi_participation_chunked(
    df_join, df_list,
    *,
    window_min=WINDOW_MIN, k_thresh=K_THRESH, min_install_in_win=MIN_INSTALL_IN_WIN,
    user_level=USER_LEVEL, pub_level=PUB_LEVEL,
    chunk_days=CHUNK_DAYS, chunk_overlap_min=CHUNK_OVERLAP_MIN,
    # ↓ 필요하면 세부 기준을 개별 오버라이드 할 수도 있음(보통은 건들 필요 없음)
    pub_min_sus_win=20, pub_min_sus_users=10, pub_ratio_thresh=0.05,
    install_types=INSTALL_TYPES, topn_ads=3
):

    # ---------- 레벨 → 세부 임계 자동 파생 ----------
    # 유저-창: A(uniq_ads 여유), B(install 여유), C(같은날 반복)
    if user_level <= 0:   # 느슨
        USER_SUS_SIGS, USER_CONF_SIGS = 1, 2
        USER_K_MARGIN, USER_INSTALL_MARGIN = 1, 0
        USER_REPEAT_MIN = 2
    elif user_level == 1: # 보통(권장)
        USER_SUS_SIGS, USER_CONF_SIGS = 1, 2
        USER_K_MARGIN, USER_INSTALL_MARGIN = 2, 1
        USER_REPEAT_MIN = 3
    else:                 # 엄격
        USER_SUS_SIGS, USER_CONF_SIGS = 2, 3
        USER_K_MARGIN, USER_INSTALL_MARGIN = 3, 2
        USER_REPEAT_MIN = 4

    # 퍼블리셔-일: 신호(S1,S2,S3) 개수와 배수 기준
    if pub_level <= 0:    # 느슨
        PUB_SUS_SIGS, PUB_CONF_SIGS = 1, 2
        PUB_SUS_MULT, PUB_CONF_MULT = 0.8, 1.5
    elif pub_level == 1:  # 보통(권장)
        PUB_SUS_SIGS, PUB_CONF_SIGS = 1, 2
        PUB_SUS_MULT, PUB_CONF_MULT = 1.0, 2.0
    else:                 # 엄격
        PUB_SUS_SIGS, PUB_CONF_SIGS = 2, 3
        PUB_SUS_MULT, PUB_CONF_MULT = 1.2, 2.5

    # 과소 데이터 유저 사전 스킵 임계: k 기준에서 자동 파생
    prefilter_user_min = max(5, k_thresh)

    # ---------- 메타 병합 ----------
    meta = df_list[["ads_idx","ads_type"]].drop_duplicates("ads_idx")
    meta["is_install_like"] = meta["ads_type"].isin(install_types)

    # ---------- 최소 컬럼 + 전처리 ----------
    j = (df_join[["click_key","ads_idx","dvc_idx","user_ip","click_date","pub_sub_rel_id","mda_idx"]]
         .merge(meta[["ads_idx","is_install_like"]], on="ads_idx", how="left", validate="m:1"))
    j["dt"] = pd.to_datetime(j["click_date"], errors="coerce")
    j = j[j["dt"].notna()].copy()
    j["dvc_idx"] = pd.to_numeric(j["dvc_idx"], errors="coerce").fillna(0).astype("int64")
    j["user_ip"] = j["user_ip"].astype(str).str.strip()
    j["user_id"] = np.where(j["dvc_idx"].ne(0), "dvc:"+j["dvc_idx"].astype(str), "ip:"+j["user_ip"])
    j = j[(j["user_id"]!="ip:") & (j["user_id"]!="ip:nan")]
    j = j[["user_id","dt","ads_idx","pub_sub_rel_id","mda_idx","is_install_like"]]

    start = j["dt"].min().floor("D")
    end   = j["dt"].max().ceil("D")
    if pd.isna(start) or pd.isna(end):
        empty_u = pd.DataFrame(columns=["abuse_type","user_id","win_start","win_end","uniq_ads","install_cnt","ads_list","severity"])
        empty_p = pd.DataFrame(columns=["abuse_type","pub_sub_rel_id","mda_idx","date","sus_windows","sus_users","sus_ratio","sus_ads_topn","severity"])
        return empty_u, empty_p

    out_user_parts, out_pub_parts = [], []
    cur = start; ol = pd.Timedelta(minutes=chunk_overlap_min)

    while cur < end:
        cur_end = min(cur + pd.Timedelta(days=chunk_days), end)
        c = j.loc[(j["dt"] >= (cur - ol)) & (j["dt"] < (cur_end + ol))].copy()
        if c.empty: cur = cur_end; continue

        # 사전 필터
        cnt_user = c.groupby("user_id", observed=True).size()
        keep_users = cnt_user[cnt_user >= prefilter_user_min].index
        c = c[c["user_id"].isin(keep_users)]
        if c.empty: cur = cur_end; continue

        c["win_start"] = c["dt"].dt.floor(f"{window_min}min")
        c["date"] = c["win_start"].dt.date

        # ===== 유저-창 집계 =====
        base_u = (c.drop_duplicates(["user_id","win_start","ads_idx"])
                    .groupby(["user_id","win_start"], as_index=False)
                    .agg(uniq_ads=("ads_idx","nunique"),
                         install_cnt=("is_install_like", lambda s: int(s.fillna(False).sum()))))
        u_ads = (c.drop_duplicates(["user_id","win_start","ads_idx"])
                   .groupby(["user_id","win_start"])["ads_idx"]
                   .apply(lambda s: ",".join(map(str, sorted(s)))).reset_index(name="ads_list"))

        # 1차 후보
        u_flags = base_u[(base_u["uniq_ads"]>=k_thresh) & (base_u["install_cnt"]>=min_install_in_win)].copy()
        if not u_flags.empty:
            u_flags = u_flags.merge(u_ads, on=["user_id","win_start"], how="left")
            u_flags["win_end"] = u_flags["win_start"] + pd.Timedelta(minutes=window_min)
            u_flags["abuse_type"] = "다중참여(user)"

            # 신호 A/B/C (레벨 기반)
            A = (u_flags["uniq_ads"]    >= (k_thresh + USER_K_MARGIN)).astype(int)
            B = (u_flags["install_cnt"] >= (min_install_in_win + USER_INSTALL_MARGIN)).astype(int)
            tmp = u_flags.copy(); tmp["date"] = tmp["win_start"].dt.date
            C = (tmp.groupby(["user_id","date"])["win_start"].transform("count") >= USER_REPEAT_MIN).astype(int)

            user_sigs = A + B + C
            u_flags["severity"] = np.select(
                [user_sigs >= USER_CONF_SIGS, user_sigs >= USER_SUS_SIGS],
                ["확정","의심"], default="정상"
            )
            u_flags = u_flags[u_flags["severity"]!="정상"]
            if not u_flags.empty:
                u_flags["severity"] = pd.Categorical(u_flags["severity"], categories=["의심","확정"], ordered=True)
                out_user_parts.append(u_flags[["abuse_type","user_id","win_start","win_end","uniq_ads","install_cnt","ads_list","severity"]])

        # ===== 퍼블리셔-일 집계 =====
        base_p = (c.drop_duplicates(["pub_sub_rel_id","mda_idx","user_id","win_start","ads_idx"])
                    .groupby(["pub_sub_rel_id","mda_idx","user_id","win_start"], as_index=False)
                    .agg(uniq_ads=("ads_idx","nunique"),
                         install_cnt=("is_install_like", lambda s: int(s.fillna(False).sum()))))
        p_flags = base_p[(base_p["uniq_ads"]>=k_thresh) & (base_p["install_cnt"]>=min_install_in_win)].copy()
        if not p_flags.empty:
            p_flags["date"] = p_flags["win_start"].dt.date

        p_day_tot = (base_p.assign(date=base_p["win_start"].dt.date)
                          .groupby(["pub_sub_rel_id","mda_idx","date"], as_index=False)
                          .size().rename(columns={"size":"total_user_windows"}))
        if not p_flags.empty:
            p_day_hit = (p_flags.groupby(["pub_sub_rel_id","mda_idx","date"], as_index=False)
                               .agg(sus_windows=("win_start","nunique"),
                                    sus_users=("user_id","nunique")))
        else:
            p_day_hit = pd.DataFrame(columns=["pub_sub_rel_id","mda_idx","date","sus_windows","sus_users"])

        pub_daily = (p_day_tot.merge(p_day_hit, on=["pub_sub_rel_id","mda_idx","date"], how="left")
                               .fillna({"sus_windows":0,"sus_users":0}))
        pub_daily["sus_ratio"] = np.where(pub_daily["total_user_windows"]>0,
                                          pub_daily["sus_windows"]/pub_daily["total_user_windows"], np.nan)

        # 후보(퍼블리셔 일 단위)
        pub_flags = pub_daily[
            (pub_daily["sus_windows"]>=pub_min_sus_win) |
            (pub_daily["sus_users"]>=pub_min_sus_users) |
            (pub_daily["sus_ratio"]>=pub_ratio_thresh)
        ].copy()

        if not pub_flags.empty:
            # Top-N 광고 (의심창에서 빈번)
            p_win_ads = (c.drop_duplicates(["pub_sub_rel_id","mda_idx","user_id","win_start","ads_idx"])
                           .groupby(["pub_sub_rel_id","mda_idx","user_id","win_start"])["ads_idx"]
                           .apply(list).reset_index(name="ads_list"))
            if not p_flags.empty:
                p_flags_ads = p_flags.merge(p_win_ads, on=["pub_sub_rel_id","mda_idx","user_id","win_start"], how="left")
                pf_exploded = (p_flags_ads.explode("ads_list")
                                            .dropna(subset=["ads_list"])
                                            .rename(columns={"ads_list":"ads_idx"}))
                daily_ads_cnt = (pf_exploded.groupby(["pub_sub_rel_id","mda_idx","date","ads_idx"], as_index=False)
                                               .size().rename(columns={"size":"cnt"}))
                def _topn_str(g):
                    g2 = g.sort_values("cnt", ascending=False).head(topn_ads)
                    return ",".join([f"{int(a)}({int(c)})" for a,c in zip(g2["ads_idx"], g2["cnt"])])
                sus_ads_topn = (daily_ads_cnt.groupby(["pub_sub_rel_id","mda_idx","date"])
                                                .apply(_topn_str).reset_index(name="sus_ads_topn"))
                pub_flags = pub_flags.merge(sus_ads_topn, on=["pub_sub_rel_id","mda_idx","date"], how="left")
            else:
                pub_flags["sus_ads_topn"] = np.nan

            # 퍼블리셔 severity (레벨 기반)
            S1_sus  = (pub_flags["sus_windows"] >= (PUB_SUS_MULT  * pub_min_sus_win)).astype(int)
            S2_sus  = (pub_flags["sus_users"]  >= (PUB_SUS_MULT  * pub_min_sus_users)).astype(int)
            S3_sus  = (pub_flags["sus_ratio"]  >= (PUB_SUS_MULT  * pub_ratio_thresh)).astype(int)
            S1_conf = (pub_flags["sus_windows"] >= (PUB_CONF_MULT * pub_min_sus_win)).astype(int)
            S2_conf = (pub_flags["sus_users"]  >= (PUB_CONF_MULT * pub_min_sus_users)).astype(int)
            S3_conf = (pub_flags["sus_ratio"]  >= (PUB_CONF_MULT * pub_ratio_thresh)).astype(int)

            conf_sigs = S1_conf + S2_conf + S3_conf
            sus_sigs  = S1_sus  + S2_sus  + S3_sus
            pub_flags["severity"] = np.select(
                [conf_sigs >= PUB_CONF_SIGS, sus_sigs >= PUB_SUS_SIGS],
                ["확정","의심"], default="정상"
            )
            pub_flags = pub_flags[pub_flags["severity"]!="정상"]
            if not pub_flags.empty:
                pub_flags["abuse_type"] = "다중참여(pub)"
                pub_flags["severity"] = pd.Categorical(pub_flags["severity"], categories=["의심","확정"], ordered=True)
                out_pub_parts.append(pub_flags[["abuse_type","pub_sub_rel_id","mda_idx","date","sus_windows","sus_users","sus_ratio","sus_ads_topn","severity"]])

        cur = cur_end

    # ---------- 합치기 ----------
    if out_user_parts:
        u_all = (pd.concat(out_user_parts, ignore_index=True)
                   .drop_duplicates(["user_id","win_start"])
                   .sort_values(["severity","user_id","win_start"], ascending=[True, True, True])
                   .reset_index(drop=True))
    else:
        u_all = pd.DataFrame(columns=["abuse_type","user_id","win_start","win_end","uniq_ads","install_cnt","ads_list","severity"])

    if out_pub_parts:
        p_all = (pd.concat(out_pub_parts, ignore_index=True)
                   .drop_duplicates(["pub_sub_rel_id","mda_idx","date"])
                   .sort_values(["severity","pub_sub_rel_id","date"], ascending=[True, True, True])
                   .reset_index(drop=True))
    else:
        p_all = pd.DataFrame(columns=["abuse_type","pub_sub_rel_id","mda_idx","date","sus_windows","sus_users","sus_ratio","sus_ads_topn","severity"])

    return u_all, p_all

# 결과
u_flags, pub_flags = detect_multi_participation_chunked(df_join, df_list)

def enrich_with_multi_participation_flags(df_join, u_flags, pub_flags):
    """
    df_join에 '다중 참여' 탐지 결과를 바탕으로 abuse_5 컬럼(0,1,2)을 추가합니다.
    """
    print("--- '다중 참여' 등급 부여 시작 ---")
    
    j = df_join.copy()
    j["dt"] = pd.to_datetime(j["click_date"], errors="coerce")
    j["user_id"] = np.where(j["dvc_idx"].fillna(0).ne(0), "dvc:"+j["dvc_idx"].astype(str), "ip:"+j["user_ip"].astype(str))
    severity_map_numeric = {'정상': 0, '의심': 1, '확정': 2}

    # --- 1단계: 유저(user_id) 기반 등급 부여 ---
    if not u_flags.empty:
        j["win_start"] = j["dt"].dt.floor(f"{WINDOW_MIN}min") 
        u_flags_mapped = u_flags[['user_id', 'win_start', 'severity']].copy()
        u_flags_mapped['abuse_user_sev'] = u_flags_mapped['severity'].map(severity_map_numeric)
        
        # ▼▼▼ 수정된 부분: 필요한 'abuse_user_sev' 컬럼만 merge 하도록 명시 ▼▼▼
        j = j.merge(u_flags_mapped[['user_id', 'win_start', 'abuse_user_sev']], on=['user_id', 'win_start'], how='left')
        j.drop(columns=['win_start'], inplace=True, errors='ignore')
    else:
        j['abuse_user_sev'] = np.nan
    
    # --- 2단계: 퍼블리셔(pub_sub_rel_id) 기반 등급 부여 ---
    if not pub_flags.empty:
        j["date"] = j["dt"].dt.date
        pub_flags_mapped = pub_flags[['pub_sub_rel_id', 'mda_idx', 'date', 'severity']].copy()
        pub_flags_mapped['abuse_pub_sev'] = pub_flags_mapped['severity'].map(severity_map_numeric)

        # ▼▼▼ 수정된 부분: 필요한 'abuse_pub_sev' 컬럼만 merge 하도록 명시 ▼▼▼
        j = j.merge(pub_flags_mapped[['pub_sub_rel_id', 'mda_idx', 'date', 'abuse_pub_sev']], on=['pub_sub_rel_id', 'mda_idx', 'date'], how='left')
        j.drop(columns=['date'], inplace=True, errors='ignore')
    else:
        j['abuse_pub_sev'] = np.nan
        
    # --- 3단계: 최종 abuse_5 등급 확정 ---
    j['abuse_user_sev'] = j['abuse_user_sev'].astype(float).fillna(0)
    j['abuse_pub_sev'] = j['abuse_pub_sev'].astype(float).fillna(0)
    j['abuse_5'] = j[['abuse_user_sev', 'abuse_pub_sev']].max(axis=1).astype('int8')
    
    # 중간 과정에서 사용된 컬럼들 제거
    j.drop(columns=['dt', 'user_id', 'abuse_user_sev', 'abuse_pub_sev'], inplace=True, errors='ignore')

    return j

# =================================================================
def build_list_with_multi_participation_rate(df_list, df_join_v5):
    # (이전 코드와 동일)
    print("--- df_list에 '어뷰징 오염도' 계산 시작 ---")
    temp_df = df_join_v5[["ads_idx", "abuse_5"]].copy()
    temp_df["is_abuse"] = (temp_df["abuse_5"] > 0).astype("int8")
    abuse_rate = temp_df.groupby("ads_idx")["is_abuse"].mean().rename("abuse_5")
    df_list_v1 = df_list.merge(abuse_rate, on="ads_idx", how="left")
    df_list_v1["abuse_5"].fillna(0, inplace=True)
    print("--- df_list 보강 완료: 'abuse_5' 비율 계산 완료 ---\n")
    return df_list_v1

# --- 1. df_join에 등급 부여 실행 ---
df_join_v1 = enrich_with_multi_participation_flags(df_join_v1, u_flags, pub_flags)

# --- 2. df_list에 오염도 비율 부여 실행 ---
df_list_v1 = build_list_with_multi_participation_rate(df_list_v1, df_join_v1)

# 저장
df_join_v1.to_csv("df_join_v1.csv", index=False)
df_list_v1.to_csv("df_list_v1.csv", index=False)