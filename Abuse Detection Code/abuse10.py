# ===== 파라미터 =====
N_MIN_GROUP   = 200      # 퍼블리셔 1차 스크리닝 최소 N
N_MIN_ENTITY  = 30       # 엔티티(dvc/ip) 최소 N
GROUP_BY_CTIT = True     # df_settle의 CTIT 보강 사용 여부

SEC_MODE_WARN, SEC_MODE_RISK = 0.45, 0.62   # 특정 '초(0~59)' 최빈 점유율 임계
IAT_MODE_WARN, IAT_MODE_RISK = 0.65, 0.85   # 클릭간격(초) 모드 점유율(±1s 허용)

CTIT_BIN = 1.0
CTIT_CV_WARN,  CTIT_CV_RISK  = 0.18, 0.065   # CV 낮을수록 의심
CTIT_MODE_WARN, CTIT_MODE_RISK= 0.45, 0.62  # CTIT 최빈구간 점유율

MEDIA_SERVER_PATTERNS = [
    r'^43\.203\.', r'^3\.38\.', r'^16\.184\.', r'^54\.180\.',
    r'^15\.165\.', r'^13\.125\.', r'^52\.79\.', r'^34\.64\.', r'^175\.126\.'
]

# ===== IP 품질 필터 =====
def _is_media_server_ip(ip):
    if pd.isna(ip): return False
    s=str(ip).strip()
    return any(re.match(p, s) for p in MEDIA_SERVER_PATTERNS)
def _is_private_ip(ip):
    if pd.isna(ip): return False
    s=str(ip).strip()
    return s.startswith("10.") or s.startswith("192.168.") or bool(re.match(r"^172\.(1[6-9]|2[0-9]|3[01])\.", s))
def _valid_web_ip(ip): 
    return (isinstance(ip,str) or not pd.isna(ip)) and (not _is_media_server_ip(ip)) and (not _is_private_ip(ip))

# ===== 유틸 =====
def _preprocess_join(df_join: pd.DataFrame):
    use = ["click_key","ads_idx","mda_idx","pub_sub_rel_id","dvc_idx","user_ip","click_date","click_time"]
    use = [c for c in use if c in df_join.columns]
    j = df_join[use].copy()
    j["ads_idx"]        = pd.to_numeric(j.get("ads_idx"), errors="coerce").astype("Int64")
    j["mda_idx"]        = pd.to_numeric(j.get("mda_idx"), errors="coerce").astype("Int64")
    j["pub_sub_rel_id"] = pd.to_numeric(j.get("pub_sub_rel_id"), errors="coerce").astype("Int64")
    j["dvc_idx"]        = pd.to_numeric(j.get("dvc_idx"), errors="coerce").fillna(0).astype("Int64")
    j["ts"]             = pd.to_datetime(j["click_date"], errors="coerce")  # 예: 2025-08-17 21:07:37
    j = j[j["ts"].notna()]
    j["sec_in_min"]     = j["ts"].dt.second
    j["ts_s"]           = (j["ts"].view("int64")//10**9).astype("int64")   # epoch sec
    # 엔티티 키: 앱=dvc_idx, 웹(dvc=0)=품질통과 user_ip
    web = (j["dvc_idx"]==0)
    j["_entity"] = np.where(web, j["user_ip"].where(j["user_ip"].apply(_valid_web_ip)),
                            j["dvc_idx"].astype("string"))
    j = j[j["_entity"].notna()]
    j["_entity"] = j["_entity"].astype("string")
    return j

def _sec_mode_share(sec_series: pd.Series):
    if len(sec_series)==0: return np.nan
    vc = sec_series.value_counts(normalize=True)
    return float(vc.iloc[0])

def _iat_mode_share(ts_s: np.ndarray):
    if len(ts_s)<2: return np.nan
    d = np.diff(np.sort(ts_s))
    d = d[(d>0) & (d<3600*3)]  # 3h 초과 배치성 간격 제거
    if len(d)==0: return np.nan
    b = pd.Series(d).round().astype("int64")
    vc = b.value_counts()
    # ±1초 이웃 합치기
    merged={}
    for k,c in vc.items():
        if k in merged: merged[k]+=c
        else: merged[k]=c+vc.get(k-1,0)+vc.get(k+1,0)
    return max(merged.values())/len(b) if merged else np.nan

def _ctit_features(df_settle: pd.DataFrame):
    if {"click_key","ads_idx","ctit"}.issubset(df_settle.columns)==False:
        return pd.DataFrame(columns=["click_key","ads_idx","ctit_s"])
    s = df_settle[["click_key","ads_idx","ctit"]].copy()
    x = pd.to_numeric(s["ctit"], errors="coerce")
    if x.dropna().median()>1000: x = x/1000.0   # ms→s
    x = x.where((x>=0)&(x<=6*3600))
    s["ctit_s"]=x
    return s.dropna(subset=["ctit_s"])[["click_key","ads_idx","ctit_s"]]

def _ctit_group_stats(df: pd.DataFrame):
    if df.empty: return pd.DataFrame(columns=["grp","n","cv","mode_share"])
    g = df.groupby(["grp"], observed=True)["ctit_s"]
    stat = g.agg(n="count", mean="mean", std="std").reset_index()
    stat["cv"] = stat["std"]/stat["mean"].replace(0,np.nan)
    b = df.copy()
    b["ctit_bin"]=(b["ctit_s"]/CTIT_BIN).round().astype("Int64")
    top = (b.groupby(["grp","ctit_bin"], observed=True).size()
             .rename("cnt").reset_index()
             .sort_values(["grp","cnt"], ascending=[True,False])
             .groupby("grp", observed=True).head(1))
    stat = stat.merge(top[["grp","cnt"]], on="grp", how="left")
    stat["mode_share"]=stat["cnt"]/stat["n"]
    return stat[["grp","n","cv","mode_share"]]

def _severity(sec_mode, iat_mode, n, ctit_cv=None, ctit_mode=None, n_min=N_MIN_ENTITY):
    if n < n_min: return "정상"
    risk = ( (not pd.isna(sec_mode) and sec_mode>=SEC_MODE_RISK) or
             (not pd.isna(iat_mode) and iat_mode>=IAT_MODE_RISK) or
             (ctit_cv   is not None and not pd.isna(ctit_cv)   and ctit_cv   <= CTIT_CV_RISK) or
             (ctit_mode is not None and not pd.isna(ctit_mode) and ctit_mode >= CTIT_MODE_RISK) )
    if risk: return "확정"
    warn = ( (not pd.isna(sec_mode) and sec_mode>=SEC_MODE_WARN) or
             (not pd.isna(iat_mode) and iat_mode>=IAT_MODE_WARN) or
             (ctit_cv   is not None and not pd.isna(ctit_cv)   and ctit_cv   <= CTIT_CV_WARN) or
             (ctit_mode is not None and not pd.isna(ctit_mode) and ctit_mode >= CTIT_MODE_WARN) )
    return "의심" if warn else "정상"

# ===== 메인 =====
def run_drilldown_to_df(df_list, df_join, df_settle, df_rpt):
    j = _preprocess_join(df_join)

    # 1) mda_idx → pub_sub_rel_id 1차 스크리닝
    pub_g   = j.groupby(["mda_idx","pub_sub_rel_id"], observed=True)
    pub_feat= pub_g.agg(n=("click_key","count"),
                        sec_mode=("sec_in_min", _sec_mode_share)).reset_index()
    pub_iat = pub_g["ts_s"].apply(lambda s: _iat_mode_share(s.values)).reset_index(name="iat_mode")
    pub_feat = pub_feat.merge(pub_iat, on=["mda_idx","pub_sub_rel_id"], how="left")
    pub_feat["sev1"] = pub_feat.apply(
        lambda r: _severity(r["sec_mode"], r["iat_mode"], r["n"], None, None, N_MIN_GROUP), axis=1
    )
    flagged_pubs = pub_feat.loc[pub_feat["sev1"].isin(["의심","확정"]), ["mda_idx","pub_sub_rel_id"]]
    if flagged_pubs.empty:
        return j.head(0).assign(severity=[])

    # 2) 플래그 퍼블리셔 내부: entity(dvc or web-ip) 정밀
    jj = j.merge(flagged_pubs.drop_duplicates(), on=["mda_idx","pub_sub_rel_id"], how="inner")
    ent_cols = ["mda_idx","pub_sub_rel_id","_entity"]
    ent_g   = jj.groupby(ent_cols, observed=True)
    ent_feat= ent_g.agg(n=("click_key","count"),
                        sec_mode=("sec_in_min", _sec_mode_share)).reset_index()
    ent_iat = ent_g["ts_s"].apply(lambda s: _iat_mode_share(s.values)).reset_index(name="iat_mode")
    ent_feat = ent_feat.merge(ent_iat, on=ent_cols, how="left")

    # (선택) CTIT 보강
    ent_feat["ctit_cv"] = np.nan
    ent_feat["ctit_mode"] = np.nan
    if GROUP_BY_CTIT:
        ctit = _ctit_features(df_settle)
        tmp = jj.merge(ctit, on=["click_key","ads_idx"], how="inner")
        tmp["grp"] = tmp["_entity"]
        cg = _ctit_group_stats(tmp[["grp","ctit_s"]].copy())
        cv_map   = dict(zip(cg["grp"].astype("string"), cg["cv"]))
        mode_map = dict(zip(cg["grp"].astype("string"), cg["mode_share"]))
        ent_feat["ctit_cv"]   = ent_feat["_entity"].map(cv_map)
        ent_feat["ctit_mode"] = ent_feat["_entity"].map(mode_map)

    ent_feat["severity"] = ent_feat.apply(
        lambda r: _severity(r["sec_mode"], r["iat_mode"], r["n"], r["ctit_cv"], r["ctit_mode"], N_MIN_ENTITY), axis=1
    )

    # 3) 행 복원 → 단일 DF
    bad_entities = ent_feat.loc[ent_feat["severity"].isin(["의심","확정"]), ent_cols+["severity"]]
    out = jj.merge(bad_entities, on=ent_cols, how="inner")
    out = (out[["click_key","ads_idx","mda_idx","pub_sub_rel_id","_entity","ts","severity"]]
              .rename(columns={"_entity":"entity_key","ts":"click_dt"})
              .sort_values(["severity","click_dt"], ascending=[False,True])
              .reset_index(drop=True))
    return out

# 실행:
abuse_df = run_drilldown_to_df(df_list, df_join, df_settle, df_rpt)

# 데이터 라벨링
# 1) severity → abuse_10 매핑 (중복 click_key는 최댓값(확정>의심>정상) 유지)
SEV_MAP = {"정상": 0, "의심": 1, "확정": 2}
label_map = (abuse_df.assign(abuse_10=abuse_df["severity"].map(SEV_MAP))
             .sort_values("abuse_10", ascending=False)
             .drop_duplicates("click_key")[["click_key","abuse_10"]])

# 2) 공통 적용 함수: click_key 있으면 merge, 없으면 0으로
def apply_labels_by_click(df):
    out = df.copy()
    if "click_key" in out.columns:
        out = out.merge(label_map, on="click_key", how="left")
        out["abuse_10"] = out["abuse_10"].fillna(0).astype("int8")
    else:
        # 행 추적 불가: 보수적으로 전부 정상(0)
        out["abuse_10"] = np.int8(0)
    return out

df_join_v1   = apply_labels_by_click(df_join_v1)
df_settle_v1 = apply_labels_by_click(df_settle_v1)

# 3) df_list에 ads_idx별 어뷰징 구성비(0~1) 부착
#    - 분모: df_join_v1의 ads_idx별 전체 클릭 수
#    - 분자: abuse_10 > 0(의심+확정) 클릭 수
ads_total = (df_join_v1.groupby("ads_idx").size()
             .rename("total_clicks")).to_frame()
ads_abuse = (df_join_v1.loc[df_join_v1["abuse_10"]>0]
             .groupby("ads_idx").size()
             .rename("abusive_clicks")).to_frame()

ads_ratio = ads_total.join(ads_abuse, how="left").fillna(0)
ads_ratio["abuse_10"] = (ads_ratio["abusive_clicks"] / ads_ratio["total_clicks"]).clip(0,1)

df_list_v1 = df_list_v1.merge(
    ads_ratio[["abuse_10"]].reset_index(),
    on="ads_idx", how="left"
)
df_list_v1["abuse_10"] = df_list_v1["abuse_10"].fillna(0.0).astype("float32")

# 확인 후 저장
df_join_v1.to_csv("df_join_v1.csv", index=False)
df_list_v1.to_csv("df_list_v1.csv", index=False)
df_settle_v1.to_csv("df_settle_v1.csv", index=False)