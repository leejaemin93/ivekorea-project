# ===== 파라미터 (실무용 임계치) =====
SWITCH_MODE = "auto"           # "auto" | "force_ip" | "force_device"

# IP 품질(보수적)
IP_TOP_SHARE_TH      = 0.999        # 최다 IP 점유율
IP_UNIQUE_RATIO_TH   = 0.00001      # 고유비율
MIN_ROWS_FOR_IP_CHECK= 1000         # 최소 표본

# 분위수
P95_Q, P99_Q, P999_Q = 0.95, 0.99, 0.999

# 시간 컬럼 후보
TIME_COLS = ["click_time","click_date"]

# 라벨 임계(보수)
CONFIRM_HITS_MIN   = 3     # 확정 최소 룰수
SUSPECT_HITS_MIN   = 2     # 의심 최소 룰수
CONFIRM_SCORE_TH   = 5.0   # 확정 점수
SUSPECT_SCORE_TH   = 2.0   # 의심 점수
Z_GATE             = 5.0
ADS_OVERLAP_MIN_WINDOWS = 50
IP_REPEAT_STRICT_DAYS   = 1
DEV_ACTIVE_DAYS_MAX     = 1

# (선택) 매체사 서버 IP 프리픽스(휴리스틱) ------------------------------------------
MEDIA_SERVER_PATTERNS = [
    r'^43\.203\.', r'^3\.38\.', r'^16\.184\.', r'^54\.180\.',
    r'^15\.165\.', r'^13\.125\.', r'^52\.79\.', r'^34\.64\.', r'^175\.126\.'
]

def _is_media_server_ip(ip_str):
    """휴리스틱: 매체사 서버 IP 프리픽스 매칭(정확히 하려면 CIDR 기반으로 교체 권장)."""
    if ip_str is None: return False
    s = str(ip_str).strip()
    if not s or s == "nan": return False
    for pat in MEDIA_SERVER_PATTERNS:
        if re.match(pat, s):
            return True
    return False

# 유틸 ==============================================================================

import numpy as np, pandas as pd

def _check_cols(df, need, name="df"):
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"{name} 필수 컬럼 누락: {miss}")

def _prep_time(df: pd.DataFrame) -> pd.DataFrame:
    """click_time/ click_date → t, day, 그리고 5/10/60분 버킷. dvc_idx 결측/0 제거 X(품질 판단에서 처리)."""
    d = df.copy()
    t = pd.to_datetime(d["click_time"], errors="coerce") if "click_time" in d else None
    if t is None or t.isna().all():
        t = pd.to_datetime(d["click_date"], errors="coerce") if "click_date" in d else None
    d["t"] = t
    d = d.dropna(subset=["t"])
    d["day"] = d["t"].dt.date
    ts = d["t"].astype("int64") // 1_000_000_000
    d["b5"], d["b10"], d["b1h"] = (ts // 300), (ts // 600), (ts // 3600)
    # 메모리 절감
    for c in ["user_ip","dvc_idx","ads_idx","mda_idx","pub_sub_rel_id"]:
        if c in d.columns: d[c] = d[c].astype("category")
    return d

def _pick_group_key(df):
    if "mda_idx" in df.columns: return "mda_idx"
    if "pub_sub_rel_id" in df.columns: return "pub_sub_rel_id"
    return None

def _ip_quality(df: pd.DataFrame):
    """그룹별 IP 품질(서버 IP 예외) 요약: unique_ratio, top1_share, ip_unreliable, media_server_detected."""
    if "user_ip" not in df.columns:
        key = _pick_group_key(df)
        idx = [-1] if key is None else df[key].unique()
        out = pd.DataFrame({"group_key": idx, "rows":[len(df)], "unique_ip":[0], "top1_share":[1.0]})
        out["unique_ratio"] = 0.0
        out["top1_ip"] = ""
        out["media_server_detected"] = False
        out["ip_unreliable"] = False
        return key, out

    key = _pick_group_key(df)
    if key is None:
        vc = df["user_ip"].astype(str).value_counts(dropna=False)
        rows = int(len(df)); uniq = int(vc.size)
        top1_ip = vc.index[0] if len(vc) else ""
        top1_share = float(vc.iloc[0]/rows) if rows else 1.0
        out = pd.DataFrame([{
            "group_key": -1, "rows": rows, "unique_ip": uniq,
            "top1_share": top1_share, "top1_ip": top1_ip
        }])
    else:
        grp = df.groupby(key, observed=True)
        rows = grp.size()
        uniq = grp["user_ip"].apply(lambda s: s.astype(str).nunique())
        top1_counts = grp["user_ip"].apply(lambda s: s.astype(str).value_counts().iloc[0] if len(s) else 0)
        top1_ips    = grp["user_ip"].apply(lambda s: s.astype(str).value_counts().index[0] if len(s) else "")
        out = pd.DataFrame({
            "group_key": rows.index,
            "rows": rows.values,
            "unique_ip": uniq.values,
            "top1_count": top1_counts.values,
            "top1_ip": top1_ips.values
        })
        out["top1_share"] = (out["top1_count"] / out["rows"]).astype(float)

    out["unique_ratio"] = out["unique_ip"].clip(lower=1) / out["rows"].clip(lower=1)
    out["media_server_detected"] = out["top1_ip"].apply(_is_media_server_ip)
    out["ip_unreliable"] = (
        (out["rows"] >= MIN_ROWS_FOR_IP_CHECK)
        & (~out["media_server_detected"])
        & (out["top1_share"] >= IP_TOP_SHARE_TH)
        & (out["unique_ratio"] <= IP_UNIQUE_RATIO_TH)
    )
    return key, out

def _drop_dvc_zero(df: pd.DataFrame) -> pd.DataFrame:
    """dvc_idx==0 제거(웹 디바이스 없음)."""
    if "dvc_idx" not in df.columns: return df
    return df[(df["dvc_idx"].astype(str).str.strip() != "0") & (df["dvc_idx"].notna())]

# 피처 ==============================================================================

def _compute_features(df: pd.DataFrame):
    """IP/Device 피처 계산(서버 IP 제외, dvc_idx==0 제외)."""
    d = _drop_dvc_zero(df)

    # 서버 IP 제거(선택)
    if "user_ip" in d.columns:
        mask_server = d["user_ip"].apply(_is_media_server_ip).fillna(False)
        d = d.loc[~mask_server].copy()

    # Device 피처
    if "user_ip" in d.columns:
        dev_ip = d[["dvc_idx","user_ip"]].drop_duplicates()
        dev_ip_cnt = dev_ip.groupby("dvc_idx", observed=True)["user_ip"].nunique().rename("ip_count")
    else:
        dev_ip_cnt = d.groupby("dvc_idx", observed=True).size().rename("ip_count")
    dev_b1h = (d[["dvc_idx","b1h","user_ip"]].drop_duplicates()
               if "user_ip" in d.columns else d[["dvc_idx","b1h"]].drop_duplicates().assign(user_ip="na"))
    ip_change_speed = (dev_b1h.groupby(["dvc_idx","b1h"], observed=True)["user_ip"].nunique()
                       .groupby("dvc_idx", observed=True).quantile(P99_Q).rename("ip_change_speed"))
    daily_actions_p95 = (d.groupby(["dvc_idx","day"], observed=True).size()
                         .groupby("dvc_idx", observed=True).quantile(P95_Q).rename("daily_actions_p95"))
    active_days_dev = d.groupby("dvc_idx", observed=True)["day"].nunique().rename("active_days_device")
    dev_feat = pd.concat([dev_ip_cnt, ip_change_speed, daily_actions_p95, active_days_dev], axis=1).fillna(0)

    # IP 피처
    ip_feat = None
    if "user_ip" in d.columns and len(d):
        ip_dev_cnt = d.groupby("user_ip", observed=True)["dvc_idx"].nunique().rename("device_count")
        ip_dev_day = d[["user_ip","dvc_idx","day"]].drop_duplicates()
        device_repeat_rate_ip = (ip_dev_day.groupby(["user_ip","dvc_idx"], observed=True)["day"].nunique()
                                 .groupby("user_ip", observed=True).mean().rename("device_repeat_rate_ip"))
        burst5_ip = (d[["user_ip","b5","dvc_idx"]].drop_duplicates()
                     .groupby(["user_ip","b5"], observed=True)["dvc_idx"].nunique()
                     .groupby("user_ip", observed=True).quantile(P95_Q).rename("burst5_ip"))
        active_days_ip = d.groupby("user_ip", observed=True)["day"].nunique().rename("active_days_ip")

        if "ads_idx" in d.columns:
            cnt = (d[["user_ip","ads_idx","b10","dvc_idx"]].drop_duplicates()
                   .groupby(["user_ip","ads_idx","b10"], observed=True)["dvc_idx"].nunique())
            ads_overlap_rate    = (cnt >= 2).astype("int8").groupby("user_ip", observed=True).mean().rename("ads_overlap_rate")
            ads_overlap_windows = cnt.groupby("user_ip", observed=True).size().rename("ads_overlap_windows")
        else:
            ads_overlap_rate    = pd.Series(0.0, index=ip_dev_cnt.index, name="ads_overlap_rate")
            ads_overlap_windows = pd.Series(0,   index=ip_dev_cnt.index, name="ads_overlap_windows")

        ip_feat = pd.concat([ip_dev_cnt, device_repeat_rate_ip, burst5_ip,
                             ads_overlap_rate, ads_overlap_windows, active_days_ip], axis=1).fillna(0)
    return ip_feat, dev_feat

def _ensure_scores(ip_feat: pd.DataFrame | None, dev_feat: pd.DataFrame):
    # IP 점수
    if ip_feat is not None:
        ipf = ip_feat.copy()
        if len(ipf)==0:
            ipf["z_fanout"]=[]; ipf["score"]=[]
        else:
            ipf["z_fanout"] = (ipf["device_count"]-ipf["device_count"].mean())/(ipf["device_count"].std(ddof=0)+1e-9)
            ipf["score"] = (
                3.0*ipf["z_fanout"]
                + 2.0*(ipf.get("burst5_ip",0) >= ipf.get("burst5_ip",pd.Series([0])).quantile(P999_Q)).astype(int)
                + 1.5*(ipf.get("ads_overlap_rate",0) >= ipf.get("ads_overlap_rate",pd.Series([0])).quantile(P99_Q)).astype(int)
                - 2.0*(ipf.get("device_repeat_rate_ip", np.inf) <= 1.1).astype(int)
                - 1.5*(((ipf.get("active_days_ip",0) >= 7) & (ipf.get("device_count",0) <= 3)).astype(int))
            )
    else:
        ipf = None
    # Device 점수
    devf = dev_feat.copy()
    if len(devf)==0:
        devf["z_fanin"]=[]; devf["score"]=[]
    else:
        devf["z_fanin"] = (devf["ip_count"]-devf["ip_count"].mean())/(devf["ip_count"].std(ddof=0)+1e-9)
        devf["score"] = (
            3.0*devf["z_fanin"]
            + 1.0*(devf.get("ip_change_speed",0) >= devf.get("ip_change_speed",pd.Series([0])).quantile(P999_Q)).astype(int)
            - 1.0*(devf.get("daily_actions_p95",0) <= devf.get("daily_actions_p95",pd.Series([0])).quantile(P99_Q)).astype(int)
            - 0.8*((devf.get("active_days_device",0) >= 7).astype(int))
        )
    return ipf, devf

# 탐지(라벨링) =======================================================================

def detect_fanout_fanin(df_join: pd.DataFrame):
    """
    입력: df_join(원본 참여 로그)
    반환: (ip_lab, dev_lab) — '의심/확정'만 포함한 요약 테이블
    """
    cols = [c for c in ["user_ip","dvc_idx","ads_idx","mda_idx","pub_sub_rel_id"]+TIME_COLS if c in df_join.columns]
    _check_cols(df_join, cols, "df_join")  # 최소 하나라도 없으면 여기서 에러로 확인

    d = _prep_time(df_join[cols])
    if len(d)==0:
        return pd.DataFrame(), pd.DataFrame()

    # IP 품질 진단 → auto 모드 결정
    grp_key, ipq = _ip_quality(d)
    ip_reliable_rate = float((~ipq["ip_unreliable"]).mean()) if len(ipq) else 0.0
    if SWITCH_MODE == "force_ip":
        use_ip = ("user_ip" in d.columns)
    elif SWITCH_MODE == "force_device":
        use_ip = False
    else:
        use_ip = ("user_ip" in d.columns) and (ip_reliable_rate > 0.95)

    # IP 양호 그룹만 사용
    if use_ip and grp_key is not None:
        good = set(ipq.loc[~ipq["ip_unreliable"], "group_key"])
        d_ip = d[d[grp_key].isin(good)]
    else:
        d_ip = None

    ip_feat, dev_feat = _compute_features(d if d_ip is None else d_ip)
    if not use_ip:
        ip_feat = None

    ipf, devf = _ensure_scores(ip_feat, dev_feat)

    # 라벨링
    # IP
    if ipf is not None:
        ip = ipf.copy()
        if len(ip):
            q_ip_dev_999  = ip["device_count"].quantile(P999_Q)
            q_ip_burst_999= ip["burst5_ip"].quantile(P999_Q) if "burst5_ip" in ip else np.nan
            ads_rate = ip["ads_overlap_rate"].where(ip.get("ads_overlap_windows",0) >= ADS_OVERLAP_MIN_WINDOWS, 0.0)
            c1 = (ip["z_fanout"]>=Z_GATE) | (ip["device_count"]>=q_ip_dev_999)
            c2 = (ads_rate >= 0.50)
            c3 = (ip.get("burst5_ip",0) >= max(q_ip_burst_999 if pd.notna(q_ip_burst_999) else 5, 5))
            c4 = (ip.get("device_repeat_rate_ip", np.inf) <= 1.1) & (ip.get("active_days_ip", np.inf) <= IP_REPEAT_STRICT_DAYS)
            hits = (c1.astype(int)+c2.astype(int)+c3.astype(int))
            ip["final_label"] = "정상"
            ip.loc[(hits>=CONFIRM_HITS_MIN) | (ip["score"]>=CONFIRM_SCORE_TH) | (c4 & (c1|c2|c3)), "final_label"] = "확정"
            ip.loc[(ip["final_label"]!="확정") & ((hits>=SUSPECT_HITS_MIN) | (ip["score"]>=SUSPECT_SCORE_TH)), "final_label"] = "의심"
            ip_lab = (ip.reset_index().rename(columns={"index":"user_ip"}))
            ip_lab = ip_lab[ip_lab["final_label"].isin(["의심","확정"])]
        else:
            ip_lab = pd.DataFrame(columns=["user_ip","final_label"])
    else:
        ip_lab = pd.DataFrame(columns=["user_ip","final_label"])

    # Device
    dev = devf.copy()
    if len(dev):
        q_dev_ip_999 = dev["ip_count"].quantile(P999_Q)
        d1 = (dev["z_fanin"]>=Z_GATE) | (dev["ip_count"]>=q_dev_ip_999)
        d2 = (dev.get("ip_change_speed",0) >= max(dev.get("ip_change_speed",pd.Series([0])).quantile(P999_Q), 5))
        d3 = (dev.get("active_days_device", np.inf) <= DEV_ACTIVE_DAYS_MAX)
        hits = (d1.astype(int)+d2.astype(int))
        dev["final_label"] = "정상"
        dev.loc[(hits>=CONFIRM_HITS_MIN) | (dev["score"]>=CONFIRM_SCORE_TH) | (d3 & (d1|d2)), "final_label"] = "확정"
        dev.loc[(dev["final_label"]!="확정") & ((hits>=SUSPECT_HITS_MIN) | (dev["score"]>=SUSPECT_SCORE_TH)), "final_label"] = "의심"
        dev_lab = (dev.reset_index().rename(columns={"index":"dvc_idx"}))
        dev_lab = dev_lab[dev_lab["final_label"].isin(["의심","확정"])]
    else:
        dev_lab = pd.DataFrame(columns=["dvc_idx","final_label"])

    return ip_lab[["user_ip","final_label"]], dev_lab[["dvc_idx","final_label"]]

# 라벨 전파(행수 보존, 1/2 캡핑 가능) =================================================

def apply_abuse8_labels(
    df_join: pd.DataFrame,
    ip_lab: pd.DataFrame,
    dev_lab: pd.DataFrame,
    *,
    require_both_for_2: bool = True,   # True: 2는 IP=2 & Dev=2 동시에일 때만
    cap_flag_share: float | None = 0.10,   # 최종 (1+2) 상한(전체 비율). None이면 캡핑 안함
    min_one_share_of_flagged: float = 0.20, # 플래그 중 최소 이 비율은 1로 유지
    w_ip: float = 1.0, w_dev: float = 1.0   # 1 다운샘플 시 우선순위 가중치
):
    """
    반환: df_join_v1 (원본 컬럼 + abuse_8(int8)), stats(dict)
    - 행수 보존
    - 중간 컬럼 제거
    - 캡핑 시 2는 보호, 1만 줄임(1이 0이 되더라도 최소 일부 1 유지)
    """
    j = df_join.copy()

    # 매핑(다대일 → 행수 불변)
    sev_map = {"정상":0, "의심":1, "확정":2}

    if ("user_ip" in j.columns) and (ip_lab is not None) and len(ip_lab):
        ip_map = ip_lab.set_index("user_ip")["final_label"].map(sev_map)
        j["_sev_ip"] = j["user_ip"].map(ip_map).fillna(0).astype("int8")
    else:
        j["_sev_ip"] = 0

    if ("dvc_idx" in j.columns) and (dev_lab is not None) and len(dev_lab):
        dev_map = dev_lab.set_index("dvc_idx")["final_label"].map(sev_map)
        j["_sev_dev"] = j["dvc_idx"].map(dev_map).fillna(0).astype("int8")
    else:
        j["_sev_dev"] = 0

    # 기본 라벨
    if require_both_for_2:
        base2 = j["_sev_ip"].eq(2) & j["_sev_dev"].eq(2)
    else:
        base2 = (j[["_sev_ip","_sev_dev"]].max(axis=1) == 2)
    base1 = (j[["_sev_ip","_sev_dev"]].max(axis=1) > 0) & (~base2)

    lab = np.where(base2, 2, np.where(base1, 1, 0)).astype("int8")

    # (선택) 캡핑: 2 보호, 1만 축소
    if cap_flag_share is not None:
        N = len(j)
        target = int(np.floor(cap_flag_share * N))

        idx2 = np.where(lab == 2)[0]
        idx1 = np.where(lab == 1)[0]
        keep2 = set(idx2.tolist())

        # 이미 2가 타깃 초과면 → 전부 2 유지, 1은 모두 0
        if len(keep2) >= target:
            new_lab = np.zeros(N, dtype="int8")
            new_lab[list(keep2)] = 2
            lab = new_lab
        else:
            remain = target - len(keep2)
            # 1에서 우선순위 점수(강한 신호 먼저 유지)
            score1 = (w_ip*j["_sev_ip"].values + w_dev*j["_sev_dev"].values).astype(float)
            # 1의 인덱스만 추출 후 상위 remain 유지
            rank = np.argsort(-score1[idx1], kind="mergesort")
            keep1 = set(idx1[rank[:max(0, remain)]].tolist())

            # 그래도 1이 전부 사라지는 걸 방지(최소 유지)
            min1 = int(np.floor(min_one_share_of_flagged * max(target,1)))
            if len(keep1) == 0 and len(idx1) > 0 and min1 > 0:
                # 랜덤이 싫으면 시간순 등으로 일부 고정 유지
                keep1 = set(idx1[:min(len(idx1), min1)].tolist())

            new_lab = np.zeros(N, dtype="int8")
            if keep2: new_lab[list(keep2)] = 2
            if keep1: new_lab[list(keep1)] = np.maximum(new_lab[list(keep1)], 1)
            lab = new_lab

    j["abuse_8"] = lab.astype("int8")
    # 중간 컬럼 제거
    j.drop(columns=["_sev_ip","_sev_dev"], inplace=True, errors="ignore")

    # 통계
    vc = j["abuse_8"].value_counts().sort_index()
    total = int(len(j))
    stats = {
        "total_rows": total,
        "label_dist": {int(k): int(v) for k,v in vc.items()},
        "abuse_rate": float(100.0 * j["abuse_8"].gt(0).mean())
    }
    return j, stats

# 결과
ip_lab, dev_lab = detect_fanout_fanin(df_join)

# 2) 원본에 abuse_8 라벨 전파 (행수 그대로, 2는 보호, 1만 캡핑)
df_join_v1, stats = apply_abuse8_labels(
    df_join_v1, ip_lab, dev_lab,
    require_both_for_2=True,     # 확정은 IP=2 & Dev=2 동시일 때만
    cap_flag_share=0.10,         # (1+2) 최종 캡 10%
    min_one_share_of_flagged=0.20,# 그 중 20%는 1로 남김(1이 증발하지 않게)
    w_ip=1.0, w_dev=1.0
)

# 저장
df_join_v1.to_csv("df_join_v1.csv", index=False)