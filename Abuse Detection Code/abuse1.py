import pandas as pd, numpy as np, os, uuid
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype as is_dt
from math import erfc, sqrt
import scipy.special
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
# =========================================
df_list   = pd.read_csv("csv_output/1_IVE_광고목록.csv")
df_join   = pd.read_csv("csv_output/3_IVE_광고참여정보.csv")
df_settle = pd.read_csv("csv_output/2_IVE_광고적립.csv")
df_rpt    = pd.read_csv("csv_output/아이브1년치_참여데이터.csv", index_col=0)

# ===== 전역 파라미터 =====
REJOIN_UNIT = "ads_code"       # 'ads_code' 또는 'ads_idx' (집계 단위)
USE_CLICK_KEY = True           # click_key 고유 개수로 카운트할지 여부
WEB_IP_POLICY = "guarded"      # 'guarded'|'use'|'ignore'

# ---- 과다시도 임계치(전역) ----
OVERCLICK_WINDOW_DAYS = None   # None=전체 기간, 예: 30 → 최근 30일만
SUS_ATTEMPTS_TH     = 40       # 의심 임계치(≥) z-score 2시그마
CONF_ATTEMPTS_TH    = 57       # 확정 임계치(≥)  z-score 3시그마 ※ SUS ≤ CONF 권장

# ---- IP 품질 판정 기준 ----
IP_TOP_SHARE_TH = 0.95
IP_UNIQUE_RATIO_TH = 0.001
MIN_ROWS_FOR_IP_CHECK = 200

# ---- IP 품질 그룹핑 키 우선순위 ----
GROUP_KEY_ORDER = ("mda_idx", "pub_sub_rel_id")

# ================= 유틸 =================
def _pick_group_key(df: pd.DataFrame) -> str | None:
    for k in GROUP_KEY_ORDER:
        if k in df.columns: return k
    return None

def _ip_bad_groups(df: pd.DataFrame, key: str | None) -> set:
    """서버 IP 의심 그룹 set 반환. 'user_ip'가 없으면 모든 그룹을 bad로 간주(-1)."""
    if "user_ip" not in df.columns:
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

def _build_user_id_with_guard(j: pd.DataFrame) -> pd.Series:
    """디바이스 우선, 웹은 정책(WEB_IP_POLICY)과 IP 품질에 따라 user_id 구성."""
    dvc = pd.to_numeric(j["dvc_idx"], errors="coerce").fillna(0).astype("Int64")
    ip  = j["user_ip"].astype(str).str.strip() if "user_ip" in j.columns else pd.Series([""], index=j.index)

    if WEB_IP_POLICY == "ignore":
        return np.where(dvc.ne(0), "dvc:"+dvc.astype(str), np.nan)

    key = _pick_group_key(j)
    bad = _ip_bad_groups(j, key)

    if key is None:
        web_ok = "user_ip" in j.columns and ((-1) not in bad)
        return np.where(
            dvc.ne(0), "dvc:"+dvc.astype(str),
            np.where((WEB_IP_POLICY=="guarded") & (~web_ok), np.nan, "ip:"+ip)
        )

    keyvals = j[key]
    in_bad = keyvals.astype("Int64").isin(bad) if pd.api.types.is_integer_dtype(keyvals) else keyvals.astype(str).isin(bad)

    return np.where(
        dvc.ne(0), "dvc:"+dvc.astype(str),
        np.where((WEB_IP_POLICY=="guarded") & in_bad, np.nan, "ip:"+ip)
    )

# ============== 메인: 단일 함수 ==============
def detect_overclick(df_list: pd.DataFrame, df_join: pd.DataFrame) -> pd.DataFrame:
    """
    하루 구분 없이 '과다 시도'를 한 번에 집계.
    전역 임계치:
      - SUS_ATTEMPTS_TH (의심), CONF_ATTEMPTS_TH (확정)
      - OVERCLICK_WINDOW_DAYS 기간 제한(옵션)
    반환: flag>0(의심/확정)만 포함한 DataFrame
      [abuse_flag(1/2), abuse_type, user_id, <REJOIN_UNIT>, cnt, first_click, last_click]
    """
    # 임계치 정합성 보정
    global SUS_ATTEMPTS_TH, CONF_ATTEMPTS_TH
    if (CONF_ATTEMPTS_TH is not None) and (SUS_ATTEMPTS_TH is not None) and (CONF_ATTEMPTS_TH < SUS_ATTEMPTS_TH):
        SUS_ATTEMPTS_TH = min(SUS_ATTEMPTS_TH, CONF_ATTEMPTS_TH)

    # 0) REJOIN_UNIT 확보(ads_code면 df_list에서 붙임)
    need = [c for c in ("ads_idx","user_ip","dvc_idx","click_date","click_key","mda_idx","pub_sub_rel_id") if c in df_join.columns]
    j = df_join[need].copy()

    if REJOIN_UNIT == "ads_code":
        if "ads_idx" not in j.columns or "ads_code" not in df_list.columns:
            raise KeyError("REJOIN_UNIT='ads_code' 사용 시 df_list에 'ads_code'가 필요합니다.")
        meta = df_list[["ads_idx","ads_code"]].drop_duplicates()
        j = j.merge(meta, on="ads_idx", how="inner", validate="m:1")

    # 1) 날짜·user_id 정규화
    j["click_date"] = pd.to_datetime(j["click_date"], errors="coerce")
    j = j[j["click_date"].notna()].copy()
    j["user_id"] = _build_user_id_with_guard(j)
    j = j[j["user_id"].notna()].copy()

    # 2) 기간 제한(옵션)
    if OVERCLICK_WINDOW_DAYS is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(OVERCLICK_WINDOW_DAYS))
        j = j[j["click_date"] >= cutoff]

    # 3) 집계: user_id × REJOIN_UNIT
    grp_key = REJOIN_UNIT if REJOIN_UNIT in j.columns else "ads_idx"
    if USE_CLICK_KEY and ("click_key" in j.columns):
        g = (j.groupby(["user_id", grp_key], as_index=False)
               .agg(cnt=("click_key","nunique"),
                    first_click=("click_date","min"),
                    last_click=("click_date","max")))
    else:
        g = (j.groupby(["user_id", grp_key], as_index=False)
               .agg(cnt=("click_date","size"),
                    first_click=("click_date","min"),
                    last_click=("click_date","max")))

    # 4) 의심/확정 플래그 부여(전역 임계치)
    g["abuse_flag"] = 0
    if SUS_ATTEMPTS_TH is not None:
        g.loc[g["cnt"] >= int(SUS_ATTEMPTS_TH), "abuse_flag"] = 1
    if CONF_ATTEMPTS_TH is not None:
        g.loc[g["cnt"] >= int(CONF_ATTEMPTS_TH), "abuse_flag"] = 2

    # 5) 결과 정리(의심/확정만 노출)
    out = g[g["abuse_flag"] > 0].copy()
    out["abuse_type"] = np.where(out["abuse_flag"]==2,
                                 "확정",
                                 "의심")
    cols = ["abuse_flag","abuse_type","user_id", grp_key, "cnt", "first_click", "last_click"]
    out = out[cols].sort_values(["abuse_flag","cnt","last_click"], ascending=[False, False, True]).reset_index(drop=True)
    return out

# 결과
over = detect_overclick(df_list, df_join)

# 데이터 분리
def _ensure_unit_col(df_list, df_join):
    """REJOIN_UNIT이 ads_code면 join에 ads_code를 붙여 사용키 통일."""
    j = df_join.copy()
    if REJOIN_UNIT == "ads_code":
        meta = df_list[["ads_idx","ads_code"]].drop_duplicates()
        j = j.merge(meta, on="ads_idx", how="left", validate="m:1")
    return j

def apply_abuse_to_join(df_list, df_join, over):
    """df_join에 abuse_1(2/1/0) 부여 → df_join_v1 반환"""
    j = _ensure_unit_col(df_list, df_join)
    j["user_id"] = _build_user_id_with_guard(j)
    key = REJOIN_UNIT if REJOIN_UNIT in j.columns else "ads_idx"

    flags = over[["user_id", key, "abuse_flag"]].drop_duplicates()
    j = j.merge(flags, on=["user_id", key], how="left")
    j["abuse_1"] = j["abuse_flag"].fillna(0).astype("int8")
    j = j.drop(columns=["abuse_flag","user_id"], errors="ignore")
    return j

def build_list_with_rate(df_list, df_join_v1):
    """
    광고목록(df_list)에 광고별 어뷰징 '비율' 컬럼 abuse_1 추가(0.0~1.0).
    분자: df_join_v1에서 abuse_1>0 인 행(의심+확정), 분모: 해당 ads_idx 전체 참여수.
    """
    tmp = df_join_v1[["ads_idx","abuse_1"]].copy()
    tmp["is_abuse"] = (tmp["abuse_1"] > 0).astype("int8")
    rate = tmp.groupby("ads_idx", observed=True)["is_abuse"].mean().rename("abuse_1")
    out = df_list.merge(rate, on="ads_idx", how="left")
    out["abuse_1"] = out["abuse_1"].fillna(0).astype("float32")
    return out

# ===== 실행 =====
df_join_v1 = apply_abuse_to_join(df_list, df_join, over)   # 행 단위 플래그(2/1/0)
df_list_v1 = build_list_with_rate(df_list, df_join_v1)     # 광고별 어뷰징 비율(0.0~1.0)

# ===== 저장 =====
df_join_v1.to_csv("df_join_v1.csv", index=False, encoding="utf-8-sig")
df_list_v1.to_csv("df_list_v1.csv", index=False, encoding="utf-8-sig")