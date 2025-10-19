# 정상 / 어뷰징 데이터 분류
def split_by_labels(df, label_cols):
    cols = [c for c in label_cols if c in df.columns]
    if not cols:
        return df.copy(), df.iloc[0:0].copy()
    arr = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols}).fillna(0).astype("int8")
    mask_abuse = arr.gt(0).any(axis=1)
    return df.loc[~mask_abuse].copy(), df.loc[mask_abuse].copy()

# 라벨 컬럼 정의
join_labels   = ["abuse_1","abuse_2","abuse_3","abuse_4","abuse_5","abuse_8","abuse_9","abuse_10"]
settle_labels = ["abuse_2","abuse_4","abuse_6","abuse_7","abuse_10"]
rpt_labels    = ["abuse_3","abuse_4","abuse_7"]

# 분리 (변수에만 저장)
df_join_clean,   df_join_abuse   = split_by_labels(df_join_v1,   join_labels)
df_settle_clean, df_settle_abuse = split_by_labels(df_settle_v1, settle_labels)
df_rpt_clean,    df_rpt_abuse    = split_by_labels(df_rpt_v1,    rpt_labels)

# 분류된 데이터 저장
df_join_clean.to_csv("qqqq/df_join_clean.csv", encoding="utf-8-sig", index=False)
df_join_abuse.to_csv("qqqq/df_join_abuse.csv", encoding="utf-8-sig", index=False)
df_settle_clean.to_csv("qqqq/df_settle_clean.csv", encoding="utf-8-sig", index=False)
df_settle_abuse.to_csv("qqqq/df_settle_abuse.csv", encoding="utf-8-sig", index=False)
df_rpt_clean.to_csv("qqqq/df_rpt_clean.csv", encoding="utf-8-sig", index=False)
df_rpt_abuse.to_csv("qqqq/df_rpt_abuse.csv", encoding="utf-8-sig", index=False)