# ===== 전역 파라미터 (Global Parameters) =====

# 기본 분석 설정
MIN_PUBLISHER_CLICKS = 1000         # 통계적 유의성을 위한 최소 샘플 크기
MEGA_PUBLISHER_THRESHOLD = 1_000_000 # 거대 퍼블리셔 기준 (1백만 클릭 이상)

# === 의심 단계 (Suspicious) 임계값 ===
# 학술 연구 기반: 정상 사용자 패턴 상위 5% 수준
SUSPICIOUS_TIME_CONCENTRATION = 0.6     # 시간 집중도 60% (정상: 20-40%)
SUSPICIOUS_DEVICE_PER_CLICK = 50        # 디바이스당 50클릭 (정상: 5-20클릭)
SUSPICIOUS_IP_PER_CLICK = 30            # IP당 30클릭 (정상: 5-15클릭)
SUSPICIOUS_SINGLE_DEVICE_SHARE = 0.3    # 단일 디바이스 30% 점유 (정상: 5-15%)
SUSPICIOUS_SINGLE_IP_SHARE = 0.2        # 단일 IP 20% 점유 (정상: 2-10%)

# === 확정 단계 (Confirmed) 임계값 ===
# 업계 표준: 명백한 어뷰징으로 간주되는 수준
CONFIRMED_TIME_CONCENTRATION = 0.8      # 시간 집중도 80% (비정상적 집중)
CONFIRMED_DEVICE_PER_CLICK = 500        # 디바이스당 500클릭 (명백한 Bot 패턴)  
CONFIRMED_IP_PER_CLICK = 300            # IP당 300클릭 (자동화 도구 의심)
CONFIRMED_SINGLE_DEVICE_SHARE = 0.7     # 단일 디바이스 70% 점유 (Bot Farm)
CONFIRMED_SINGLE_IP_SHARE = 0.5         # 단일 IP 50% 점유 (Proxy/VPN 의심)

# === 거대 퍼블리셔 특별 기준 ===
MEGA_DEVICE_PER_CLICK_SUSPICIOUS = 200  # 거대 퍼블리셔 디바이스당 200클릭
MEGA_DEVICE_PER_CLICK_CONFIRMED = 1000  # 거대 퍼블리셔 디바이스당 1000클릭
MEGA_IP_PER_CLICK_SUSPICIOUS = 100      # 거대 퍼블리셔 IP당 100클릭  
MEGA_IP_PER_CLICK_CONFIRMED = 500       # 거대 퍼블리셔 IP당 500클릭

# IP 품질 검사 패턴
MEDIA_SERVER_PATTERNS = [
    r'^43\.203\.', r'^3\.38\.', r'^16\.184\.', r'^54\.180\.', 
    r'^15\.165\.', r'^13\.125\.', r'^52\.79\.', r'^34\.64\.', r'^175\.126\.'
]

# 개별 의심 사례 기준
INDIVIDUAL_DEVICE_THRESHOLD = 1000      # 디바이스당 1000클릭 이상
INDIVIDUAL_IP_THRESHOLD = 500           # IP당 500클릭 이상

# 라벨링 파라미터
SEVERITY_MAPPING = {"정상": 0, "의심": 1, "확정": 2}
DEFAULT_CAP_ABUSE_RATE = None     # 필요시 예: 0.10
MIN_SUSPICIOUS_RATIO   = 0.20

# ===== 유틸 =====
def _check_required_columns(df, req, name="DataFrame"):
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"{name} 필수 컬럼 누락: {miss}")

def _is_media_server_ip(ip):
    if pd.isna(ip): return False
    s = str(ip).strip()
    return any(re.match(p, s) for p in MEDIA_SERVER_PATTERNS)

def _is_private_ip(ip):
    if pd.isna(ip): return False
    s = str(ip).strip()
    return any(re.match(p, s) for p in [r'^10\.', r'^192\.168\.', r'^172\.(1[6-9]|2[0-9]|3[01])\.'])

def _is_valid_ip_for_analysis(ip):
    if pd.isna(ip): return False
    return not (_is_media_server_ip(ip) or _is_private_ip(ip))

def _preprocess_data(df):
    req = ['click_date','mda_idx','pub_sub_rel_id','dvc_idx','user_ip']
    _check_required_columns(df, req, "Input")
    x = df.copy()
    x['click_datetime'] = pd.to_datetime(x['click_date'], errors='coerce')
    x['hour'] = x['click_datetime'].dt.hour
    x['date'] = x['click_datetime'].dt.date
    for c in ['mda_idx','pub_sub_rel_id','dvc_idx']:
        if c in x.columns: x[c] = pd.to_numeric(x[c], errors='coerce')
    return x

# ===== 패턴 분석 =====
def _analyze_temporal_concentration(pub):
    mx = pub['hour'].value_counts(normalize=True).max()
    if mx >= CONFIRMED_TIME_CONCENTRATION: return "확정", f"시간 집중도 {mx*100:.1f}%"
    if mx >= SUSPICIOUS_TIME_CONCENTRATION: return "의심", f"시간 집중도 {mx*100:.1f}%"
    return "정상", None

def _analyze_device_concentration(pub, is_mega=False):
    app = pub[pub['dvc_idx']!=0]
    if len(app)<=100: return "정상", []
    cnts = app['dvc_idx'].value_counts()
    avg = len(app)/len(cnts)
    top = (cnts.iloc[0]/len(app)) if len(cnts)>0 else 0
    conf_th = MEGA_DEVICE_PER_CLICK_CONFIRMED if is_mega else CONFIRMED_DEVICE_PER_CLICK
    sus_th  = MEGA_DEVICE_PER_CLICK_SUSPICIOUS if is_mega else SUSPICIOUS_DEVICE_PER_CLICK
    reasons, sev = [], "정상"
    if avg>=conf_th: sev="확정"; reasons.append(f"디바이스당 {avg:.0f}클릭")
    elif avg>=sus_th: sev="의심"; reasons.append(f"디바이스당 {avg:.0f}클릭")
    if top>=CONFIRMED_SINGLE_DEVICE_SHARE: sev="확정"; reasons.append(f"단일 디바이스 {top*100:.1f}%")
    elif top>=SUSPICIOUS_SINGLE_DEVICE_SHARE and sev=="정상":
        sev="의심"; reasons.append(f"단일 디바이스 {top*100:.1f}%")
    return sev, reasons

def _analyze_ip_concentration(pub, is_mega=False):
    web = pub[pub['dvc_idx']==0]
    if len(web)<=100: return "정상", []
    vw = web[web['user_ip'].apply(_is_valid_ip_for_analysis)]
    if len(vw)<=50: return "정상", []
    cnts = vw['user_ip'].value_counts()
    avg = len(vw)/len(cnts)
    top = (cnts.iloc[0]/len(vw)) if len(cnts)>0 else 0
    conf_th = MEGA_IP_PER_CLICK_CONFIRMED if is_mega else CONFIRMED_IP_PER_CLICK
    sus_th  = MEGA_IP_PER_CLICK_SUSPICIOUS if is_mega else SUSPICIOUS_IP_PER_CLICK
    reasons, sev = [], "정상"
    if avg>=conf_th: sev="확정"; reasons.append(f"IP당 {avg:.0f}클릭")
    elif avg>=sus_th: sev="의심"; reasons.append(f"IP당 {avg:.0f}클릭")
    if top>=CONFIRMED_SINGLE_IP_SHARE: sev="확정"; reasons.append(f"단일 IP {top*100:.1f}%")
    elif top>=SUSPICIOUS_SINGLE_IP_SHARE and sev=="정상":
        sev="의심"; reasons.append(f"단일 IP {top*100:.1f}%")
    return sev, reasons

def _determine_final_severity(sevs):
    return "확정" if "확정" in sevs else ("의심" if "의심" in sevs else "정상")

# ===== 탐지 Core =====
def detect_publisher_abuse_patterns(df):
    d = _preprocess_data(df)
    out = []
    for (mda, pub), g in d.groupby(['mda_idx','pub_sub_rel_id']):
        if len(g)<MIN_PUBLISHER_CLICKS: continue
        mega = len(g)>=MEGA_PUBLISHER_THRESHOLD
        t_sev, t_r = _analyze_temporal_concentration(g)
        d_sev, d_rs= _analyze_device_concentration(g, mega)
        i_sev, i_rs= _analyze_ip_concentration(g, mega)
        final = _determine_final_severity([t_sev,d_sev,i_sev])
        reasons=[]
        if t_r: reasons.append(t_r)
        reasons+=d_rs+i_rs
        if final!="정상" and reasons:
            out.append({
                'mda_idx':mda,'pub_sub_rel_id':pub,'total_clicks':len(g),
                'severity':final,'reasons':'; '.join(reasons),
                'is_mega_publisher':mega,
                'app_clicks':int((g['dvc_idx']!=0).sum()),
                'web_clicks':int((g['dvc_idx']==0).sum())
            })
    return pd.DataFrame(out)

def detect_individual_suspicious_cases(df):
    cases=[]
    app=df[df['dvc_idx']!=0]
    if len(app)>0:
        vc=app['dvc_idx'].value_counts()
        ext=vc[vc>=INDIVIDUAL_DEVICE_THRESHOLD]
        for dev,clk in ext.items():
            sub=app[app['dvc_idx']==dev]
            nun=sub[['mda_idx','pub_sub_rel_id']].nunique()
            cases.append({'type':'Device','entity':dev,'clicks':int(clk),
                          'pub_diversity':f"{nun['mda_idx']}매체/{nun['pub_sub_rel_id']}퍼블리셔"})
    web=df[df['dvc_idx']==0]
    if len(web)>0:
        vw=web[web['user_ip'].apply(_is_valid_ip_for_analysis)]
        if len(vw)>0:
            vc=vw['user_ip'].value_counts()
            ext=vc[vc>=INDIVIDUAL_IP_THRESHOLD]
            for ip,clk in ext.items():
                sub=vw[vw['user_ip']==ip]
                nun=sub[['mda_idx','pub_sub_rel_id']].nunique()
                cases.append({'type':'IP','entity':ip,'clicks':int(clk),
                              'pub_diversity':f"{nun['mda_idx']}매체/{nun['pub_sub_rel_id']}퍼블리셔"})
    return cases

def detect_abuse_main(df):
    abuse_results = detect_publisher_abuse_patterns(df)
    individual_cases = detect_individual_suspicious_cases(df)
    return abuse_results, individual_cases

# ===== 라벨링 Core =====
def _validate_inputs(df_original, abuse_results):
    _check_required_columns(df_original, ['mda_idx','pub_sub_rel_id'], "원본")
    if len(abuse_results)>0:
        _check_required_columns(abuse_results, ['mda_idx','pub_sub_rel_id','severity'], "어뷰징결과")

def _prepare_publisher_mapping(abuse_results):
    if len(abuse_results)==0:
        return pd.DataFrame(columns=['mda_idx','pub_sub_rel_id','severity_code'])
    m=abuse_results[['mda_idx','pub_sub_rel_id','severity']].copy()
    m['severity_code']=m['severity'].map(SEVERITY_MAPPING)
    return m.groupby(['mda_idx','pub_sub_rel_id'])['severity_code'].max().reset_index()

def _apply_basic_labels(df_original, mapping_df):
    x=df_original.copy()
    x['mda_idx']=pd.to_numeric(x['mda_idx'],errors='coerce')
    x['pub_sub_rel_id']=pd.to_numeric(x['pub_sub_rel_id'],errors='coerce')
    if len(mapping_df)==0:
        x['abuse_9']=0
    else:
        m=mapping_df.copy()
        m['mda_idx']=pd.to_numeric(m['mda_idx'],errors='coerce')
        m['pub_sub_rel_id']=pd.to_numeric(m['pub_sub_rel_id'],errors='coerce')
        x=x.merge(m[['mda_idx','pub_sub_rel_id','severity_code']], on=['mda_idx','pub_sub_rel_id'], how='left')
        x['abuse_9']=x['severity_code'].fillna(0).astype('int8')
        x.drop(columns=['severity_code'], inplace=True)
    return x

def _apply_capping(df_labeled, cap_abuse_rate=None, min_suspicious_ratio=MIN_SUSPICIOUS_RATIO):
    if cap_abuse_rate is None: return df_labeled
    n=len(df_labeled); target=int(np.floor(cap_abuse_rate*n))
    vc=df_labeled['abuse_9'].value_counts().sort_index()
    cur=vc.get(1,0)+vc.get(2,0)
    if cur<=target: return df_labeled
    y=df_labeled.copy()
    conf_idx=y[y['abuse_9']==2].index
    susp_idx=y[y['abuse_9']==1].index
    if len(conf_idx)>=target:
        y.loc[susp_idx,'abuse_9']=0
    else:
        remain=target-len(conf_idx)
        keep_min=int(np.ceil(min_suspicious_ratio*target))
        keep=min(max(keep_min,0), min(remain, len(susp_idx)))
        if keep<len(susp_idx):
            keep_ids=np.random.choice(susp_idx, size=keep, replace=False)
            drop_ids=susp_idx[~np.isin(susp_idx, keep_ids)]
            y.loc[drop_ids,'abuse_9']=0
    return y

def _calculate_statistics(df_labeled, abuse_results):
    n=len(df_labeled); vc=df_labeled['abuse_9'].value_counts().sort_index()
    s=vc.get(1,0); c=vc.get(2,0); z=vc.get(0,0)
    return {
        'total_rows': n,
        'normal_count': int(z),
        'suspicious_count': int(s),
        'confirmed_count': int(c),
        'abuse_rate_percent': round((s+c)/n*100 if n else 0, 3),
        'detected_publishers': int(len(abuse_results))
    }

def apply_abuse_labels(df_original, abuse_results, *, cap_abuse_rate=None, min_suspicious_ratio=MIN_SUSPICIOUS_RATIO):
    _validate_inputs(df_original, abuse_results)
    mapping=_prepare_publisher_mapping(abuse_results)
    labeled=_apply_basic_labels(df_original, mapping)
    labeled=_apply_capping(labeled, cap_abuse_rate, min_suspicious_ratio)
    stats=_calculate_statistics(labeled, abuse_results)
    return labeled, stats

# 결과 및 라벨링
abuse_results, individual_cases = detect_abuse_main(df_join)
df_join_v1, stats = apply_abuse_labels(df_join_v1, abuse_results, cap_abuse_rate=DEFAULT_CAP_ABUSE_RATE)

# 저장
df_join_v1.to_csv("df_join_v1.csv", index=False)