#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""올바른 어뷰징 스코어링 - 10개 로직 종합, 매체사 규모별 차별화"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def proper_scoring():
    """올바른 어뷰징 스코어링"""
    logger.info("🎯 올바른 종합 어뷰징 스코어링 시작...")
    
    # 설정 로드
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logic_weights = config['logic_weights']
    
    # 데이터 로드
    df = pd.read_csv('qqqq/df_join_abuse.csv')
    abuse_cols = [f'abuse_{i}' for i in range(1, 11) if f'abuse_{i}' in df.columns]
    
    logger.info(f"데이터: {len(df):,}행, 로직: {len(abuse_cols)}개")
    
    # 1단계: 매체사 규모 분류
    logger.info("매체사 규모 분류...")
    
    mda_traffic = df.groupby('mda_idx').size().sort_values(ascending=False)
    
    # 규모별 분류 (트래픽 기준)
    large_threshold = mda_traffic.quantile(0.9)    # 상위 10% = 대형
    medium_threshold = mda_traffic.quantile(0.7)   # 상위 30% = 중형
    
    large_mdas = set(mda_traffic[mda_traffic >= large_threshold].index)
    medium_mdas = set(mda_traffic[(mda_traffic >= medium_threshold) & (mda_traffic < large_threshold)].index)
    small_mdas = set(mda_traffic[mda_traffic < medium_threshold].index)
    
    logger.info(f"대형: {len(large_mdas)}개, 중형: {len(medium_mdas)}개, 소형: {len(small_mdas)}개")
    
    # 2단계: 매체사별 종합 스코어링
    logger.info("매체사별 종합 스코어링...")
    
    mda_results = []
    
    for mda_id in df['mda_idx'].unique():
        if pd.isna(mda_id):
            continue
            
        mda_data = df[df['mda_idx'] == mda_id]
        total_traffic = len(mda_data)
        
        # 규모 분류
        if mda_id in large_mdas:
            mda_size = 'LARGE'
            size_factor = 1.2  # 대형은 기준 강화
        elif mda_id in medium_mdas:
            mda_size = 'MEDIUM' 
            size_factor = 1.0  # 중형은 기준
        else:
            mda_size = 'SMALL'
            size_factor = 0.8  # 소형은 기준 완화
        
        total_score = 0.0
        logic_scores = {}
        logic_details = {}
        
        # 3단계: 로직별 점수 계산 (10개 모두)
        for col in abuse_cols:
            if col not in mda_data.columns:
                continue
                
            logic_num = int(col.split('_')[1])
            if logic_num not in logic_weights:
                continue
            
            abuse_data = mda_data[mda_data[col] > 0]
            if len(abuse_data) == 0:
                continue
            
            # 기본 지표
            frequency = len(abuse_data)
            severity = abuse_data[col].mean()
            abuse_rate = frequency / total_traffic
            
            # 규모별 조정된 magnitude 계산
            if mda_size == 'LARGE':
                # 대형: 비율 중심, 절대값 억제
                magnitude = (abuse_rate ** 0.9) * (np.log1p(frequency) ** 0.2) * severity
            elif mda_size == 'MEDIUM':
                # 중형: 균형
                magnitude = (abuse_rate ** 0.8) * (np.log1p(frequency) ** 0.3) * severity  
            else:
                # 소형: 절대값도 고려
                magnitude = (abuse_rate ** 0.7) * (np.log1p(frequency) ** 0.4) * severity
            
            # 4단계: 극값 조절 (tanh 포화함수)
            normalized_mag = np.tanh(magnitude * 3)  # 0~1 사이로 조절
            
            # 5단계: 최종 로직 점수
            base_weight = logic_weights[logic_num]
            recency_boost = 1.1 if logic_num in [1, 3, 6, 7, 8] else 1.0
            
            logic_score = base_weight * normalized_mag * recency_boost * size_factor
            
            total_score += logic_score
            logic_scores[logic_num] = logic_score
            logic_details[logic_num] = {
                'rate': abuse_rate,
                'frequency': frequency,
                'severity': severity,
                'magnitude': magnitude,
                'normalized': normalized_mag
            }
        
        if total_score > 0:
            # 상위 기여 로직
            top_contribs = sorted(logic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            contrib_str = ', '.join([f"L{lid}:{score:.3f}" for lid, score in top_contribs])
            
            mda_results.append({
                'entity_id': mda_id,
                'total_score': total_score,
                'size_category': mda_size,
                'traffic_size': total_traffic,
                'logic_count': len(logic_scores),
                'contributions': contrib_str,
                'top_logic': max(logic_scores.items(), key=lambda x: x[1])[0] if logic_scores else None,
                'score_balance': len([s for s in logic_scores.values() if s > total_score * 0.1])  # 10% 이상 기여하는 로직 수
            })
    
    # 결과 정렬
    mda_df = pd.DataFrame(mda_results).sort_values('total_score', ascending=False).reset_index(drop=True)
    
    logger.info(f"매체사 완료: {len(mda_df)}개")
    
    # 전역 변수 저장
    globals()['mda_scores'] = mda_df
    
    # 퍼블리셔 스코어링 (간단화)
    logger.info("퍼블리셔 스코어링...")
    pub_results = []
    pub_sample = df.groupby('pub_sub_rel_id').apply(lambda x: x.head(1000)).reset_index(drop=True)
    
    for pub_id in pub_sample['pub_sub_rel_id'].value_counts().head(1000).index:
        if pd.isna(pub_id):
            continue
        pub_data = pub_sample[pub_sample['pub_sub_rel_id'] == pub_id]
        parent_mda = pub_data['mda_idx'].mode()[0] if len(pub_data['mda_idx'].mode()) > 0 else 'unknown'
        
        total_score = 0.0
        for col in abuse_cols:
            if col not in pub_data.columns:
                continue
            logic_num = int(col.split('_')[1])
            if logic_num not in logic_weights:
                continue
            
            abuse_data = pub_data[pub_data[col] > 0]
            if len(abuse_data) == 0:
                continue
                
            abuse_rate = len(abuse_data) / len(pub_data)
            severity = abuse_data[col].mean()
            magnitude = (abuse_rate ** 0.8) * (np.log1p(len(abuse_data)) ** 0.3) * severity
            normalized_mag = np.tanh(magnitude * 2)
            
            logic_score = logic_weights[logic_num] * normalized_mag * 0.8  # 퍼블리셔 할인
            total_score += logic_score
        
        if total_score > 0:
            pub_results.append({
                'entity_id': pub_id,
                'parent_mda': parent_mda,
                'total_score': total_score
            })
    
    pub_df = pd.DataFrame(pub_results).sort_values('total_score', ascending=False).reset_index(drop=True)
    logger.info(f"퍼블리셔 완료: {len(pub_df)}개")
    
    # 사용자 스코어링 (샘플링)
    logger.info("사용자 스코어링...")
    user_sample = df.sample(n=min(100000, len(df)), random_state=42)
    user_sample = user_sample.copy()
    user_sample['user_key'] = user_sample.apply(lambda row: 
        f"dvc:{int(row['dvc_idx'])}" if pd.notna(row['dvc_idx']) and row['dvc_idx'] != 0 
        else f"ip:{row['user_ip']}" if pd.notna(row['user_ip']) 
        else 'unknown', axis=1)
    
    user_results = []
    for user_key in user_sample['user_key'].value_counts().head(3000).index:
        if user_key == 'unknown':
            continue
        user_data = user_sample[user_sample['user_key'] == user_key]
        
        total_score = 0.0
        for col in abuse_cols:
            if col not in user_data.columns:
                continue
            logic_num = int(col.split('_')[1])
            if logic_num not in logic_weights:
                continue
                
            abuse_data = user_data[user_data[col] > 0]
            if len(abuse_data) == 0:
                continue
                
            abuse_rate = len(abuse_data) / len(user_data)
            severity = abuse_data[col].mean()
            magnitude = (abuse_rate ** 0.85) * (np.log1p(len(abuse_data)) ** 0.25) * severity
            normalized_mag = np.tanh(magnitude * 2)
            
            logic_score = logic_weights[logic_num] * normalized_mag * 0.6  # 사용자 할인
            total_score += logic_score
        
        if total_score > 0:
            user_results.append({
                'entity_id': user_key,
                'total_score': total_score
            })
    
    user_df = pd.DataFrame(user_results).sort_values('total_score', ascending=False).reset_index(drop=True)
    logger.info(f"사용자 완료: {len(user_df)}개")
    
    # 종합 통합
    overall_list = []
    overall_list.append(mda_df.head(50).assign(dimension='MDA', normalized_score=lambda x: x['total_score'] * 1.0))
    overall_list.append(pub_df.head(100).assign(dimension='PUB', normalized_score=lambda x: x['total_score'] * 0.8))
    overall_list.append(user_df.head(100).assign(dimension='USER', normalized_score=lambda x: x['total_score'] * 0.6))
    
    overall_df = pd.concat([df[['entity_id', 'dimension', 'normalized_score', 'total_score']] for df in overall_list], ignore_index=True)
    overall_df = overall_df.sort_values('normalized_score', ascending=False).reset_index(drop=True)
    overall_df['rank'] = range(1, len(overall_df) + 1)
    
    # 전역 변수 저장
    globals()['mda_scores'] = mda_df
    globals()['pub_scores'] = pub_df  
    globals()['user_scores'] = user_df
    globals()['overall_scores'] = overall_df
    
    logger.info("✅ 올바른 종합 스코어링 완료!")
    
    return mda_df, pub_df, user_df, overall_df

if __name__ == "__main__":
    mda_scores, pub_scores, user_scores, overall_scores = proper_scoring()
    
    print("\n🏆 올바른 TOP 15 매체사:")
    print(mda_scores[['entity_id', 'total_score', 'size_category', 'traffic_size', 'logic_count', 'score_balance', 'contributions']].head(15))
    
    # 539번 분석
    if 539 in mda_scores['entity_id'].values:
        mda_539 = mda_scores[mda_scores['entity_id']==539].iloc[0]
        rank_539 = mda_scores[mda_scores['entity_id']==539].index[0] + 1
        print(f"\n🔍 539번 상세 분석:")
        print(f"순위: {rank_539}")
        print(f"점수: {mda_539['total_score']:.3f}")
        print(f"규모: {mda_539['size_category']}")
        print(f"기여 로직수: {mda_539['logic_count']}개")
        print(f"균형 로직수: {mda_539['score_balance']}개")
        print(f"기여도: {mda_539['contributions']}")
    
    # 규모별 분포
    print(f"\n📊 TOP 15 규모별 분포:")
    size_dist = mda_scores.head(15)['size_category'].value_counts()
    for size, count in size_dist.items():
        print(f"{size}: {count}개")
    
    print(f"\n📚 TOP 10 퍼블리셔:")
    print(pub_scores[['entity_id', 'parent_mda', 'total_score']].head(10))
    
    print(f"\n📱 TOP 10 사용자:")
    print(user_scores[['entity_id', 'total_score']].head(10))
    
    print(f"\n🏆 종합 TOP 20:")
    print(overall_scores[['rank', 'dimension', 'entity_id', 'normalized_score']].head(20))
    
    print(f"\n🎯 스코어링 특징:")
    print("✅ 10개 로직 모두 종합 평가")
    print("✅ 대형/중형/소형 매체사별 차별화된 기준")  
    print("✅ 극값은 tanh 함수로 적절히 조절")
    print("✅ 한 로직 편향 방지 (balance 점수)")
    print("✅ 4개 차원 데이터프레임 완성")
