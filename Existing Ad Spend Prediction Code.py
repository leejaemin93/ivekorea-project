#!/usr/bin/env python
# coding: utf-8

# - 25 전처리 완료 및 글로벌
# - 앙상블 포함

# In[1]:


import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


ads_year = pd.read_csv("df_rpt_clean.csv")
ads_list = pd.read_csv("df_list_v1.csv")


# In[3]:


is_test_ad = ads_list["ads_name"].str.contains("테스트|서비스종료|삭제", na=False, case=False)
is_valid_test_ad = ads_list["ads_name"].str.contains("브레인 테스트|게임테스트", na=False, case=False)
condition_to_delete = is_test_ad & ~is_valid_test_ad
ads_list = ads_list[~condition_to_delete].copy()


# In[4]:


# --- 분류 함수 (제외 키워드 적용) ---

def categorize_domain(df, domain_name, keywords, search_cols, exclude_keywords=None):
    """
    DataFrame의 지정된 열(search_cols)에서 키워드를 찾아 
    새로운 도메인을 부여하는 함수입니다. (제외 키워드 적용)
    
    Args:
        df (pd.DataFrame): 작업할 데이터프레임
        domain_name (str): 새로 부여할 도메인 이름 (예: '금융/보험')
        keywords (list): 찾을 키워드 리스트 (예: ['피싱', '보험'])
        search_cols (list): 키워드를 검색할 열 이름 리스트 (예: ['ads_name', 'ads_summary'])
        exclude_keywords (list, optional): 제외할 키워드 리스트 (예: ['고양이'])
        
    Returns:
        pd.DataFrame: 도메인이 추가된 데이터프레임
    """
    # 1. 키워드 리스트를 OR(|) 조건으로 묶어 하나의 검색 패턴으로 만듭니다.
    keyword_pattern = '|'.join(keywords)
    
    # 2. 검색할 모든 열을 대상으로 키워드가 포함되었는지 확인하는 전체 조건을 만듭니다.
    contains_keyword = df[search_cols].apply(
        lambda col: col.str.contains(keyword_pattern, na=False)
    ).any(axis=1)
    
    # 3. 제외 키워드가 있을 경우, 제외 키워드가 포함된 행을 False로 처리
    if exclude_keywords is not None and len(exclude_keywords) > 0:
        exclude_pattern = '|'.join(exclude_keywords)
        contains_exclude = df[search_cols].apply(
            lambda col: col.str.contains(exclude_pattern, na=False)
        ).any(axis=1)
    else:
        contains_exclude = pd.Series([False]*len(df), index=df.index)
    
    # 4. 'domain' 열이 비어있는(NaN) 행 중에서, 키워드가 포함되고, 제외 키워드는 포함되지 않은 행만 선택
    condition = (df['domain'].isna()) & (contains_keyword) & (~contains_exclude)
    
    # 5. 해당 행의 'domain' 열에 새로운 도메인 이름을 채워 넣습니다.
    df = df.copy()  # 원본 데이터 보호
    df.loc[condition, 'domain'] = domain_name
    
    print(f"✅ '{domain_name}' 카테고리 분류 완료! ({condition.sum()}개 적용, 제외 키워드: {exclude_keywords})")
    return df

# --- 함수 사용 예시 ---

# 0. 먼저 'domain' 열을 생성하고 NaN으로 초기화합니다.
ads_list['domain'] = np.nan

# 1. '금융/보험' 도메인 분류 실행 (제외 키워드: '고양이')
keywords_finance = ['금융','피싱', '보험', '주식', '펀드', '스탁', '신한', '머니트리카드', '공모주', 'KB', '업비트', '거래소', '환급금', '상조','더핀','증권','뱅크','보험료']
exclude_keywords_finance = ['고양이','차차차','중고차','Idle']
search_columns = ['ads_name', 'ads_save_way']
ads_list = categorize_domain(ads_list, '금융/보험', keywords_finance, search_columns, exclude_keywords=exclude_keywords_finance)


# 2. '게임' 도메인 분류 실행
keywords_game = [
    '게임', '스코어', '캐릭터', '서브퀘스트', '포커', '퍼즐', '라스트워', '피자 레디', 
    '복권', 'RAID', '머지아일랜더스', '키우기', 'Merge', 'Puzzle', '아케론', '다크엔젤',
    '악마단', 'RPG', '서바이벌', '라바', '모바일', '사전예약', '타이니팜', '이계밥', '타이쿤',
    '강쥐','2X','골목주방', '올바른고','철물파크','레벨','도달','달성','00점','원스토어','Idle','Lv.',
    '초이스 맞추기','클리어','에리어','빗썸','Complete','Tycoon','다이노 투 레이스','획득',
    '구역 해제','모으기','소울 로그','잠금 해제','외계 시대','좀비','붉은 도시', '코인피클', '디펜스', '디바인엣지','영웅줍줍','K데몬헌터스','조조의 꿈','편의점 정리왕 3D'
]
# 제외할 키워드 리스트를 정의합니다.
exclude_keywords_game = ['프리다이빙']

# 함수를 호출할 때 exclude_keywords 인자를 정확히 전달합니다.
ads_list = categorize_domain(
    ads_list, 
    '게임', 
    keywords_game, 
    search_columns, 
    exclude_keywords=exclude_keywords_game
)


# 3-1. '생활/쇼핑/이커머스_상품소비' 도메인 분류 실행
keywords_goods_consumption = [
    # 기존 키워드 유지 (삭제/수정 금지)
    # 식품/건강
    '로얄캐닌', '콜라겐', '흑염소', '콘드로이친', '비타민', '간식', '베지밀', '식품', '블랙마카', '절임배추', '뉴트리하루', '영양제', '천심련 아나파랙틴', '고려은단', '오쏘몰', '간장게장', '왕뚜껑 킹스브레이브', '헬스케어', '케어', '홈헬스',
    '바지락', '닭가슴살', '식단', '단백질', '제주농장', '진공포장', 'MSM', '글루타치온', '완도', '김',
    # 패션/뷰티
    '향수', '여신티켓', '화장품', '스나이델뷰티', '바바더닷컴', '팬츠',
    '목걸이', '가방', '슬링백', '반팔티', '보정속옷', '캔버스', '카프스킨', '립앤칙스', '멀티밤', '톤업크림', '크림샌드',
    # 리빙/인테리어
    '핸드타올', '방석', '오늘의집', '한샘몰', '바자르', '방향제', '네온라이트',
    '싱크대배수구', '배수통', '배수관', '방충망', '모기장', '하수구트랩', '도어스토퍼', '선반', '매트',
    # 디지털/가전
    'LG전자', '삼성닷컴', '몬스타기어', '몬스타pc스토어',' 전자',
    '공기청정기', '필터', '랜턴', '마사지기', '모기채', '카드단말기', '키오스크', '포스기', '터미널', '커피메이커', '드립포트', '전기그릴',
    # 반려동물
    '강아지', '애견', '고양이', '덴탈껌', '치약껌', '이동가방', '리터네코',
    # 기타 상품소비
    '연장체인', '대나무체인', '퍼퓸', '오드퍼퓸', '세트', '골프 퍼팩트맨', '부스터 마사지건', '알로에겔', '알로에', '폼클렌징', '영화할인권', '영화예매권', '보일러',
    # 신규 추가: 상품소비(구매) 관련 키워드
    '토스터기', '안마기', '티포트', '크림', '토너', '경주빵', '이어폰', '충전케이블', '베개', '카사바칩', '샘물', '에그트레이', '귀이개', '마사지건', '실내사이클', '분리수거함', '보조배터리', '밀대', '코인타올', '목난로',
    # 신규 추가: 샘플 데이터에서 추출된 상품소비(구매) 관련 키워드
    '토마스 풀 패밀리', '멀티 와이드 그릴', '뉴네이처 알티지오메가3파워', '황후지화', '다이어트커피', '체지방감소', '레오폴디 로봇청소기', '스텐밀폐용기', '그린바나나 정', '깐알밤', '아쿠아 부스트 옴므 올인원', '체중계', '빨래바구니', '쎈쏘팔메토', '순녹용 골드', '진공청소기', '다이렉트 다이어트N유산균', '발효흑삼고', '레몬밤 다이어트', '샤인 미스트', '요거트메이커', '피부관리기', '올포유', '캐리어', '에어 프라이어', '후라이팬', '오로라플러스플라즈마', '피톤치드 탈취제', '카무트 효소', '모로실', '쿨토시', '헤어 에센스', '와인잔', '접이식 카트', '홍삼정', '리챔', '돼지양념구이', '믹스웰 블랜더', '차전자피식이섬유분말스틱', '브랜드밀효소', '연어코프로테오글리칸프리미엄', '비타 이뮨 플래티넘 골드', '타우린L-아르기닌', '냄비', '셀 이펙터 세럼', '오트밀', '관절보단', '기억력개선', '스트롱 업 차전자피', '호박 진액', '발효홍삼 산삼배양근 활력진골드 앰플', '핸디스팀다리미', '코헴무선스팀다리미', '코힐밤', '마호가니 원형도마', '마호가니 사각도마', '에디번 전기 그릴', '멀티믹서기', '철갑상어 건강즙', '홍삼로얄젤리스틱', '6년근홍삼정에브리원', '로댕 쏘 화이트업 마스크팩', '이데베논 페롤릭 부스터 앰플', '발효구기자 차', '모로오렌지 C3G 프리미엄', '목동알탕', '목동 곤이알찜', '올플레임 IH 세라믹 후라이팬', '올플레임 IH 세라믹 궁중팬', '캐리어 기내용', '레디백', '헤어클리닉', '쎈류신에너지파워', '에브리원 헤어드라이어', '아쿠아 무드등가습기', '마뜨앙 여행용 기내형캐리어', '찹살떡', '설기', '퍼펙트 커버 쿠션''온열', '카트', '투표권', '가르시니아', '브로멜', '파인애플효소', '하루힘차', '여주해죽순돼지감자차', '당당컷', '라끄시안', '굿매너', '화장지', '뷰티앤소프트', '더화이트', '키친타올', '물티슈', '나틴다', '클렌징', '립스틱', '로션', '자연에서 온 녹차', '블랙앵거스', 'LA갈비', '소불고기', '돼지불고기', '라에스테', '퍼펙트 듀오', '샴푸', '트리트먼트', '드라이기', '인덕션', '에어프라이어', '세정티슈', '벌꿀', '헤어팩', '양말', '더플백', '진공 쌀통', '찜질박사', '주방세제', '세탁세제', '섬유유연제', '톱날과도', '기황단', '레몬즙', '오메가3', '락토페린', '브레인1088', '루테인', '허니로얄제리', '참진한 흑마늘', '칫솔', '이불', '패드', '바싹불고기', '냉감 패드', '냉감 바디필로우', '에어매쉬토퍼', '레몬자몽즙', '레자몽', '레티놀C', '짜장', '해물짬뽕', '홍삼진', '홍삼액진', '흑삼진액', '스팀 헤어팩', '헤어로스 세럼', '헤리티지', '셀 부스팅 니들', '카무트효소', '리놀렌산', '장어활력', '녹용', '아르기닌', '마카', '샬롱', '이지쿡', '글라스뷰', '스위스밀리터리', '이엔비', '인사덴탑', 'NMN', 'Tubble', '메디트리', '네이처프리', '코오롱제약', '보령파워', '코스모팜', '신꼬', '칸투칸', '독일 코겔', '햄토피아', '동의삼', '순수달', '밸런스어게인', '퍼니트', '맛다움', 'G마켓',
    # 신규 추가: 샘플 데이터 기반 추가 키워드 (기존 키워드와 중복 제외)
    '케라틴 키퍼', '액상 마그네슘', '콤부차', '복합유산균', '골드 앰플', '워터 글라이드', '프로바이오틱스', '유기농애사비스틱', '유기농레몬생강즙', '오트오브맘', '프로틴메이트', '아르간 에센스', '스트레이트 펌', '헤어앰플', '스프레이', '염모제', '귀리 오트오브맘', '쿠폰팩', '마늘후랑크', '동치미 물냉면', '곡물발효효소', '함흥냉면', '신라면', '치즈카츠', '천연펄프', '팔토시', '닭한마리 볶음탕', '캡슐세제', '팔목형', '손목형', '보양 추어탕', '귀리쌀죽', '선 스크린', '바른어묵', '우거지 뼈해장국', '소내장탕', '카무트 누룽지', '보풀제거기', '골드 등급 패키지', '실버 등급 패키지', '블랙 등급 패키지', '팥찜질팩', '규조토페인트','스탠드다리미',
    '가정용천국의계단','침구청소기','프랭클린 이염방지시트','젖병소독기','자차청정기','유럽미장',
    '규조토페인트','커리쉴', '루트 레미디', '스칼프 토닉', '프레시포레스트향', '소프트켄넬', '얼음조끼', 'homegrow', '창문열차단', '암막시트지', '발 받침대', '반팔 와이셔츠', '인감도장', '럭스엑스팟', '용가리', '목문용방문손잡이', '플로우아쿠아슈즈', '골프모자', '보스턴백', '건전지', '트램폴린', '남자셔츠', '쿠션커버', '오리젠사료오리지날', '골프피규어', '수면조끼', '베이포레소맥스', '쿠션', '볼마커', '라셀르간냉식', 'Modern High', '팜아크', '아크용접기', '바를 수딩 쿨링젤', '지아자연애장판', '유막제거제', '진정 앰플', '퍼팩트컴', '각인반지', '니트릴장갑', '부산국밥', '팰리세이드호환튜닝용품', '드리미 음식물처리기', '43인치스탠바이미', '휴나인 매스틱 검', '팰리세이드튜닝', '팰리세이드튜닝 c필러수납', '팰리세이드튜닝142', '팰리세이드 사이드스텝', '팰리세이드 쓰레기통 튜닝', '코인비엠에스', '갤럭시북K71AR', 'NT950XFT-A51A', '하주씨앤씨', '긱베이프 레전드3','루이보스티', '양배추즙', '토마토즙', '대추즙', '헛개진액', '수제청', '포스파티딜세린', '올리브오일', '미백마스크팩', '진정마스크팩', '유산균', '등산화깔창', '게이밍 컴퓨터의자', '디퓨저', '골프패치', '남자링거티', '기능성반팔 쿨 카라티', '즐거운가구 렌지대', 'NUTRALIFE 카테킨', '비엔엠코리아', '미마유모차', '에피소X', '블로커콤비블라인드', '엘리카', '벨벳', '엘라카테슬라', '테슬라', '오란다', '오토바이 자석 거치대', '바벨런스', '쿠션커버', '쿠션', '수면조끼', '팥찜질팩', '방수앞치마', '명함제작', '스티커제작', '드레이프블라우스', '꼬막', '방수앞치마', '유리문 도어락', '무타공 도어락', '모기퇴치기', '해충 퇴치기', 'Onliv', '잔더','롯데마트 제타', '락앤락', '동국제약', '닥터로니', '휴리엔', '백년약방', '네일로그', '안락의자', '리빙박스', '아이언커버', '와이셔츠', '블라우스', '한우사골곰탕', '곶감', 'usb', '청국장', '베누스타 청소기', '컴퓨터의자', '인테리어의자', '그로밋 키링', '해장국', '딸기칩', '개완', '티머그', '차총', '보이차', '구찌뽕 추출물', '백향과', '황토볼', '차열페인트', '곰팡이방지페인트', '결로방지페인트', '벽지페인트', '가정용미니에어건', '네오프렌구명조끼', '휴나인 애사비','삼성 G100','삼성 H170D','삼성 F100D','키친플래그','제주탄산수',
    # 신규 추가: 분석된 상품소비 키워드
    '밴디 양방향 미끄럼방지 실내 욕실화', '삼성제약 내간에밀크씨슬실리마린', 'US 스킨아우라히알루론비타', 'US 프로바이오포스콜리500', '부사 사과', '스테비아 방울토마토', '천도복숭아', '하우스 감귤', '청사과', '프리미엄 파로', '백골뱅이', '대추 방울토마토', 'vintage', '체리 운드', '밤꿀', '네오디움 자석', '쉬폰커튼', '암막커튼', '코드스트랩', '스와들', '셀프페인트', '젯소', '페인트', '친환경페인트', '마이크로시멘트', '휘게로 페인트', '베스트에어컨 투인원', '여성 여름 셔츠', '치약', '피톤치드', '나노건', '피톤치드연무기', 'LCN마이코셉트', '업젤', '닭갈비', '오리주물럭', '프로파일','전자담배','논가스용접기',
    # 신규 추가: 분석된 데이터에서 상품소비로 분류할 키워드
    '볶은아몬드', '무선 쿨링 고데기', '골프트로피', '스탠맨해머전동드릴', '블루키워드', '리브라 에보', '유산지어', '지오 24MR400W 지오비전', '개밥청춘 위픽 에어드', '테이블야자', '떡볶이택배', '구운계란', '엘카라 프로폴리스', '크리스탈 감사패', '커피원두', '인섹트도그', '미마 모카색상 절충형유모차', '여행용 프리볼트 고데기', '수술복', '갤럭시S25케이스', '꼬리곰탕', '아론샵 홀드맥스 헤어젤', '루피는 참치마요', '야채참치', '밀크팡 산양유프로틴', '대추차', '27US550', '윈도우11', '닥터스초이스 콘드', '자전거라이트어', '자바라의자', '스탠드스팀다리미', '스팀다리미', '구강세정기', '제본', '산삼', '식물등', '테라리움', '책상 스탠드', '슬릿분', '백일상대여', '출차주의등', '수모', '도라지청', '환갑현수막', '칠순현수막', '관리기', '결명자차', '구기자차', '보리차 원액',
    # 신규 추가: 100행 데이터에서 확실한 상품소비 키워드
    '구운아몬드', '캐슈넛', '임산부효소', '업소용청소기', '아기잠옷', '아기파자마', 'PPF 왁스', '실내수영복', '용융소금', '스캇나인 아구아헤', '스타리온 대형 45박스 올냉장 냉장고', '전자저울', '멜라토닌', '감사패', '장수돌침대', '골프마스크팩', '교구장', '체리', '코튼 폴로 반팔니트', '코튼 반팔 폴로 여름니트', '차거름망', '곰탕육수', '씽크어스 하트', '차판', '자숙문어', '메리네 라비앙독', '여성 반팔 셔츠', '샤인머스켓', '오토바이리스', '양갈비', '까렌다쉬샤프', '잔디깎기', '하수구 뚫는 기계', '메탈지그', '청양고추', '예초기', '배달오토바이리스', '오리젠사료', '시트지제작', '우정링', '커플팔찌', '홍삼음료', '클립온 선글라스', '노시부석션팁', '닥터커피', '아몬드', '안전벨트', '건강하삼', '햄스터 케이지', 'ab슬라이드', '숙취해소제', '삼성ssd', '석재볼라드', '마사지젤', '가죽스티커', '콘체 블렌더',
    # 신규 추가: 83행 데이터에서 확실한 상품소비 키워드
    '메이튼 맥세이프자동차거치대', '베베끌레르 퍼펙트제로', '리플렛', '어닝', '방수액', '스트레치필름', '오토바이자석거치대', '핸드워시 답례품', '가죽트레이', '아만', '휴대폰 맥세이프 마그네틱 투명케이스', '어그 밴딩 슬리퍼', '손톱강화제', '까렌다쉬 샤프', '메탈파일', '원두커피', '인형뽑기기계', '일본지게차', '피오렌자또', '씨메커피머신', '토삭스', '파고라', '패션뷰티몰', '세레스쥬얼리',
    '최대 77% 할인. 럭키세븐 특가','29cm', '상품찜','JONS 신규 가입 혜택','참소라', '글루어트', '새미쥬얼리', '조개구이', '초코파이','나물짤순이'
]
exclude_keywords_goods_consumption = ['연남동 스튜디오']
ads_list = categorize_domain(
    ads_list,
    '상품소비',
    keywords_goods_consumption,
    search_columns,
    exclude_keywords=exclude_keywords_goods_consumption
)

# 3-2. '생활/쇼핑/이커머스_생활서비스' 도메인 분류 실행
keywords_life_service = [
    # 기존 키워드 유지 (삭제/수정 금지)
    # 교육/부동산
    '교육', '학습', '영어', '보카', '홈런', '온리원', '부동산', '경매', '기출문제', '학원',
    # 여행/문화
    '호텔', '하나투어', '골프네임택', '스마트캐디', '프리다이빙',
    # 생활서비스
    '운동', '휘트니스', '테라피', '캠핑', '물구나무', '운동기구', '참여신청', '달다방',
    # 신규 추가: 회원가입(서비스) 관련 키워드
    '회원가입', '서비스 가입', '라이프플러스', '트라이브', 'H.Point 회원 가입하기',
    # 샘플 데이터에서 추출된 생활서비스 관련 키워드
    '공용 화장실 해제','이혼전문변호사',
    # 신규 추가: 샘플 데이터 기반 추가 키워드 (기존 키워드와 중복 제외)
    '최초 오픈', '기부완료', '쿠폰팩 가입', '인터넷신규가입','Shipping','패러글라이딩', '일상비일상의틈', 'NOL', 'NOL (신규회원 국내숙소 특별할인)', '최초 오픈', '정답입력',
    # 추가: 리뷰 전체보기 클릭 후 URL 제출, 스토어 알림받기, 명소찾기
    '리뷰 전체보기 클릭 후 URL 제출', '스토어 알림받기', '명소찾기',
    # 신규 추가: 분석된 생활서비스 키워드
    '상세페이지제작', '비상주사무실',  'V컬러링 이용권',
    # 신규 추가: 100행 데이터에서 확실한 생활서비스 키워드
    '로고제작',
    # 신규 추가: 83행 데이터에서 확실한 생활서비스 키워드
    '특수청소', '전단지배포및제작','바카르','젤톡스','SK브로드밴드'
]
exclude_keywords_life_service = []
ads_list = categorize_domain(
    ads_list,
    '생활서비스',
    keywords_life_service,
    search_columns,
    exclude_keywords=exclude_keywords_life_service
)

# 3-3. '생활/쇼핑/이커머스_플랫폼' 도메인 분류 실행
keywords_platform = [
    # 기존 키워드 유지 (삭제/수정 금지)
    '쿠팡', '아이템매니아', '중고거래', '백화점', '당근마켓', '동네GS', 'GS25', '배달의민족',
    # '네이버스마트스토어',
    # 샘플 데이터에서 추출된 플랫폼 관련 키워드
    'G마켓',
    # 신규 추가: 샘플 데이터 기반 추가 키워드 (기존 키워드와 중복 제외)
    '에누리 가격비교', '네이버 쇼핑', '네이버 상품 해시태그', '상품 태그 맞추기',
    # # 신규 추가: 분석된 플랫폼 키워드
    # '최대 77% 할인. 럭키세븐 특가',
    # # 신규 추가: 100행 데이터에서 확실한 플랫폼 키워드
    # '29cm', '상품찜',
    # # 신규 추가: 83행 데이터에서 확실한 플랫폼 키워드
    # 'JONS 신규 가입 혜택', '바카르'
]
exclude_keywords_platform = []
ads_list = categorize_domain(
    ads_list,
    '플랫폼',
    keywords_platform,
    search_columns,
    exclude_keywords=exclude_keywords_platform
)

# 아래는 분류가 애매하거나 추가 검토가 필요한 항목 리스트입니다.
# 분류가 애매한 항목 (키워드로 분류가 어려운 경우)
# - 퀴즈 맞추기, 퀴즈 정답 맞추기, [퀴즈] ~, ~맞추기, [정답] ~, 클릭, 받기, 상세페이지제작, 대전 비상주사무실, [간편참여] 특가요정, 강재인 (저장 후 주차장 URL), 다이아XXXX (저장 후 주차장 URL), 푸슈 (저장 후 주차장 URL), 일산보청기 (저장 후 주차장 URL), 미사 피부과 병원 미엘 (저장 후 주차장 URL), 수원중고차 (저장 후 주차장 URL), 인천제일바다낚시 (저장 후 주차장 URL), 라플로르드무아 (저장 후 주차장 URL), 창원세무사 조영빈 사무소 (저장 후 주차장 URL), 대전유리창청소 (저장 후 주차장 URL), 부천한방병원 차오름 (명소찾기), 강남수입차정비 내화모터스 (명소찾기), 이리와코리아 10개옵션 맞추기, 피지컬방이헬스 맞추기, 논가스용접기 200프로 맞추기, 유산지어 맞추기, 테이블야자 맞추기, 떡볶이택배 맞추기, 구운계란 인생 맞추기, 개밥청춘 위픽 에어드 맞추기, 우주인전자담배 맞추기, 지오 24MR400W 지오비전 맞추기, 엘카라 프로폴리스 맞추기, 크리스탈 감사패 맞추기, 인섹트도그7.5kg 맞추기, 잘봄XXX (저장 후 주차장 URL), [퀴즈] 강남 하수구뚫음, 삼성 F100D, 닥터초이스맞추기, 볶은아몬드, [퀴즈] 금천구 하수구막힘, 습자지 맞추기, [퀴즈] 무선 쿨링 고데기, 좋은상패26 닷컴 골프트로피 맞추기, 스탠맨해머전동드릴 맞추기, 블루키워드 맞추기, 동동마켓 리브라 에보 맞추기, 전자담배 액상(미성년자불가미션), 커피원두 맞추기, 상세페이지제작, 대전 비상주사무실, [정답] 치약(네이버 상품 상세내용) 등은 명확한 카테고리 분류가 어려워 추가 검토 필요
# - 닥터초이스맞추기, 삼성 F100D, 볶은아몬드, [퀴즈] 금천구 하수구막힘, 습자지 맞추기, [퀴즈] 무선 쿨링 고데기, 좋은상패26 닷컴 골프트로피 맞추기, 스탠맨해머전동드릴 맞추기, 블루키워드 맞추기, 동동마켓의 엘리카 핏 83 맞추기, 동동마켓의 엘리카 핏 60 맞추기, 동동마켓 리브라 에보 맞추기, 동동마켓 엘리카 니콜라테슬라 벨벳83 맞추기, 동동마켓의 벨벳60 맞추기, 동동마켓 엘라카테슬라 맞추기, 동동마켓 4구엘리카 테슬라 맞추기, 피지컬방이헬스 맞추기, 이리와코리아 10개옵션 맞추기, 미마유모차 절충형유모차 맞추기, 논가스용접기 200프로 맞추기, 유산지어 맞추기, 테이블야자 맞추기, 떡볶이택배 맞추기, 구운계란 인생 맞추기, 기능성반팔 쿨 카라티 무지 맞추기, 즐거운가구 렌지대1800 맞추기, NUTRALIFE 카테킨 맞추기, 비엔엠코리아 10개옵션 맞추기, 오란다, 바벨런스, 골프패치, 남자링거티, 지오 24MR400W 지오비전 맞추기, 우주인전자담배 맞추기, 등산화깔창, 진정마스크팩, 미백마스크팩, 유산균, 디퓨저, 양배추즙, 토마토즙, 대추즙, 헛개진액, 수제청, 포스파티딜세린, 올리브오일, 게이밍 컴퓨터의자, 꼬막, 방수앞치마, 유리문 도어락, 무타공 도어락, 모기퇴치기, 해충 퇴치기, 드레이프블라우스, 명함제작, 스티커제작, 잔더, Onliv, 스토어 알림받기, 아이티시스템 (저장 후 주차장 URL), 강재인 (저장 후 주차장 URL), 인천이혼전문변호사 (저장 후 주차장 URL), 다이아XXXX (저장 후 주차장 URL), 푸슈 (저장 후 주차장 URL), 일산보청기 (저장 후 주차장 URL), 미사 피부과 병원 미엘 (저장 후 주차장 URL), 수원중고차 (저장 후 주차장 URL), 인천제일바다낚시 (저장 후 주차장 URL), 라플로르드무아 (저장 후 주차장 URL), 창원세무사 조영빈 사무소 (저장 후 주차장 URL), 대전유리창청소 (저장 후 주차장 URL), 부천한방병원 차오름 (명소찾기), 강남수입차정비 내화모터스 (명소찾기), 에피소X XX출 (저장 후 주차장 URL), [간편참여] 특가요정, 락앤락 첫구매 이벤트, 상세페이지제작, 대전 비상주사무실
# Mr.Shipping (ads_name: Mr.Shipping, ads_save_way: 참여) : 어떤 카테고리인지 불분명, 추가 검토 필요
# Travel 포캐스트, 헬씨 포캐스트 (ads_save_way: 결제 및 콘텐츠 이용) : 콘텐츠/여행/서비스 등 복합적, 추가 검토 필요
# 사주링 맞추기, 닥터초이스맞추기, 스탠드다리미, 가정용천국의계단, 침구청소기, 패러글라이딩, 습자지 맞추기, 프랭클린 이염방지시트 맞추기, 해님 젖병소독기 맞추기, 해님UVLED젖병소독기4세대 맞추기, 에어힐러자차청정기 맞추기, 유럽미장, 유럽미장셀프, 규조토페인트, 등 "퀴즈 맞추기"류 : 콘텐츠/앱테크/기타 등 분류 애매, 추가 검토 필요
# 강재인, 인천이혼전문변호사, 다이아XXXX, 푸슈, 에피소X 등 (ads_save_way: 퀴즈 정답 맞추기, ads_name에 (저장 후 주차장 URL) 포함) : 분류 불명확, 추가 검토 필요
# - Mr.Shipping (참여, ads_name: Mr.Shipping) : 어떤 카테고리인지 불분명, 추가 검토 필요


# 5.'SNS/커뮤니케이션' 도메인 분류 실행
keywords_sns = ['페이스북', '인스타', '채팅', '친구', '동네', '소개팅', '데이팅', '골드스푼', '여보야', '팬더티비', '상담', '라임', '카페', '카카오톡', '후엠아이','여보랑','비긴즈','커넥트 CONNECT']
exclude_keywords_sns = []
ads_list = categorize_domain(ads_list, 'SNS/커뮤니케이션', keywords_sns, search_columns, exclude_keywords=exclude_keywords_sns)


# 7.'콘텐츠' 도메인 분류 실행
keywords_content = ['탑툰', '웹툰', '미툰', '미노벨', '운세', '워치페이스', '파일썬', '웹하드', '애니툰', '포춘텔러','꿀밤티비', '꿀물티비', '핑크티비','토정비결','굿툰','TikTok','나의 커리어 DNA','천신','성향매핑', '커플케미', '영화리뷰','멜론팔로우', '궁합', '포춘쿠키','홀스브릿지','영화 보스','회원가입 후 결제','KWDA','투표권 +','포캐스트','사주']
exclude_keywords_content = []
ads_list = categorize_domain(ads_list, '콘텐츠', keywords_content, search_columns, exclude_keywords=exclude_keywords_content)

# 8. '앱테크/리워드' 도메인 분류 실행
keywords_apptech = ['앱테크', '돈버는', '리워드어플', '오토링', '짤', '아이부자', 'Cash Giraffe', 'Cash Cow', '파블로', '서베이','MyB','돈 버는 미션','판도라박스', '도형그리기', '박스찾기', '시간잡기','서베이']
exclude_keywords_apptech = []
ads_list = categorize_domain(ads_list, '앱테크/리워드', keywords_apptech, search_columns, exclude_keywords=exclude_keywords_apptech)


# 10. '유틸리티/툴' 도메인 분류 실행
keywords_util = ['통화녹음', '익시오', 'AI', '뤼튼', '구독', '체크플러스']
exclude_keywords_util = []
ads_list = categorize_domain(ads_list, '유틸리티/툴', keywords_util, search_columns, exclude_keywords=exclude_keywords_util)

# 11. '지역/상점' 도메인 분류 실행
keywords_local = ['맛집', '식당', '점', '시장', '휘트니스', '쭈앤쭈', '족발선생',
'소곱판','박가네 빈대떡','풍년옥','철판집','청류','지도','플레이스','연남동',
'방이별관','카카오맵','공항', '클럽', '스튜디오','잠실','그래한의원','신의주찹쌀순대','라 스위스 서촌',
'의원','PT 청라','숙성회','서촌', '한의원', '순대', '쭈꾸미', '막국수','포항', '계곡', '테라피','캄왁싱','이지은웨딩','제주 이모카세',
'수성구 치과', '강남 하수구뚫음', '두정역 메이크업 속눈썹', '불당동 윤곽관리', '봉선동돼지갈비','제주 장어정식', '광주 첨단 갈매기살', '제주 외도동 물회', '외도동 횟집' ,'대전유리창청소', '한남동 인터내셔널 필라테스' , '수원중고차',
# 신규 추가: 미분류 데이터에서 추출된 지역/상점 키워드
'치과', '고깃집', '술집', '이자카야', '헬스장', '케이크', '가족사진', '맞춤정장', '요트투어', '보청기', '피부과', '병원', '미엘', '바다낚시', '세무사', '사무소', '유리창청소', '한방병원', '수입차정비', '모터스', '냉면', '숙성명작', '평양냉면', '여수', '구미', '충주', '용산', '광안리', '진주', '가좌', '칠암동', '진해', '초천동', '진주혁신도시', '굴포천역', '수원역', '강서', '가정동', '일산', '인천', '달항아리', '청라', '부산', '동래역', '랍스타', '고양시', '중앙로', '노량진고시원', '안양 정형외과', '대구 정형외과', '경산시 중산동', '연남 초야', '구리 비뇨기과', '피지컬방이헬스', '경남 아파트']
exclude_keywords_local = []
ads_list = categorize_domain(
    ads_list, 
    '지역/상점', 
    keywords_local, 
    search_columns, 
    exclude_keywords=exclude_keywords_local
)

# 12. '기타' 도메인 분류 실행
# 기타 내부에 속하는 도메인 -> 보안, 뉴스
keywords_etc = ['뉴스','충청남도','다함께차차차','피싱', 'OTP', '안심', '스마트사인', '인증', '범죄알리미', '약속번호','원키퍼','렌트카',
'로그인보호서비스','차차차','보호','사람인','리본카','보안플러스','건강지키미','종합광고대행사','캐디톡','인크루트','네이버맵','정답 미션','붙임머리', '치료', '블로그', '마케팅','제로네이트','법무사','알바천국', '알바몬', '인터넷가입', '누수', '국무조정실','국가', '붙임머리', '치료', '블로그', '마케팅', '입주청소']
exclude_keywords_etc = []
ads_list = categorize_domain(
    ads_list,
    '기타',
    keywords_etc,
    search_columns,
    exclude_keywords=exclude_keywords_etc
)

# 이제까지 분류되지 않은(nan) 모든 행을 '기타'로 할당합니다.
ads_list.loc[ads_list['domain'].isna(), 'domain'] = '기타'


# --- 최종 결과 확인 ---
print("\n--- 분류되지 않은 데이터 개수 ---")
print(f"미분류 데이터: {ads_list['domain'].isna().sum()}개")

print("\n--- 전체 도메인 분류 현황 ---")
print(ads_list['domain'].value_counts())



print("\n--- 각 도메인별 샘플 데이터 ---")
for domain in ads_list['domain'].dropna().unique():
    print(f"\n✅ '{domain}'으로 분류된 데이터:")
    sample_data = ads_list[ads_list['domain'] == domain][['ads_name', 'domain']].head()
    print(sample_data)


# In[5]:


conditions = [
    # 3단계: 최종 수익 창출 (구매, 게임(특정퀘스트))
    (ads_list['ads_type'].isin([9, 12])) | (ads_list['ads_category'].isin([5, 6, 10, 11])),
    # 2단계: 행동 유도 (설치, 실행, 참여, 퀴즈, 구독 등)
    (ads_list['ads_type'].isin([1, 2, 3, 7, 11])) | (ads_list['ads_category'].isin([1, 2, 3, 4, 7, 8])),
    # 1단계: 단순 노출 및 클릭
    (ads_list['ads_type'].isin([4, 5, 6, 8, 10]))
]
values = [3, 2, 1]
# 3. numpy.select를 사용하여 새로운 컬럼 생성
ads_list['ads_category_id'] = np.select(conditions, values, default=0)


# In[6]:


ads_year.info()


# ## 데이터 합치기 + 개별 데이터 피처 엔지니어링

# In[7]:


import datetime
import pandas as pd

# 광고 종료일 변환
ads_list['ads_edate'] = pd.to_datetime(ads_list['ads_edate'], errors='coerce')

# 2025-08-29 이후 종료되는 광고만
compare_date = datetime.date(2025, 8, 29)
ads_list_filtered = ads_list[
    (ads_list['ads_edate'].dt.date >= compare_date) &
    (ads_list['delyn'] == 'N')
]

# 필요한 컬럼만 추출
ads_list_c = ads_list_filtered[['ads_idx','ads_category_id', 'ads_sdate','ads_edate','domain']]

# ads_year 불필요 컬럼 제거
ads_year.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# 두 데이터 합치기 (ads_idx 기준)
df = pd.merge(ads_year, ads_list_c, on='ads_idx', how='inner')


# In[8]:


df.info()


# In[9]:


def downcast(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype_name = df[col].dtype.name

        # object, datetime은 건너뛰기
        if dtype_name == 'object' or 'datetime' in dtype_name:
            continue
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif dtype_name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.1f}% 압축됨'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


df = downcast(df)


# In[10]:


# 클릭수보다 전환수가 큰 경우 전환수를 클릭수로 맞춤
df.loc[
    df['rpt_time_turn'] > df['rpt_time_clk'], 
    'rpt_time_turn'
] = df['rpt_time_clk']

# 확인
over_conversion_after = df[df['rpt_time_turn'] > df['rpt_time_clk']]
print("클릭수보다 전환수가 큰 행 수 (조정 후):", len(over_conversion_after))


# In[11]:


# 고유 ID 생성: ads_idx와 mda_idx를 결합
df["unique_id"] = (
    df["ads_idx"].astype(str).str.zfill(6) +   # ads_idx는 항상 6자리
    df["mda_idx"].astype(str).str.zfill(4)     # mda_idx는 항상 4자리
)

# 필요하다면 숫자로 변환
df["unique_id"] = df["unique_id"].astype(int)


# In[12]:


# 일별로 데이터 정리!!!!

# rpt_time_date를 datetime으로 변환
df["rpt_time_date"] = pd.to_datetime(df["rpt_time_date"])

# 그룹화하면서 ads_idx, mda_idx도 첫 번째 값으로 가져오기
df_day = (
    df.groupby(["unique_id", "rpt_time_date"], as_index=False)
    .agg({
        "rpt_time_turn": "sum",
        "ads_idx": "first",
        "mda_idx": "first",
        "ads_category_id": "first",
        "domain": "first",
        'ads_edate':'first'

    })
    .rename(columns={"rpt_time_turn": "turn_sum"})
)



# In[13]:


df_day['ads_edate'].describe()


# In[14]:


df_day['mda_idx'].nunique()


# In[15]:


df_day.info()


# In[16]:


# 30일 모두 기록이 있는 광고만 필터링

# # 날짜 변환
# df_day["rpt_time_date"] = pd.to_datetime(df_day["rpt_time_date"])

# # 기준일과 최근 30일 구간
# max_date = df_day["rpt_time_date"].max()
# cutoff_date = max_date - pd.Timedelta(days=29)   # 30일 포함 위해 -29

# # 최근 30일 날짜 리스트
# recent_days = pd.date_range(start=cutoff_date, end=max_date, freq="D")

# # unique_id별 날짜 수 세기
# id_date_counts = (
#     df_day[df_day["rpt_time_date"].between(cutoff_date, max_date)]
#     .groupby("unique_id")["rpt_time_date"]
#     .nunique()
# )

# # 30일 모두 있는 unique_id만 추출
# valid_ids = id_date_counts[id_date_counts == len(recent_days)].index

# # 최종 필터링
# df_day = df_day[df_day["unique_id"].isin(valid_ids)].copy()


# ## 최근 30일 필터링

# In[17]:


# import pandas as pd
# # 최근 30일 중 20일 이상 기록된 광고만 필터링

# # 날짜 변환
# df_day["rpt_time_date"] = pd.to_datetime(df_day["rpt_time_date"])

# # 기준일 (데이터셋에서 가장 최근 날짜)
# max_date = df_day["rpt_time_date"].max()
# cutoff_date = max_date - pd.Timedelta(days=29)   # 30일 포함하려면 -29

# # 최근 30일 구간만 추출
# df_recent_window = df_day[df_day["rpt_time_date"].between(cutoff_date, max_date)]

# # unique_id별 기록된 날짜 수 세기
# id_date_counts = df_recent_window.groupby("unique_id")["rpt_time_date"].nunique()

# # 20일 이상 기록된 unique_id만 선택
# valid_ids = id_date_counts[id_date_counts >= 30].index

# # 최종 필터링 (해당 unique_id 전체 기록 유지)
# df_day = df_day[df_day["unique_id"].isin(valid_ids)].copy()


# In[18]:


print(df_day['unique_id'].nunique())
print(df_day['ads_idx'].nunique())
print(df_day['mda_idx'].nunique())


# In[19]:


# # 날짜가 30일 이상인 것만 가져가기 정규분포에 의해서
# date_counts = df_day.groupby("unique_id")["rpt_time_date"].nunique()
# valid_ids = date_counts[date_counts >= 30].index
# df_day = df_day[df_day["unique_id"].isin(valid_ids)]


# In[20]:


print(df_day['unique_id'].nunique())
print(df_day['ads_idx'].nunique())
print(df_day['mda_idx'].nunique())


# In[21]:


df_day.info()


# In[22]:


# 데이터 보간 작업 시작.
# 오늘 날짜
end_date = pd.to_datetime("2025-08-29")

filled_list = []

for uid, group in df_day.groupby("unique_id"):
    # 아이디별 시작 날짜 ~ 8/29까지 생성
    start_date = group['rpt_time_date'].min()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # 날짜 기준으로 reindex
    g = group.set_index("rpt_time_date").reindex(all_dates)

    # flag: 기존 데이터 있으면 1, 없으면 0
    g["flag"] = g["turn_sum"].notna().astype(int)

    # ID 다시 넣기
    g["unique_id"] = uid
    g = g.reset_index().rename(columns={"index": "rpt_time_date"})

    filled_list.append(g)

df_filled = pd.concat(filled_list, ignore_index=True)

print(df_filled.head(15))


# In[23]:


df_filled.isna().sum()


# In[24]:


# 고정값 채우기 (광고별/매체별 단위)
df_filled[['ads_idx','mda_idx','ads_category_id','domain']] = (
    df_filled.groupby('unique_id')[['ads_idx','mda_idx','ads_category_id','domain']].ffill().bfill()
)

# 수치형 지표는 0으로
cols_num = ['turn_sum']
df_filled[cols_num] = df_filled[cols_num].fillna(0)

# # unique_id별 중앙값 구하기
# medians = df_filled.groupby("unique_id")[cols_num].transform("median")
# # NA → 해당 id의 중앙값으로 대체
# df_filled[cols_num] = df_filled[cols_num].fillna(medians)


# In[25]:


df_filled.isna().sum()


# In[26]:


# 날짜가 25 이상인 것만 가져가기 ACF에 의해서
date_counts = df_filled.groupby("unique_id")["rpt_time_date"].nunique()
valid_ids = date_counts[date_counts >= 25].index
df_filled = df_filled[df_filled["unique_id"].isin(valid_ids)]


# In[27]:


print(df_filled['unique_id'].nunique())
print(df_filled['ads_idx'].nunique())
print(df_filled['mda_idx'].nunique())


# ## 파생피처 추가

# ### 일별 광고별 매체별

# In[28]:


#ads_idx + mda_idx 기준으로 "전환이 발생한 날짜 수
group = (
    df_filled.groupby(["ads_idx", "mda_idx"])
    .agg(turn_day_count=('turn_sum', lambda x: (x > 0).sum()))
    .reset_index()
)
group = group.rename(columns={'turn_sum': 'turn_day_count'})

df_feat = df_filled.merge(group, on=["ads_idx", "mda_idx"], how="left")


# In[29]:


df_feat


# In[ ]:





# ### 평균 관련 피처들

# In[30]:


import gc

def add_mean_features(df, mean_features, idx_features):
    # 기준 피처 확인 
    assert (idx_features[0] == 'rpt_time_date') and \
           len(idx_features) in [2, 3]
    
    # 파생 피처명 설정 
    if len(idx_features) == 2:
        feature_name = idx_features[1] + '_mean_turn'
    else:
        feature_name = idx_features[1] + '/' + idx_features[2] + '_mean_turn'
    
    # 기준 피처를 토대로 그룹화해 월간 평균 판매량 구하기 
    group = df.groupby(idx_features).agg({'turn_sum': 'mean'})
    group = group.reset_index()
    group = group.rename(columns={'turn_sum': feature_name})
    
    # df와 group 병합 
    df = df.merge(group, on=idx_features, how='left')
    # 데이터 다운캐스팅 
    df = downcast(df, verbose=False)
    # 새로 만든 feature_name 피처명을 mean_features 리스트에 추가 
    mean_features.append(feature_name)
    
    # 가비지 컬렉션
    del group
    gc.collect()
    
    return df, mean_features


# In[31]:


# 그룹화 기준 피처 중 'ads_idx'가 포함된 파생 피처명을 담을 리스트
ads_mean_features = []

# ['rpt_time_date', 'ads_idx']로 그룹화한 월간 평균 판매량 파생 피처 생성
df_feat, ads_mean_features = add_mean_features(df=df_feat,
                                                 mean_features=ads_mean_features,
                                                 idx_features=['rpt_time_date', 'ads_idx'])

# ['rpt_time_date', 'ads_idx', 'ads_category_id']로 그룹화한 월간 평균 판매량 파생 피처 생성
df_feat, ads_mean_features = add_mean_features(df=df_feat,
                                                 mean_features=ads_mean_features,
                                                 idx_features=['rpt_time_date', 'ads_idx', 'ads_category_id'])


# In[32]:


mda_mean_features = []

# ['rpt_time_date', 'mda_idx']로 그룹화한 월간 평균 판매량 파생 피처 생성
df_feat, mda_mean_features = add_mean_features(df=df_feat,
                                                 mean_features=mda_mean_features,
                                                 idx_features=['rpt_time_date', 'mda_idx'])

# ['rpt_time_date', 'mda_idx', 'ads_category_id']로 그룹화한 월간 평균 판매량 파생 피처 생성
df_feat, mda_mean_features = add_mean_features(df=df_feat,
                                                 mean_features=mda_mean_features,
                                                 idx_features=['rpt_time_date', 'mda_idx', 'ads_category_id'])


# ## 시차피처화

# In[33]:


def add_lag_features(df, lag_features_to_clip, idx_features, 
                     lag_feature, nlags=3, clip=False):
    # 정렬 필수
    df = df.sort_values(idx_features + ['rpt_time_date']).copy()

    for i in range(1, nlags+1):
        lag_feature_name = f"{lag_feature}_lag{i}"

        # 그룹별 shift + NA를 0으로 처리
        df[lag_feature_name] = (
            df.groupby(idx_features)[lag_feature]
              .shift(i)
              .fillna(0)
        )

        if clip:
            lag_features_to_clip.append(lag_feature_name)

    return df, lag_features_to_clip


# In[34]:


# turn_sum의 시차 피처
lag_features = [] # 시차 피처명을 담을 리스트
idx_features = ["ads_idx", "mda_idx"] # 기준 피처

# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=idx_features,
                                                  lag_feature='turn_sum', 
                                                  nlags=3,
                                                  clip=False) 


# In[35]:


# turn_day_count의 시차 피처
lag_features = [] # 시차 피처명을 담을 리스트
idx_features = ["ads_idx", "mda_idx"] # 기준 피처

# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=idx_features,
                                                  lag_feature='turn_day_count', 
                                                  nlags=3,
                                                  clip=False) 


# In[36]:


df_feat = downcast(df_feat, verbose=False)


# In[37]:


df_feat.info()


# In[38]:


# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=['mda_idx'],
                                                  lag_feature='ads_idx_mean_turn', 
                                                  nlags=3,
                                                  clip=False)

# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=['mda_idx', 'ads_category_id'],
                                                  lag_feature='ads_idx/ads_category_id_mean_turn', 
                                                  nlags=3,
                                                  clip=False)
df_feat = df_feat.drop(ads_mean_features, axis=1)


# In[39]:


# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=['mda_idx'],
                                                  lag_feature='mda_idx_mean_turn', 
                                                  nlags=3,
                                                  clip=False)
# idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
df_feat, lag_features = add_lag_features(df=df_feat, 
                                                  lag_features_to_clip=lag_features,
                                                  idx_features=['mda_idx', 'ads_category_id'],
                                                  lag_feature='mda_idx/ads_category_id_mean_turn', 
                                                  nlags=3,
                                                  clip=False)
df_feat = df_feat.drop(mda_mean_features, axis=1)


# In[40]:


df_feat['turn_sum_lag_mean'] = df_feat[['turn_sum_lag1',
                                          'turn_sum_lag2', 
                                          'turn_sum_lag3']].mean(axis=1)


# In[41]:


df_feat['lag_amount1'] = df_feat['turn_sum_lag1']/df_feat['turn_sum_lag2']
df_feat['lag_amount1'] = df_feat['lag_amount1'].replace([np.inf, -np.inf], 
                                                        np.nan).fillna(0)

df_feat['lag_amount2'] = df_feat['turn_sum_lag2']/df_feat['turn_sum_lag3']
df_feat['lag_amount2'] = df_feat['lag_amount2'].replace([np.inf, -np.inf], 
                                                        np.nan).fillna(0)


# In[42]:


df_feat.info()


# In[43]:


df_feat.head()


# In[44]:


# 광고 기간
# 기준 날짜를 datetime으로 변환
base_date = pd.to_datetime("2025-08-29")

# 각 행의 rpt_time_date와 차이 계산
df_feat['due_day'] = (base_date - pd.to_datetime(df_feat['rpt_time_date'])).dt.days


# In[45]:


domain_mapping = {'금융/보험':0, '게임':1, '상품소비':2, '생활서비스':3, '플랫폼':4, 'SNS/커뮤니케이션':5,
           '콘텐츠':6, '앱테크/리워드':7, '유틸리티/툴':8, '지역/상점':9, '기타':10}
df_feat['domain'] = df_feat['domain'].map(domain_mapping)
df_feat['domain'] = df_feat['domain'].astype('category')


# In[46]:


df_feat.info()


# ## 모델 구현 데이터로 형태 변경

# In[47]:


df_feat['year'] = df_feat['rpt_time_date'].dt.year
df_feat['month'] = df_feat['rpt_time_date'].dt.month
df_feat['day'] = df_feat['rpt_time_date'].dt.day
df_feat["unique_id"] = df_feat["unique_id"].astype(int)


# In[48]:


df_feat.info()


# In[49]:


df_feat['turn_sum'].describe()


# In[50]:


# df_feat.to_csv("df_feat.csv", index=False)


# In[51]:


df_feat.head()


# In[52]:


df_feat['turn_sum'].describe()


# In[53]:


print(df_feat['unique_id'].nunique())
print(df_feat['ads_idx'].nunique())
print(df_feat['mda_idx'].nunique())


# In[54]:


24955
1659
322


# In[55]:


print(df_feat['unique_id'].nunique())


# In[ ]:


df_feat.to_csv("df_feat.csv", index=False)


# In[ ]:


df_feat = pd.read_csv("df_feat.csv")


# ## 이동평균 성능 테스트

# In[56]:


train_list = []
valid_list = []
test_list = []

for ad_id, group in df_feat.groupby("unique_id"):
    group = group.sort_values("rpt_time_date")
    
    # 마지막 3일 예측 → test, 그 전날 3일 → valid, 나머지 → train
    test = group.tail(7)
    valid = group.iloc[-14:-7]
    train = group.iloc[:-14]
    
    train_list.append(train)
    valid_list.append(valid)
    test_list.append(test)

train_df = pd.concat(train_list)
valid_df = pd.concat(valid_list)
test_df = pd.concat(test_list)

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)
print("Test shape:", test_df.shape)


# In[57]:


all_test_preds = []   # 전체 test 예측값 저장
all_test_true = []    # 전체 실제값 저장
all_test_dates = []   # 날짜 저장 (선택)

for uid, train in train_df.groupby("unique_id"):
    valid = valid_df[valid_df["unique_id"] == uid]
    test = test_df[test_df["unique_id"] == uid]

    if train.empty or valid.empty or test.empty:
        continue

    mean_value = train["turn_sum"].mean()
    test_pred = np.repeat(mean_value, len(test))

    # 저장
    all_test_preds.extend(test_pred)
    all_test_true.extend(test["turn_sum"].values)
    all_test_dates.extend(test["rpt_time_date"].values)

# numpy array로 변환
all_test_preds = np.array(all_test_preds)
all_test_true = np.array(all_test_true)
all_test_dates = pd.to_datetime(all_test_dates)


# In[58]:


print(results_df.groupby("구간")[["RMSE", "MAE","MSE", "MAPE(%)", "R2"]].mean())


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

results = []

for ad_id, group in df_feat.groupby("unique_id"):
    group = group.sort_values("rpt_time_date")

    # 마지막 7일 = Test
    test = group.tail(7)
    if len(test) < 7:  # 데이터 부족한 광고는 skip
        continue

    # baseline = 과거 전체 turn_sum 평균
    baseline_mean = group["turn_sum"].iloc[:-7].mean()

    # 예측값 = baseline_mean을 test 기간 동안 그대로 복붙
    y_pred = np.repeat(baseline_mean, len(test))
    y_true = test["turn_sum"].values

    # 오차 지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # MAPE (0 division 방지)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    results.append({
        "unique_id": ad_id,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape
    })

# 광고별 baseline 성능
baseline_results = pd.DataFrame(results)

# 전체 평균 성능
overall = baseline_results.mean(numeric_only=True).to_frame("평균").T

print("📊 광고별 baseline 성능")
print(baseline_results.head())

print("\n📈 전체 baseline 평균 성능")
print(overall)


# ## 데이터 분리

# In[59]:


# 데이터 나누기 전에 한 번만 실행
df_feat['ads_idx'] = df_feat['ads_idx'].astype('category')
df_feat['mda_idx'] = df_feat['mda_idx'].astype('category')
df_feat['domain']  = df_feat['domain'].astype('category')


# In[60]:


train_list = []
valid_list = []
test_list = []

for ad_id, group in df_feat.groupby("unique_id"):
    group = group.sort_values("rpt_time_date")
    
    # 마지막 3일 예측 → test, 그 전날 3일 → valid, 나머지 → train
    test = group.tail(7)
    valid = group.iloc[-14:-7]
    train = group.iloc[:-14]
    
    train_list.append(train)
    valid_list.append(valid)
    test_list.append(test)

train_df = pd.concat(train_list)
valid_df = pd.concat(valid_list)
test_df = pd.concat(test_list)

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)
print("Test shape:", test_df.shape)


# In[61]:


df_feat['mda_idx'].nunique()


# In[62]:


df_feat.columns


# In[63]:


features = ['ads_idx', 'mda_idx',
       'ads_category_id', 'domain', 
       'flag', 'turn_day_count', 
       'turn_sum_lag1', 'turn_sum_lag2', 'turn_sum_lag3', 
       'turn_day_count_lag1', 'turn_day_count_lag2', 'turn_day_count_lag3', 
       'ads_idx_mean_turn_lag1', 'ads_idx_mean_turn_lag2', 'ads_idx_mean_turn_lag3',
       'ads_idx/ads_category_id_mean_turn_lag1', 'ads_idx/ads_category_id_mean_turn_lag2', 'ads_idx/ads_category_id_mean_turn_lag3', 
       'mda_idx_mean_turn_lag1', 'mda_idx_mean_turn_lag2', 'mda_idx_mean_turn_lag3',
       'mda_idx/ads_category_id_mean_turn_lag1', 'mda_idx/ads_category_id_mean_turn_lag2', 'mda_idx/ads_category_id_mean_turn_lag3', 
       'turn_sum_lag_mean', 'lag_amount1', 'lag_amount2', 'due_day', 'year', 'month', 'day']


# In[64]:


# 훈련 데이터 (피처)
X_train = train_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)
# 검증 데이터 (피처)
X_valid = valid_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)
# 테스트 데이터 (피처)
X_test = test_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)

# 훈련 데이터 (타깃값)
y_train = np.log1p(train_df["turn_sum"])
# 검증 데이터 (타깃값)
y_valid = np.log1p(valid_df["turn_sum"])
# 테스트 데이터(타깃값)
y_test = np.log1p(test_df["turn_sum"])


# ## 평가함수

# In[65]:


# 로그변환된 target을 지수변환하기 위한 함수
def exp_transform(y):
    return np.expm1(y)  # log1p -> expm1, 0 값도 안전하게 처리

# MAPE 계산 함수
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100


# ## 모델

# ### randomforest

# In[66]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# 모델 초기화
rf = RandomForestRegressor(random_state=42)

# 1️⃣ Train 데이터 학습
rf.fit(X_train, y_train)

# 2️⃣ Train 데이터 예측
rf_pred_val = rf.predict(X_valid)
rf_pred_val = np.maximum(rf_pred_val, 0)

rf_pred_val_exp = exp_transform(rf_pred_val)
y_val_exp = valid_df["turn_sum"]

# 3️⃣ Train 성능 평가
mse_train = mean_squared_error(y_val_exp, rf_pred_val_exp)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_val_exp, rf_pred_val_exp)
r2_train = r2_score(y_val_exp, rf_pred_val_exp)
mape_train = mean_absolute_percentage_error(y_val_exp, rf_pred_val_exp)

print("valid 성능:")
print(f"R2: {r2_train:.4f}, MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAPE: {mape_train:.2f}%")


# In[67]:


# 4️⃣ Test 데이터 예측
rf_pred_test = rf.predict(X_test)
rf_pred_test = np.maximum(rf_pred_test, 0)

rf_pred_test_exp = exp_transform(rf_pred_test)
y_test_exp = test_df["turn_sum"]

# 5️⃣ Test 성능 평가
mse_test = mean_squared_error(y_test_exp, rf_pred_test_exp)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_exp, rf_pred_test_exp)
r2_test = r2_score(y_test_exp, rf_pred_test_exp)
mape_test = mean_absolute_percentage_error(y_test_exp, rf_pred_test_exp)

print("Test 성능:")
print(f"R2: {r2_test:.4f}, MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%")


# ### XGBoost

# In[68]:


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 모델 초기화
xgb_model = xgb.XGBRegressor(enable_categorical=True,  
                            random_state=42,)

# 1️⃣ Train 데이터 학습
xgb_model.fit(X_train, y_train)

# 2️⃣ Train 데이터 예측
xgb_pred_val = xgb_model.predict(X_valid)

xgb_pred_val_exp = exp_transform(xgb_pred_val)
y_train_exp = valid_df["turn_sum"]

# 3️⃣ Train 성능 평가
mse_train = mean_squared_error(y_train_exp, xgb_pred_val_exp)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_exp, xgb_pred_val_exp)
r2_train = r2_score(y_train_exp, xgb_pred_val_exp)
mape_train = mean_absolute_percentage_error(y_train_exp, xgb_pred_val_exp)

print("valid 성능:")
print(f"R2: {r2_train:.4f}, MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAPE: {mape_train:.2f}%")

# R2: 0.7891, MAE: 0.04, MSE: 1.16, RMSE: 1.08, MAPE: 49148378.12%



# In[69]:


# 4️⃣ Test 데이터 예측
xgb_pred_test = xgb_model.predict(X_test)

xgb_pred_test_exp = exp_transform(xgb_pred_test)
y_test_exp = test_df["turn_sum"]

# 5️⃣ Test 성능 평가
mse_test = mean_squared_error(y_test_exp, xgb_pred_test_exp)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_exp, xgb_pred_test_exp)
r2_test = r2_score(y_test_exp, xgb_pred_test_exp)
mape_test = mean_absolute_percentage_error(y_test_exp, xgb_pred_test_exp)

print("Test 성능:")
print(f"R2: {r2_test:.4f}, MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%")

# R2: 0.6685, MAE: 0.06, MSE: 3.11, RMSE: 1.76, MAPE: 52551856.25%


# ### lgbm

# In[70]:


import lightgbm as lgb

cat_features = ['ads_idx', 'mda_idx', 'domain']

lgb_model = lgb.LGBMRegressor(random_state=42,
                              categorical_feature=cat_features)

# 1️⃣ Train 데이터 학습
lgb_model.fit(X_train, y_train)

# 2️⃣ Train 데이터 예측
lgb_pred_val = lgb_model.predict(X_valid)

lgb_pred_val_exp = exp_transform(lgb_pred_val)
y_train_exp = valid_df["turn_sum"]

# 3️⃣ Train 성능 평가
mse_train = mean_squared_error(y_train_exp, lgb_pred_val_exp)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_exp, lgb_pred_val_exp)
r2_train = r2_score(y_train_exp, lgb_pred_val_exp)
mape_train = mean_absolute_percentage_error(y_train_exp, lgb_pred_val_exp)

print("valid 성능:")
print(f"R2: {r2_train:.4f}, MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAPE: {mape_train:.2f}%")
# R2: 0.7318, MAE: 0.04, MSE: 1.47, RMSE: 1.21, MAPE: 26160021.13%


# In[71]:


# 4️⃣ Test 데이터 예측
lgb_pred_test = lgb_model.predict(X_test)

lgb_pred_test_exp = exp_transform(lgb_pred_test)
y_test_exp = test_df["turn_sum"]

# 5️⃣ Test 성능 평가
mse_test = mean_squared_error(y_test_exp, lgb_pred_test_exp)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_exp, lgb_pred_test_exp)
r2_test = r2_score(y_test_exp, lgb_pred_test_exp)
mape_test = mean_absolute_percentage_error(y_test_exp, lgb_pred_test_exp)

print("Test 성능:")
print(f"R2: {r2_test:.4f}, MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%")
# R2: 0.6670, MAE: 0.06, MSE: 3.12, RMSE: 1.77, MAPE: 28496119.89%


# ## 트리모델 앙상블 

# ### 평균으로

# In[72]:


pred_val_exp = (pred_val_exp + xgb_pred_val_exp + lgb_pred_val_exp) / 3

y_train_exp = valid_df["turn_sum"]

# 3️⃣ Train 성능 평가
mse_train = mean_squared_error(y_train_exp, pred_val_exp)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_exp, pred_val_exp)
r2_train = r2_score(y_train_exp, pred_val_exp)
mape_train = mean_absolute_percentage_error(y_train_exp, pred_val_exp)

print("valid 성능:")
print(f"R2: {r2_train:.4f}, MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAPE: {mape_train:.2f}%")


# In[ ]:


pred_test_exp = (pred_test_exp + xgb_pred_test_exp + lgb_pred_test_exp) / 3

y_test_exp = test_df["turn_sum"]

# 5️⃣ Test 성능 평가
mse_test = mean_squared_error(y_test_exp, pred_test_exp)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_exp, pred_test_exp)
r2_test = r2_score(y_test_exp, pred_test_exp)
mape_test = mean_absolute_percentage_error(y_test_exp, pred_test_exp)

print("Test 성능:")
print(f"R2: {r2_test:.6f}, MAE: {mae_test:.6f}, MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.2f}%")


# ### 가중 평균

# In[79]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def rmse_inverse_weight_blend(y_true, preds_dict):
    # 모델별 RMSE
    errs = {k: np.sqrt(mean_squared_error(y_true, v)) for k, v in preds_dict.items()}
    
    # 역-RMSE 가중치
    inv = {k: 1 / (e + 1e-9) for k, e in errs.items()}
    weights = {k: v / sum(inv.values()) for k, v in inv.items()}
    
    # 최종 블렌딩 예측
    blend = sum(weights[k] * preds_dict[k] for k in preds_dict)
    
    return blend, weights, errs


# ---------------------
# Validation 기반
# ---------------------
y_true_val = y_val_exp
preds_val = {
    "rf":  rf_pred_val_exp,
    "xgb": xgb_pred_val_exp,
    "lgb": lgb_pred_val_exp,
}

blend_val, weights_val, errs_val = rmse_inverse_weight_blend(y_true_val, preds_val)

print("📊 Validation 가중치 기반 Blending")
print("모델별 RMSE:", errs_val)
print("가중치:", weights_val)

# 블렌딩 성능 평가
mse_val = mean_squared_error(y_true_val, blend_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_true_val, blend_val)
r2_val = r2_score(y_true_val, blend_val)
mape_val = mean_absolute_percentage_error(y_true_val, blend_val)

print("Valid 성능:")
print(f"R2: {r2_val:.6f}, MAE: {mae_val:.6f}, MSE: {mse_val:.6f}, RMSE: {rmse_val:.6f}, MAPE: {mape_val:.2f}%")


# ---------------------
# Test 기반
# ---------------------
y_true_test = y_test_exp
preds_test = {
    "rf":  rf_pred_test_exp,
    "xgb": xgb_pred_test_exp,
    "lgb": lgb_pred_test_exp,
}

blend_test, weights_test, errs_test = rmse_inverse_weight_blend(y_true_test, preds_test)

print("\n📊 Test 가중치 기반 Blending")
print("모델별 RMSE:", errs_test)
print("가중치:", weights_test)

# 블렌딩 성능 평가
mse_test = mean_squared_error(y_true_test, blend_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_true_test, blend_test)
r2_test = r2_score(y_true_test, blend_test)
mape_test = mean_absolute_percentage_error(y_true_test, blend_test)

print("Test 성능:")
print(f"R2: {r2_test:.6f}, MAE: {mae_test:.6f}, MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.2f}%")


# In[82]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# ---------------------
# 1️⃣ Validation 데이터에서 Base 모델 예측값 묶기
# ---------------------
meta_X_val = np.vstack([
    xgb_pred_val_exp,  # XGBoost 예측값
    lgb_pred_val_exp   # LightGBM 예측값
]).T  # shape: (n_samples, 3)

meta_y_val = y_val_exp  # Validation 정답

# ---------------------
# 2️⃣ 메타모델 학습 (Ridge)
# ---------------------
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_val, meta_y_val)

print("메타모델 학습 완료 ✅")
print("가중치 (계수):", meta_model.coef_)
print("절편:", meta_model.intercept_)

# ---------------------
# 3️⃣ Test 데이터에서도 Base 모델 예측값 묶기
# ---------------------
meta_X_test = np.vstack([
    xgb_pred_test_exp,
    lgb_pred_test_exp
]).T

# 최종 블렌딩 예측
blend_test = meta_model.predict(meta_X_test)

# ---------------------
# 4️⃣ 성능 평가
# ---------------------
mse_test  = mean_squared_error(y_test_exp, blend_test)
rmse_test = np.sqrt(mse_test)
mae_test  = mean_absolute_error(y_test_exp, blend_test)
r2_test   = r2_score(y_test_exp, blend_test)
mape_test = mean_absolute_percentage_error(y_test_exp, blend_test)

print("\n📊 Meta-model (Ridge) 기반 Stacking 성능")
print(f"R2: {r2_test:.6f}, MAE: {mae_test:.6f}, MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.2f}%")


# In[83]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# # 실제 날짜 생성
# X_test['rpt_time_date'] = pd.to_datetime(X_test[['year','month','day']])

# # 예측값 역변환 (로그 예측일 경우)
# y_pred = np.expm1(lgb_model.predict(X_test))
# y_true = np.expm1(y_test)  # 로그 변환돼 있다면

# 시각화
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 실제 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=y_test_exp, marker='o', color='green', label='Actual')

# 예측 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=blend_test, marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted Turn Count by Date', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn Count', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# #### 가중평균 원본

# In[74]:


w_rf, w_xgb, w_lgb = 0, 0.5, 0.5

# Valid 예측 (가중평균)
pred_val_exp = (
    w_rf  * rf_pred_val_exp +
    w_xgb * xgb_pred_val_exp +
    w_lgb * lgb_pred_val_exp
)

y_val_exp = valid_df["turn_sum"]

# Valid 성능 평가
mse_val = mean_squared_error(y_val_exp, pred_val_exp)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val_exp, pred_val_exp)
r2_val = r2_score(y_val_exp, pred_val_exp)
mape_val = mean_absolute_percentage_error(y_val_exp, pred_val_exp)

print("Valid 성능:")
print(f"R2: {r2_val:.4f}, MAE: {mae_val:.2f}, MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, MAPE: {mape_val:.2f}%")


# In[75]:


# Test 예측 (가중평균)
# --------------------------
pred_test_exp = (
    w_rf  * rf_pred_test_exp +
    w_xgb * xgb_pred_test_exp +
    w_lgb * lgb_pred_test_exp
)

y_test_exp = test_df["turn_sum"]

# Test 성능 평가
mse_test = mean_squared_error(y_test_exp, pred_test_exp)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_exp, pred_test_exp)
r2_test = r2_score(y_test_exp, pred_test_exp)
mape_test = mean_absolute_percentage_error(y_test_exp, pred_test_exp)

print("Test 성능:")
print(f"R2: {r2_test:.6f}, MAE: {mae_test:.6f}, MSE: {mse_test:.6f}, RMSE: {rmse_test:.6f}, MAPE: {mape_test:.2f}%")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# # 실제 날짜 생성
# X_test['rpt_time_date'] = pd.to_datetime(X_test[['year','month','day']])

# # 예측값 역변환 (로그 예측일 경우)
# y_pred = np.expm1(lgb_model.predict(X_test))
# y_true = np.expm1(y_test)  # 로그 변환돼 있다면

# 시각화
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 실제 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=y_test_exp, marker='o', color='green', label='Actual')

# 예측 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=pred_test_exp, marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted Turn Count by Date', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn Count', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# # 실제 날짜 생성
# X_test['rpt_time_date'] = pd.to_datetime(X_test[['year','month','day']])

# # 예측값 역변환 (로그 예측일 경우)
# y_pred = np.expm1(lgb_model.predict(X_test))
# y_true = np.expm1(y_test)  # 로그 변환돼 있다면

# 시각화
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 실제 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=y_valid_orig, marker='o', color='green', label='Actual')

# 예측 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=blend_val_best, marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted Turn Count by Date', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn Count', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ### 스태킹 머시기

# In[ ]:


# 훈련 데이터 (피처)
X_train = train_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)
# 검증 데이터 (피처)
X_valid = valid_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)
# 테스트 데이터 (피처)
X_test = test_df.drop(['turn_sum','rpt_time_date','ads_edate'], axis=1)

# 훈련 데이터 (타깃값)
y_train = np.log1p(train_df["turn_sum"])
# 검증 데이터 (타깃값)
y_valid = np.log1p(valid_df["turn_sum"])
# 테스트 데이터(타깃값)
y_test = np.log1p(test_df["turn_sum"])


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# -------------------------
# Base Models
# -------------------------
# model_xgb = xgb.XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05)
# model_lgb = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)

# -------------------------
# 1. Out-of-Fold 예측 생성
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), 2))  # XGB, LGB 예측값 저장
test_preds = np.zeros((len(X_test), 2))

for i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    # XGB 학습
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[valid_idx, 0] = xgb_model.predict(X_val)
    test_preds[:, 0] += xgb_model.predict(X_test) / kf.n_splits

    # LGB 학습
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_preds[valid_idx, 1] = lgb_model.predict(X_val)
    test_preds[:, 1] += lgb_model.predict(X_test) / kf.n_splits

# -------------------------
# 2. 메타 모델 학습 (Ridge)
# -------------------------
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y_train)

# -------------------------
# 3. 최종 예측
# -------------------------
final_preds = meta_model.predict(test_preds)

# 로그 -> 원래 단위 복원
y_test_original = np.expm1(y_test)
final_preds_original = np.expm1(final_preds)

# 원래 단위로 평가
mse_orig = mean_squared_error(y_test_original, final_preds_original)
rmse_orig = np.sqrt(mse_orig)
mae_orig = mean_absolute_error(y_test_original, final_preds_original)
r2_orig = r2_score(y_test_original, final_preds_original)

print(f"R² (original unit): {r2_orig:.4f}")
print(f"MAE (original unit): {mae_orig:.4f}")
print(f"RMSE (original unit): {rmse_orig:.4f}")
print(f"MSE (original unit): {mse_orig:.4f}")


# In[ ]:


print(y_test[:10])
print(final_preds[:10])


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# # 실제 날짜 생성
# X_test['rpt_time_date'] = pd.to_datetime(X_test[['year','month','day']])

# # 예측값 역변환 (로그 예측일 경우)
# y_pred = np.expm1(lgb_model.predict(X_test))
# y_true = np.expm1(y_test)  # 로그 변환돼 있다면

# 시각화
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 실제 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=y_test_original, marker='o', color='green', label='Actual')

# 예측 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=final_preds_original, marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted Turn Count by Date', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn Count', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


test_check_df = test_df.copy()
test_check_df["y_true"] = y_test
test_check_df["y_pred"] = final_preds
# ads_idx, mda_idx 조건 필터링
cond_df = test_check_df[(test_check_df["ads_idx"] == 356956) & (test_check_df["mda_idx"] == 270)]

# 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=cond_df['rpt_time_date'], y=cond_df['y_true'], marker='o', color='green', label='Actual')
sns.lineplot(x=cond_df['rpt_time_date'], y=cond_df['y_pred'], marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted (ads_idx=356956, mda_idx=270)', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn sum', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ## 실제 vs 이동평균vs 모델

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 예시: test_df 안에 7일치 날짜가 들어있다고 가정
# test_df['rpt_time_date'] = pd.date_range(start="2025-08-29", periods=7, freq="D")

# 데이터프레임 구성
plot_df = pd.DataFrame({
    "Date": test_df['rpt_time_date'],
    "Actual": y_test,
    "Ensemble": final_preds,
    "MovingAverage": all_test_preds
})

# 시각화
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")

# 실제값
sns.lineplot(x="Date", y="Actual", data=plot_df, marker='o',
             color="black", label="실제값 (y_test)")

# 앙상블
sns.lineplot(x="Date", y="Ensemble", data=plot_df, marker='s',
             color="#2E8B57", label="앙상블 (final_preds)")

# 이동평균
sns.lineplot(x="Date", y="MovingAverage", data=plot_df, marker='^',
             color="#E9353E", label="이동평균 (test_pred)")

plt.title("7일 결과 비교: 실제값 vs 앙상블 vs 이동평균", fontsize=15)
plt.xlabel("날짜", fontsize=13)
plt.ylabel("전환수", fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


print(len(test_df['rpt_time_date']))
print(len(y_test))
print(len(final_preds))
print(len(test_pred))


# ## 스트림릿 연동 부분

# In[ ]:





# In[ ]:


pirnt)tt


# In[ ]:


def make_future_testset_meta(xgb_model, lgb_model, meta_model, df, start_date="2025-08-30", days=7):
    results = []

    feature_cols = [
        'unique_id','ads_idx','mda_idx','ads_category_id','domain','flag','turn_day_count',
        'turn_sum_lag1','turn_sum_lag2','turn_sum_lag3',
        'turn_day_count_lag1','turn_day_count_lag2','turn_day_count_lag3',
        'ads_idx_mean_turn_lag1','ads_idx_mean_turn_lag2','ads_idx_mean_turn_lag3',
        'ads_idx/ads_category_id_mean_turn_lag1','ads_idx/ads_category_id_mean_turn_lag2','ads_idx/ads_category_id_mean_turn_lag3',
        'mda_idx_mean_turn_lag1','mda_idx_mean_turn_lag2','mda_idx_mean_turn_lag3',
        'mda_idx/ads_category_id_mean_turn_lag1','mda_idx/ads_category_id_mean_turn_lag2','mda_idx/ads_category_id_mean_turn_lag3',
        'turn_sum_lag_mean','lag_amount1','lag_amount2','due_day','year','month','day'
    ]

    base_cols = [
        "turn_sum", "turn_day_count",
        "ads_idx_mean_turn", "ads_idx/ads_category_id_mean_turn",
        "mda_idx_mean_turn", "mda_idx/ads_category_id_mean_turn"
    ]

    for uid, group in df.groupby("unique_id"):
        group = group.sort_values("rpt_time_date")
        last3 = group.tail(3)

        if last3.empty:
            continue

        # --- 최근 3일 기반 lag dict ---
        lag_dict = {}
        for col in base_cols:
            vals = list(last3[col].values) if col in last3 else [0]
            while len(vals) < 3:
                vals.insert(0, 0)
            lag_dict[col] = vals  

        template = last3.iloc[-1].copy()
        future_dates = pd.date_range(start=start_date, periods=days)

        for date in future_dates:
            new_row = template.copy()
            new_row["rpt_time_date"] = date
            new_row["year"], new_row["month"], new_row["day"] = date.year, date.month, date.day

            # lag 채우기
            for col in base_cols:
                vals = lag_dict[col]
                new_row[f"{col}_lag1"] = vals[-1]
                new_row[f"{col}_lag2"] = vals[-2]
                new_row[f"{col}_lag3"] = vals[-3]

            # 추가 파생
            ts1, ts2, ts3 = lag_dict["turn_sum"][-1], lag_dict["turn_sum"][-2], lag_dict["turn_sum"][-3]
            new_row["turn_sum_lag_mean"] = np.mean([ts1, ts2, ts3])
            new_row["lag_amount1"] = ts1 / ts2 if ts2 != 0 else 0
            new_row["lag_amount2"] = ts2 / ts3 if ts3 != 0 else 0

            # -------------------------------
            # 1단계: Base 모델 예측 (XGB, LGB)
            # -------------------------------
            X_test = pd.DataFrame([new_row[feature_cols].values], columns=feature_cols)

            # 범주형 처리
            X_test['ads_idx'] = X_test['ads_idx'].astype('category')
            X_test['mda_idx'] = X_test['mda_idx'].astype('category')
            X_test['domain']  = X_test['domain'].astype('category')

            pred_xgb = xgb_model.predict(X_test)[0]
            pred_lgb = lgb_model.predict(X_test)[0]

            # -------------------------------
            # 2단계: 메타모델 입력 (XGB+LGB → Ridge)
            # -------------------------------
            meta_input = np.array([[pred_xgb, pred_lgb]])
            y_pred = meta_model.predict(meta_input)[0]

            # 후처리
            y_pred = np.maximum(y_pred, 0)
            y_pred = np.expm1(y_pred)

            # -------------------------------
            # 결과 저장 + lag 업데이트
            # -------------------------------
            new_row["turn_sum"] = y_pred
            results.append(new_row)

            lag_dict["turn_sum"] = [lag_dict["turn_sum"][-2], lag_dict["turn_sum"][-1], y_pred]

    return pd.DataFrame(results)


# In[ ]:


future_df = make_future_testset_meta(
    xgb_model, lgb_model, meta_model,
    df_feat, start_date="2025-08-30", days=7
)

print(future_df.head())


# In[ ]:


df3_1_2 = future_df[['unique_id','rpt_time_date','turn_sum','ads_idx','mda_idx']].copy()


# In[ ]:


def downcast(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype_name = df[col].dtype.name

        # object, datetime은 건너뛰기
        if dtype_name == 'object' or 'datetime' in dtype_name:
            continue
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif dtype_name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.1f}% 압축됨'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


df3_1_2 = downcast(df3_1_2)


# In[ ]:


df3_1_2.to_csv('df3_1_2.csv', index=False)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. 모델 학습
# lgb_model.fit(X_train, y_train)

# 2. 모델 저장
joblib.dump(lgb_model, "lgbm_0925.pkl")
print("모델이 my_model.pkl 파일로 저장되었습니다.")


# In[ ]:


df_feat = df_feat.drop(columns=['ads_edate'])


# In[ ]:


def make_future_testset(model, df, start_date="2025-08-30", days=7):
    results = []

    feature_cols = [
        'unique_id','ads_idx','mda_idx','ads_category_id','domain','flag','turn_day_count',
        'turn_sum_lag1','turn_sum_lag2','turn_sum_lag3',
        'turn_day_count_lag1','turn_day_count_lag2','turn_day_count_lag3',
        'ads_idx_mean_turn_lag1','ads_idx_mean_turn_lag2','ads_idx_mean_turn_lag3',
        'ads_idx/ads_category_id_mean_turn_lag1','ads_idx/ads_category_id_mean_turn_lag2','ads_idx/ads_category_id_mean_turn_lag3',
        'mda_idx_mean_turn_lag1','mda_idx_mean_turn_lag2','mda_idx_mean_turn_lag3',
        'mda_idx/ads_category_id_mean_turn_lag1','mda_idx/ads_category_id_mean_turn_lag2','mda_idx/ads_category_id_mean_turn_lag3',
        'turn_sum_lag_mean','lag_amount1','lag_amount2','due_day','year','month','day'
    ]

    base_cols = [
        "turn_sum", "turn_day_count",
        "ads_idx_mean_turn", "ads_idx/ads_category_id_mean_turn",
        "mda_idx_mean_turn", "mda_idx/ads_category_id_mean_turn"
    ]

    for uid, group in df.groupby("unique_id"):
        group = group.sort_values("rpt_time_date")
        last3 = group.tail(3)

        if last3.empty:
            continue

        # base_cols별로 최근 3일 값 확보 (부족하면 0으로 채움)
        lag_dict = {}
        for col in base_cols:
            vals = list(last3[col].values) if col in last3 else [0]
            while len(vals) < 3:
                vals.insert(0, 0)
            lag_dict[col] = vals  # [day-2, day-1, day0]

        template = last3.iloc[-1].copy()
        future_dates = pd.date_range(start=start_date, periods=days)

        for date in future_dates:
            new_row = template.copy()
            new_row["rpt_time_date"] = date
            new_row["year"], new_row["month"], new_row["day"] = date.year, date.month, date.day

            # lag 매핑
            for col in base_cols:
                vals = lag_dict[col]
                new_row[f"{col}_lag1"] = vals[-1]
                new_row[f"{col}_lag2"] = vals[-2]
                new_row[f"{col}_lag3"] = vals[-3]

            # turn_sum 기반 추가 파생
            ts1, ts2, ts3 = lag_dict["turn_sum"][-1], lag_dict["turn_sum"][-2], lag_dict["turn_sum"][-3]
            new_row["turn_sum_lag_mean"] = np.mean([ts1, ts2, ts3])
            new_row["lag_amount1"] = ts1 / ts2 if ts2 != 0 else 0
            new_row["lag_amount2"] = ts2 / ts3 if ts3 != 0 else 0

            # 예측
            X_test = pd.DataFrame([new_row[feature_cols].values], columns=feature_cols)

            X_test['ads_idx'] = X_test['ads_idx'].astype('category')
            X_test['mda_idx'] = X_test['mda_idx'].astype('category')
            X_test['domain']  = X_test['domain'].astype('category')

            y_pred = model.predict(X_test)[0]
            y_pred = np.maximum(y_pred, 0)
            y_pred = np.expm1(y_pred)
            
            # 결과 저장
            new_row["turn_sum"] = y_pred
            # new_row["prediction"] = y_pred
            results.append(new_row)

            # lag_dict 업데이트 (예측값을 turn_sum에 반영)
            lag_dict["turn_sum"] = [lag_dict["turn_sum"][-2], lag_dict["turn_sum"][-1], y_pred]
            for col in base_cols:
                if col != "turn_sum":  # 다른 base 컬럼들은 그대로 유지 (고정)
                    pass

    return pd.DataFrame(results)


# In[ ]:


future_df = make_future_testset(meta_model, df_feat, start_date="2025-08-30", days=7)

# print(future_df[["unique_id","rpt_time_date","turn_sum_lag1","turn_sum_lag2","turn_sum_lag3","prediction"]].head(15))


# In[ ]:


future_df['turn_sum_fin'] = pd.to_numeric(future_df['turn_sum'], errors='coerce').astype('int')


# In[ ]:


df3_1 = future_df[['unique_id','rpt_time_date','turn_sum','ads_idx','mda_idx']].copy()


# In[ ]:


def downcast(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dtype_name = df[col].dtype.name

        # object, datetime은 건너뛰기
        if dtype_name == 'object' or 'datetime' in dtype_name:
            continue
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif dtype_name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.1f}% 압축됨'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


df3_1 = downcast(df3_1)


# In[ ]:


# import pandas as pd
# import numpy as np

# def reduce_memory_usage(df):
#     start_mem = df.memory_usage().sum() / 1024**2
#     print(f"초기 메모리 사용량: {start_mem:.2f} MB")

#     for col in df.columns:
#         col_type = df[col].dtype

#         if col_type != object and str(col_type) != "datetime64[ns]":
#             c_min = df[col].min()
#             c_max = df[col].max()

#             if str(col_type).startswith("int"):
#                 if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 else:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)

#     end_mem = df.memory_usage().sum() / 1024**2
#     print(f"최종 메모리 사용량: {end_mem:.2f} MB (절감률 {(100*(start_mem-end_mem)/start_mem):.1f}%)")
#     return df


# df3_1 = reduce_memory_usage(df3_1)


# In[ ]:


future_df.info()


# In[ ]:


# parquet (컬럼형 저장, 효율 최고)
# future_df.to_parquet("df3_1.parquet", index=False, compression="snappy")


# In[ ]:


df3_1.to_csv('df3_1.csv', index=False)


# ## 잡다구리

# In[ ]:


# 피처 중요도 추출
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importance)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.gca().invert_yaxis()  # 중요도 높은 피처가 위로 오도록
plt.title("Feature Importance")
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 예측 (로그값)
y_pred_log = xgb_model.predict(X_test)
# 로그 역변환
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
# 데이터프레임 생성
results = pd.DataFrame({
    'Actual': y_true,
    'Predicted': y_pred
})
# 시각화
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
sns.scatterplot(
    x='Actual', y='Predicted', data=results,
    alpha=0.4, color='royalblue'
)
# 완벽한 예측선 (y = x)
min_val = min(results['Actual'].min(), results['Predicted'].min())
max_val = max(results['Actual'].max(), results['Predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.title('Actual vs Predicted', fontsize=16)
plt.xlabel('Actual Values', fontsize=13)
plt.ylabel('Predicted Values', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# y_pred를 X_test에 붙이기
X_test_chek = test_df.copy()  # 안전하게 원본 복사
X_test_chek['pred_turn_sum'] = y_pred


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# # 실제 날짜 생성
# X_test['rpt_time_date'] = pd.to_datetime(X_test[['year','month','day']])

# # 예측값 역변환 (로그 예측일 경우)
# y_pred = np.expm1(lgb_model.predict(X_test))
# y_true = np.expm1(y_test)  # 로그 변환돼 있다면

# 시각화
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 실제 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=y_test_exp, marker='o', color='green', label='Actual')

# 예측 값 선
sns.lineplot(x=test_df['rpt_time_date'], y=pred_test_exp, marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted Turn Count by Date', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn Count', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


test_check_df = test_df.copy()
test_check_df["y_true"] = y_test_exp
test_check_df["y_pred"] = pred_test_exp


# In[ ]:


3569560270


# In[ ]:


test_check_df = test_df.copy()
test_check_df["y_true"] = y_test_exp
test_check_df["y_pred"] = pred_test_exp
# ads_idx, mda_idx 조건 필터링
cond_df = test_check_df[(test_check_df["ads_idx"] == 356956) & (test_check_df["mda_idx"] == 270)]

# 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=cond_df['rpt_time_date'], y=cond_df['y_true'], marker='o', color='green', label='Actual')
sns.lineplot(x=cond_df['rpt_time_date'], y=cond_df['y_pred'], marker='o', color='blue', label='Predicted')

plt.title('Actual vs Predicted (ads_idx=356956, mda_idx=270)', fontsize=16)
plt.xlabel('Date', fontsize=13)
plt.ylabel('Turn sum', fontsize=13)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


cond_df[['rpt_time_date', 'y_true', 'y_pred']]


# In[ ]:


mix_df= test_check_df.groupby(['ads_idx', 'mda_idx']).apply(
    lambda x: np.abs(x['y_true'] - x['y_pred']).sum()
).reset_index(name='abs_error_sum').sort_values(by='abs_error_sum', ascending=False)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# flag == 1인 데이터만 필터링
df_flag = test_check_df[test_check_df["flag"] == 1]

# 실제값과 모델 예측값 컬럼 이름 예시
# (이미 df 안에 y_true, y_pred_A, y_pred_B 같은 컬럼이 있다고 가정)
y_true = df_flag["y_true"].values
y_pred_A = df_flag["y_pred"].values   # 전체 학습 모델

# 성능지표 계산 함수
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "R2": f"{r2:.4f}"}

# 두 모델 결과 비교
metrics_A = get_metrics(y_true, y_pred_A)

print("모델 A (전체 학습, 광고한 날 평가):", metrics_A)


# In[ ]:


# 30일 모두 기록이 있는 광고만 필터링

# 날짜 변환
df_day["rpt_time_date"] = pd.to_datetime(df_day["rpt_time_date"])

# 기준일과 최근 30일 구간
max_date = df_day["rpt_time_date"].max()
cutoff_date = max_date - pd.Timedelta(days=29)   # 30일 포함 위해 -29

# 최근 30일 날짜 리스트
recent_days = pd.date_range(start=cutoff_date, end=max_date, freq="D")

# unique_id별 날짜 수 세기
id_date_counts = (
    df_day[df_day["rpt_time_date"].between(cutoff_date, max_date)]
    .groupby("unique_id")["rpt_time_date"]
    .nunique()
)

# 30일 모두 있는 unique_id만 추출
valid_ids = id_date_counts[id_date_counts == len(recent_days)].index

# 최종 필터링
test_30_ads = test_check_df[test_check_df["unique_id"].isin(valid_ids)].copy()


# In[ ]:


# (이미 df 안에 y_true, y_pred_A, y_pred_B 같은 컬럼이 있다고 가정)
y_true = test_30_ads["y_true"].values
y_pred_A = test_30_ads["y_pred"].values   # 전체 학습 모델

# 성능지표 계산 함수
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "R2": f"{r2:.4f}"}

# 두 모델 결과 비교
metrics_A = get_metrics(y_true, y_pred_A)

print("모델 A (전체 학습, 광고한 날 평가):", metrics_A)


# In[ ]:




