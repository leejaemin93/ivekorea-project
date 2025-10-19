# 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import re
import platform
import matplotlib.pyplot as plt
import holidays
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
import matplotlib.patheffects as path_effects


# 한글 오류 제거 
# 1. 한글 폰트 설정
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux (예: colab)
    plt.rcParams['font.family'] = 'NanumGothic'

# 2. 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False





# -------------------------------------------------------------------------
# 광고 목록 데이터 불러오기 & 독립변수 추가
# -------------------------------------------------------------------------

# --- 광고 목록 테이블 불러오기 ---
ads_list = pd.read_csv("df_list_v1.csv", usecols=['ads_idx', 'ads_edate', 'aff_idx', 'ads_name', 'ads_type', 'ads_category', 'ads_save_way', 'ads_limit', 'ads_os_type', 'ads_payment', 'ads_summary', 'ads_rejoin_type'])
ads_list['ads_edate'] = pd.to_datetime(ads_list['ads_edate'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 광고기간이 결측값인 행은 '2262-04-11'으로 채우기
# '2262-04-11' : pandas의 datetime64[ns]가 다룰 수 있는 범위
ads_list['ads_edate'] = ads_list['ads_edate'].fillna(pd.to_datetime("2262-04-11"))

# 제휴사 광고를 제외한 아이브 광고만 선택
ads_list = ads_list[ads_list['aff_idx'] == 1]

# 테스트 광고 수정 필요 -> 온리 테이트가 있다고 테스트 광고가 아님
is_test_ad = ads_list["ads_name"].str.contains("테스트|서비스종료|삭제", na=False, case=False)
is_valid_test_ad = ads_list["ads_name"].str.contains("브레인 테스트|게임테스트", na=False, case=False)
condition_to_delete = is_test_ad & ~is_valid_test_ad
ads_list = ads_list[~condition_to_delete].copy()



# --- 도메인 컬럼 추가 ---
# 분류 함수 (제외 키워드 적용)

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

# 함수 사용

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



# --- 광고 단계별 분류 ---
conditions = [
    # 3단계: 최종 수익 창출 (구매, 게임(특정퀘스트))
    (ads_list['ads_type'].isin([9, 12])) | (ads_list['ads_category'].isin([5, 6, 10, 11])),
    # 2단계: 행동 유도 (설치, 실행, 참여, 퀴즈, 구독 등)
    (ads_list['ads_type'].isin([1, 2, 3, 7, 11])) | (ads_list['ads_category'].isin([1, 2, 3, 4, 7, 8])),
    # 1단계: 단순 노출 및 클릭
    (ads_list['ads_type'].isin([4, 5, 6, 8, 10]))
]
values = [3, 2, 1]

# 새로운 컬럼 생성
ads_list['ads_3step'] = np.select(conditions, values, default=0)



# --- 앱/웹 광고 구분 ---
# ads_os_type : (7:웹) -> 1, (그외:앱) -> 0
ads_list['ads_os_type'] = ads_list['ads_os_type'].apply(lambda x: 1 if x==7 else 0).astype(int)



# --- 유저 광고 참여 비용 ---
# 1. 단순 유료 광고 식별
is_unspecified_paid = ads_list['ads_payment'].astype(str).str.contains('유료|구매|결제', na=False)

# 2. 만,천 단위 변환 함수
def convert_korean_units(text):
    text = str(text)
    if '만' in text:
        number = re.search(r'\d+', text)
        return int(number.group()) * 10000 if number else text
    elif '천' in text:
        number = re.search(r'\d+', text)
        return int(number.group()) * 1000 if number else text
    return text

ads_list["ads_payment"] = ads_list["ads_payment"].apply(convert_korean_units)

# 3. 정규표현식을 통한 괄호, 문자 등 제거
# r'\(.*?\)'는 괄호와 괄호 안의 내용을 제거
# r'[^0-9.]'는 숫자 외의 모든 글자를 제거
ads_list["ads_payment"] = ads_list["ads_payment"].astype(str).str.replace(r'\(.*?\)', '', regex=True)
ads_list["ads_payment"] = ads_list["ads_payment"].str.replace(r'[^0-9.]', '', regex=True)

# 4. 숫자형이 아닌 값들을 NaN으로 변환 (to_numeric함수는 문자형이 아닌 값들을 숫자형으로 변환, errors='coerce'는 변환 불가한 값들을 NaN으로 변환)
ads_list["ads_payment"] = pd.to_numeric(ads_list["ads_payment"], errors='coerce')
ads_list['ads_payment'] = ads_list['ads_payment'].fillna(0).astype(float)



# --- 광고 길이 컬럼 추가 ---
ads_list['ads_length'] = ads_list['ads_summary'].str.len()



# --- 나이 제한, 성별 제한 ---
# 결측값 제거
ads_list['ads_limit'] = ads_list['ads_limit'].fillna('제한없음')

# 나이 관련 제한 추가 (제한O : 1, 제한X : 0)
age_pattern = re.compile(r'(\d+세|\d+대|\d+~\d+세|\d세+~\d+세)')
ads_list['age_limit'] = ads_list['ads_limit'].astype(str).apply(lambda x: 1 if re.search(age_pattern, x) else 0).astype(int)

# 성별 관련 제한 추가 (제한O : 1, 제한X : 0)
gender_pattern = re.compile(r'(남성|여성|남녀)')
ads_list['gender_limit'] = ads_list['ads_limit'].astype(str).apply(lambda x: 1 if re.search(gender_pattern, x) else 0).astype(int)


  


# -------------------------------------------------------------------------
# 아이브1년치_참여데이터 불러오기
# -------------------------------------------------------------------------

# 참여데이터 불러오기
ive_time_report = pd.read_csv("df_rpt_clean.csv")
ive_time_report['rpt_time_date'] = pd.to_datetime(ive_time_report['rpt_time_date'], format='%Y-%m-%d', errors='coerce')

# 클릭수보다 전환수가 더 많은 행 제거하기
ive_time_report = ive_time_report[ive_time_report['rpt_time_clk'] >= ive_time_report['rpt_time_turn']]

# 참여데이터 날짜별로 그룹화하기
ive_time_report = ive_time_report.groupby(['rpt_time_date', 'ads_idx', 'mda_idx'], as_index=False).agg({'rpt_time_clk': 'sum', 'rpt_time_turn':'sum', 'rpt_time_acost':'sum', 'rpt_time_earn':'sum'})





# -------------------------------------------------------------------------
# 전처리한 아이브1년치_참여데이터에 독립변수 만들기
# -------------------------------------------------------------------------

filled_ive_time_report = ive_time_report.copy()

# --- 매체별 평균 광고 단가 ---
# 현재 행의 값은 포함하지 않고 이전 단가까지만 포함
# 그룹의 크기가 1보다 클 때만 계산하고, 1일 경우에는 0으로 채워줌

filled_ive_time_report['mda_mean_acost'] = (
    filled_ive_time_report.groupby('mda_idx')['rpt_time_acost']
      .transform(lambda x: x.shift().expanding().mean())
)

filled_ive_time_report['mda_mean_acost'] = filled_ive_time_report['mda_mean_acost'].fillna(0)



# --- 매체별 평균 매체사 단가 ---
# 현재 행의 값은 포함하지 않고 이전 단가까지만 포함
# 그룹의 크기가 1보다 클 때만 계산하고, 1일 경우에는 0으로 채워줌

filled_ive_time_report['mda_mean_earn'] = (
    filled_ive_time_report.groupby('mda_idx')['rpt_time_earn']
      .transform(lambda x: x.shift().expanding().mean())
)

filled_ive_time_report['mda_mean_earn'] = filled_ive_time_report['mda_mean_earn'].fillna(0)



# --- 매체별 평균 클릭수, 전환수 컬럼 만들기 ---
# 현재 행의 값은 포함하지 않고 이전 클릭수, 전환수만 포함

# 매체별 과거까지의 클릭수 평균
filled_ive_time_report['mda_mean_clk'] = (
    filled_ive_time_report.groupby('mda_idx')['rpt_time_clk']
      .transform(lambda x: x.shift().expanding().mean())
)

# 매체별 과거까지의 전환수 평균
filled_ive_time_report['mda_mean_turn'] = (
    filled_ive_time_report.groupby('mda_idx')['rpt_time_turn']
      .transform(lambda x: x.shift().expanding().mean())
)

# 평균은 첫 값이 결측치로 저장됨
# 따라서 결측치는 0으로 저장
filled_ive_time_report[['mda_mean_clk','mda_mean_turn']] = (
    filled_ive_time_report[['mda_mean_clk','mda_mean_turn']].fillna(0)
)



# --- 매체별 누적 광고 비용 비율 ---

# 현재 행을 제외한 과거까지의 광고 비용
filled_ive_time_report['mda_cum_acost'] = (
    filled_ive_time_report.groupby('mda_idx')['rpt_time_acost']
      .transform(lambda x: x.cumsum() - x)
)

# 현재 행을 제외한 과거까지의 전체 누적 비용
filled_ive_time_report['global_cum_acost'] = filled_ive_time_report['rpt_time_acost'].cumsum() - ive_time_report['rpt_time_acost']

# 현재 행을 제외한 매체별 비용 비율
filled_ive_time_report['mda_cost_ratio'] = (
    (filled_ive_time_report['mda_cum_acost'] / filled_ive_time_report['global_cum_acost']).fillna(0)
)
filled_ive_time_report = filled_ive_time_report.drop(['mda_cum_acost', 'global_cum_acost'], axis=1)



# --- 월 컬럼 추가하기 ---
filled_ive_time_report['month'] = filled_ive_time_report['rpt_time_date'].dt.month



# --- 분기 컬럼 추가하기 ---
filled_ive_time_report['quarter'] = (filled_ive_time_report['month'] - 1) // 3 + 1



# --- 월초, 월말 컬럼 ---

# '월초' 컬럼 생성 (날짜가 10일 이하이면 1, 아니면 0)
filled_ive_time_report['is_month_start'] = (filled_ive_time_report['rpt_time_date'].dt.day <= 10).astype(int)

# '월말' 컬럼 생성 (날짜가 25일 이상이면 1, 아니면 0)
filled_ive_time_report['is_month_end'] = (filled_ive_time_report['rpt_time_date'].dt.day >= 25).astype(int)



# --- 평일에 공휴일이 있는 주차 표시 ---
# 대한민국 공휴일 데이터
kr_holidays = holidays.KR(years=[2024, 2025])

# 날짜가 '평일 공휴일'인지 확인하는 함수
def check_weekday_holiday(date):
    # isoweekday()는 월요일=1, 화요일=2, ..., 일요일=7
    is_weekday = date.isoweekday() <= 5 
    is_holiday = date in kr_holidays
    
    if is_weekday and is_holiday:
        return 1
    else:
        return 0

# '평일 공휴일 여부' 컬럼 추가
filled_ive_time_report['is_weekday_holiday'] = filled_ive_time_report['rpt_time_date'].apply(check_weekday_holiday).astype(int)





# -------------------------------------------------------------------------
# 광고목록과 1년치 데이터 merge & 독립변수 추가
# -------------------------------------------------------------------------

# --- 광고목록에 존재하는 광고만 아이브1년치_참여데이터에서 가져오기 ---
merge_data = filled_ive_time_report.merge(ads_list, on='ads_idx', how='inner')
merge_data = merge_data.drop(['aff_idx', 'ads_type', 'ads_category', 'ads_name', 'ads_summary', 'ads_limit', 'ads_save_way', 'ads_edate'], axis=1)



# --- 각 조합별 count가 5 미만인 경우 리스트로 만들기 ---

# 적용할 모든 조합
all_group_sets = [
    ['domain'],
    ['ads_3step'],
    ['ads_os_type'],
    ['mda_idx'],
    ['domain','mda_idx'],
    ['domain', 'ads_os_type'],
    ['domain', 'ads_3step'],
    ['ads_3step', 'ads_os_type'],
    ['ads_3step', 'mda_idx'],
    ['ads_os_type', 'mda_idx'],
    ['domain', 'ads_3step', 'ads_os_type'],
    ['domain', 'ads_3step', 'mda_idx'],
    ['ads_3step', 'ads_os_type', 'mda_idx']
]

# count < 5 인 조합 이름만 담을 리스트
need_flag_sets = []

for cols in all_group_sets:
    group_counts = merge_data.groupby(cols).size()
    small_groups = group_counts[group_counts < 5].index.tolist()
    
    if small_groups:
        need_flag_sets.append(cols)



# --- 여러 조합별 turn의 하루 평균, 전환율 ---
# 전환수 합계를 비교하면 최신 매체는 과소평가되므로 하루 평균 전환수를 함께 사용한다
# 각 조합이 처음 등장한 경우에는 플래그 변수로 구분해줌

for cols in all_group_sets:
    name = '_'.join(cols)

    # 활동일수: 현재까지 등장 횟수 (0부터 시작) 
    merge_data[f'{name}_age_days_tmp'] = (
        merge_data.groupby(cols, observed=False).cumcount()
    )

    # acost, earn → 과거까지 평균 (현재 제외)
    for var in ['rpt_time_acost','rpt_time_earn']:
        short = var.replace('rpt_time_', '')
        merge_data[f'{name}_{short}_mean'] = (
            merge_data.groupby(cols, observed=False)[var]
              .transform(lambda x: x.shift().expanding().mean())
        )

    # clk → 과거까지 합계 (현재 제외)
    merge_data[f'{name}_clk_sum'] = (
        merge_data.groupby(cols, observed=False)['rpt_time_clk']
          .transform(lambda x: x.cumsum() - x)
    ) 

    # turn → 과거까지 합계 (현재 제외)
    merge_data[f'{name}_turn_sum'] = (
        merge_data.groupby(cols, observed=False)['rpt_time_turn']
          .transform(lambda x: x.cumsum() - x)
    )
    
    # cvr → 과거까지 합계 (현재 제외)
    merge_data[f'{name}_cvr'] = np.where(
        merge_data[f'{name}_clk_sum'] > 0,
        merge_data[f'{name}_turn_sum'] / merge_data[f'{name}_clk_sum'],
        np.nan
    )

    # turn_per_day = turn_sum / age_days (현재 제외)
    merge_data[f'{name}_turn_per_day'] = (
        merge_data[f'{name}_turn_sum'] /
        merge_data[f'{name}_age_days_tmp'].replace(0, np.nan)
    )

    # count<5 → NaN처리 + 플래그 (희소조합만)
    if cols in need_flag_sets:
        counts = merge_data.groupby(cols, observed=False)['rpt_time_turn'].transform('count')
        mask = counts < 5
        merge_data.loc[mask, [
            f'{name}_acost_mean',
            f'{name}_earn_mean',
            f'{name}_turn_per_day',
            f'{name}_cvr']] = np.nan
        merge_data[f'is_small_{name}'] = mask.astype(int)

    # 조합별 첫 등장 여부 플래그
    merge_data[f'is_first_{name}'] = (
        merge_data.groupby(cols, observed=False).cumcount() == 0
    ).astype(int)

    # 불필요한 컬럼 제거
    merge_data.drop(columns=[f'{name}_age_days_tmp', f'{name}_turn_sum', f'{name}_clk_sum'], inplace=True)

# NaN → 0 채우기 
num_cols = [
    c for c in merge_data.columns 
    if c.endswith(('_acost_mean','_earn_mean', '_turn_per_day'))
]
merge_data[num_cols] = merge_data[num_cols].fillna(0)



# -------------------------------------------------------------------------
# 스트림릿용 조합별 평균값 추가
# -------------------------------------------------------------------------
cols = ['domain', 'ads_3step', 'mda_idx']
name = '_'.join(cols)

vars_to_calculate = ['rpt_time_clk', 'rpt_time_turn']

for var in vars_to_calculate:
    short_name = var.replace('rpt_time_', '')
    
    new_column_name = f'{name}_{short_name}_mean'
    
    merge_data[new_column_name] = (
        merge_data.groupby(cols, observed=False)[var]
                  .transform(lambda x: x.shift().expanding().mean())
    )



# -------------------------------------------------------------------------
# 주차별로 1년치 데이터 그룹화
# -------------------------------------------------------------------------

# --- 그룹화 집계 함수 ---
# 광고별로 rpt_time_date의 첫 날을 광고 시작 날짜로 지정
merge_data['start_date'] = merge_data.groupby(['ads_idx'], observed=True)['rpt_time_date'].transform('min')
merge_data['start_date'] = pd.to_datetime(merge_data['start_date'])

# start_date 기준으로 week 계산
merge_data['days_from_start'] = (merge_data['rpt_time_date'] - merge_data['start_date']).dt.days
merge_data['week'] = (merge_data['days_from_start'] // 7) + 1

# 불필요한 컬럼 제거
merge_data = merge_data.drop(['rpt_time_date', 'days_from_start', 'start_date'], axis=1)

# ads_idx, week, mda_idx 기준으로 그룹화
merge_data = merge_data.groupby(['ads_idx', 'week', 'mda_idx'], observed=True)

# 집계함수
agg_dict={}

# y 변수 -> sum
for col in ['rpt_time_turn', 'rpt_time_clk']:
    agg_dict[col] = 'sum'

# 광고별 특성 -> first
for col in ['domain', 'ads_3step', 'ads_rejoin_type', 'ads_os_type', 'ads_payment', 'ads_length', 'age_limit', 'gender_limit']:
    agg_dict[col] = 'first'

# 날짜/시간 특성 -> first
for col in ['month', 'quarter', 'is_month_start', 'is_month_end', 'is_weekday_holiday']:
    agg_dict[col] = 'first'

# 매체 특성 -> last
for col in ['mda_mean_acost', 'mda_mean_earn', 'mda_mean_clk', 'mda_mean_turn', 'mda_cost_ratio']:
    agg_dict[col] = 'last'

# 플래그 변수 -> max
for col in ['is_first_domain', 'is_first_ads_3step', 'is_first_ads_os_type',
            'is_small_mda_idx', 'is_first_mda_idx', 'is_small_domain_mda_idx', 'is_first_domain_mda_idx',
            'is_first_domain_ads_os_type', 'is_first_domain_ads_3step', 'is_small_ads_3step_mda_idx', 'is_first_ads_3step_ads_os_type', 
            'is_small_ads_os_type_mda_idx', 'is_first_ads_3step_mda_idx', 
            'is_first_ads_os_type_mda_idx', 'is_first_domain_ads_3step_ads_os_type',
            'is_small_domain_ads_3step_mda_idx', 'is_first_domain_ads_3step_mda_idx',
            'is_small_ads_3step_ads_os_type_mda_idx', 'is_first_ads_3step_ads_os_type_mda_idx']:
    agg_dict[col] = 'max'

# 조합 특성 -> last
for col in ['domain_acost_mean', 'domain_earn_mean', 'domain_cvr', 'domain_turn_per_day', 
            'ads_3step_acost_mean', 'ads_3step_earn_mean', 'ads_3step_cvr', 'ads_3step_turn_per_day', 
            'ads_os_type_acost_mean', 'ads_os_type_earn_mean', 'ads_os_type_cvr', 'ads_os_type_turn_per_day', 
            'mda_idx_cvr', 'mda_idx_turn_per_day', 
            'domain_mda_idx_acost_mean', 'domain_mda_idx_earn_mean', 'domain_mda_idx_cvr', 'domain_mda_idx_turn_per_day',
            'domain_ads_os_type_acost_mean', 'domain_ads_os_type_earn_mean', 'domain_ads_os_type_cvr', 'domain_ads_os_type_turn_per_day',
            'domain_ads_3step_acost_mean', 'domain_ads_3step_earn_mean', 'domain_ads_3step_cvr', 'domain_ads_3step_turn_per_day', 
            'ads_3step_ads_os_type_acost_mean', 'ads_3step_ads_os_type_earn_mean', 'ads_3step_ads_os_type_cvr', 'ads_3step_ads_os_type_turn_per_day',
            'ads_3step_mda_idx_acost_mean', 'ads_3step_mda_idx_earn_mean', 'ads_3step_mda_idx_cvr', 'ads_3step_mda_idx_turn_per_day', 
            'ads_os_type_mda_idx_acost_mean', 'ads_os_type_mda_idx_earn_mean', 'ads_os_type_mda_idx_cvr', 'ads_os_type_mda_idx_turn_per_day', 
            'domain_ads_3step_ads_os_type_acost_mean', 'domain_ads_3step_ads_os_type_earn_mean', 'domain_ads_3step_ads_os_type_cvr', 'domain_ads_3step_ads_os_type_turn_per_day',
            'domain_ads_3step_mda_idx_acost_mean', 'domain_ads_3step_mda_idx_earn_mean', 'domain_ads_3step_mda_idx_cvr', 'domain_ads_3step_mda_idx_turn_per_day',
            'ads_3step_ads_os_type_mda_idx_acost_mean', 'ads_3step_ads_os_type_mda_idx_earn_mean', 'ads_3step_ads_os_type_mda_idx_cvr', 'ads_3step_ads_os_type_mda_idx_turn_per_day']:
    agg_dict[col] = 'last'

# 스트림릿 전용 컬럼 -> last
for col in ['domain_ads_3step_mda_idx_clk_mean', 'domain_ads_3step_mda_idx_turn_mean']:
    agg_dict[col] = 'last'

# 각 변수들에 대한 집계함수
week_data = merge_data.agg(agg_dict).reset_index()



# --- 일주일 중 광고 진행 일수를 컬럼으로 넣기 ---
active_days = merge_data.size().reset_index(name='active_days')
week_data = week_data.merge(active_days, on=['ads_idx', 'week', 'mda_idx'])



# --- 주차별 그룹화 후 전환율(y 변수) 컬럼 만들기 ---
# 만약 클릭수가 0이면 전환율도 0
week_data['week_cvr'] = ( week_data['rpt_time_turn'] / week_data['rpt_time_clk'] ).replace([np.inf, -np.inf], np.nan).fillna(0)
week_data = week_data.drop(['rpt_time_turn', 'rpt_time_clk'], axis=1)





# -------------------------------------------------------------------------
# 교차피처 추가
# -------------------------------------------------------------------------
feature_df = week_data.copy()

# --- # 교차피처 추가 (2개 조합) ---
# 1) domain × ads_3step
feature_df["domain_ads3step"] = (
    feature_df["domain"].astype(str) + "_" + feature_df["ads_3step"].astype(str)
)

# 2) domain × mda_idx
feature_df["domain_mda"] = (
    feature_df["domain"].astype(str) + "_" + feature_df["mda_idx"].astype(str)
)

# 3) ads_3step × mda_idx
feature_df["ads3step_mda"] = (
    feature_df["ads_3step"].astype(str) + "_" + feature_df["mda_idx"].astype(str)
)

# 4) domain × ads_os_type
feature_df["domain_os"] = (
    feature_df["domain"].astype(str) + "_" + feature_df["ads_os_type"].astype(str)
)

# 5) ads_3step × ads_os_type
feature_df["ads3step_os"] = (
    feature_df["ads_3step"].astype(str) + "_" + feature_df["ads_os_type"].astype(str)
)

# 6) mda_idx × ads_os_type
feature_df["mda_os"] = (
    feature_df["mda_idx"].astype(str) + "_" + feature_df["ads_os_type"].astype(str)
)





# -------------------------------------------------------------------------
# 2차 전처리
# -------------------------------------------------------------------------
final_df = feature_df.copy()

# --- 데이터 유형 변환 ---
final_df['ads_idx'] = final_df['ads_idx'].astype(int).astype(str)
final_df['mda_idx'] = final_df['mda_idx'].astype(int).astype(str)



# --- 'ads_idx', 'mda_idx', 'week' 정렬하기 ---
final_df = final_df.sort_values(by=['ads_idx', 'mda_idx', 'week']).reset_index(drop=True)





# -------------------------------------------------------------------------
# train, valid, test 분리
# -------------------------------------------------------------------------
model_df = final_df.copy()

# --- 스트림릿용 컬럼 제거 ---
model_df = model_df.drop(['domain_ads_3step_mda_idx_clk_mean', 'domain_ads_3step_mda_idx_turn_mean'], axis=1)



# --- train, valid, test 분리 (valid 전체 주차) ---
# 1. 광고별 최종 week 계산
final_grouped_df = model_df.copy()
ad_max_week = final_grouped_df.groupby('ads_idx', observed=True)['week'].max().reset_index()

# Cold 광고: week < 2 (1주차만 있는 광고)
cold_ads = ad_max_week.loc[ad_max_week['week'] < 2, 'ads_idx'].tolist()

# Warm 광고: week >= 2 (2주 이상 진행된 광고)
warm_ads = ad_max_week.loc[ad_max_week['week'] >= 2, 'ads_idx'].tolist()

# 2. Test set: Cold 광고의 week=1 데이터
test_df = final_grouped_df[
    (final_grouped_df['ads_idx'].isin(cold_ads)) & (final_grouped_df['week'] == 1)
].reset_index(drop=True)

# 3. Warm 광고 ID를 train/valid로 분리 (광고 단위 split)
warm_train_ids, warm_valid_ids = train_test_split(
    warm_ads,
    test_size=0.2,            # warm 광고의 20%를 validation으로
    random_state=42
)

# 4) Train/Valid 데이터셋 구성
train_df = final_grouped_df[final_grouped_df['ads_idx'].isin(warm_train_ids)].reset_index(drop=True)
valid_df = final_grouped_df[final_grouped_df['ads_idx'].isin(warm_valid_ids)].reset_index(drop=True)





# -------------------------------------------------------------------------
# 전환율 예측 모델링
# -------------------------------------------------------------------------

# --- 카테고리컬럼 전처리 ---

cat_cols = ['domain','ads_rejoin_type','ads_os_type','mda_idx','ads_3step', 'domain_ads3step', 'domain_mda', 'ads3step_mda', 'domain_os', 'ads3step_os', 'mda_os']

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    valid_df[col] = valid_df[col].astype('category')
    test_df[col]  = test_df[col].astype('category')



# --- 독립변수, 종속변수 분리 ---

X_train = train_df.drop(columns=['week_cvr','ads_idx'])
y_train = train_df['week_cvr']

X_valid = valid_df.drop(columns=['week_cvr','ads_idx'])
y_valid = valid_df['week_cvr']

X_test  = test_df.drop(columns=['week_cvr','ads_idx'])
y_test  = test_df['week_cvr']



# --- 평가함수 ---
def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, fit_kwargs=None):
    if fit_kwargs is None:
        fit_kwargs = {}
    model.fit(X_train, y_train, **fit_kwargs)
    preds_train = model.predict(X_train)
    preds_valid = model.predict(X_valid)
    preds_test  = model.predict(X_test)
    
    print(f"Train R² : {r2_score(y_train, preds_train):.4f}")
    print(f"Train MAE: {mean_absolute_error(y_train, preds_train):.4f}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, preds_train):.4f}")

    print(f"\nValid R² : {r2_score(y_valid, preds_valid):.4f}")
    print(f"Valid MAE: {mean_absolute_error(y_valid, preds_valid):.4f}")
    print(f"Valid RMSE: {root_mean_squared_error(y_valid, preds_valid):.4f}")

    print(f"\nTest  R² : {r2_score(y_test, preds_test):.4f}")
    print(f"Test  MAE: {mean_absolute_error(y_test, preds_test):.4f}")
    print(f"Test  RMSE: {root_mean_squared_error(y_test, preds_test):.4f}")



# --- LightGBM ---
lgbm_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
evaluate_model(lgbm_model, X_train, y_train, X_valid, y_valid, X_test, y_test,
               fit_kwargs={'categorical_feature': cat_cols})



# --- Feature Importance 추출 ---
importance = lgbm_model.feature_importances_
feat_names = X_train.columns

# DataFrame 정리
importance_df = pd.DataFrame({
    'feature': feat_names,
    'importance': importance
}).sort_values(by='importance', ascending=False)
# importance_df



# --- 성능 평가 결과 ---
X_train_full = pd.concat([X_train, X_valid], axis=0)
y_train_full = pd.concat([y_train, y_valid], axis=0)

# 카테고리형 컬럼을 category dtype으로 변환
for col in cat_cols:
    X_train_full[col] = X_train_full[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 최종 모델 학습
lgbm_model.fit(
    X_train_full, y_train_full,
    categorical_feature=cat_cols
)

# 예측
train_valid_preds = lgbm_model.predict(X_train_full)
test_preds = lgbm_model.predict(X_test)

# 성능 평가
final_results = {
    "Train+Valid R2": r2_score(y_train_full, train_valid_preds),
    "Train+Valid MAE": mean_absolute_error(y_train_full, train_valid_preds),
    "Train+Valid RMSE": root_mean_squared_error(y_train_full, train_valid_preds),
    "Test R2": r2_score(y_test, test_preds),
    "Test MAE": mean_absolute_error(y_test, test_preds),
    "Test RMSE": root_mean_squared_error(y_test, test_preds)
}

print("✅ lgbm_model (Train+Valid 재학습 후 성능)")
for k, v in final_results.items():
    print(f"{k}: {v:.4f}")



# --- 실제값 vs 예측값 & 잔차 그래프 ---
# 예측
y_pred = lgbm_model.predict(X_test)

# 결과 DataFrame
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": y_test - y_pred
})

# ① Actual vs Predicted 
plt.figure(figsize=(8,6))
sns.scatterplot(x="Actual", y="Predicted", data=results, alpha=0.4, color="royalblue")
min_val = min(results["Actual"].min(), results["Predicted"].min())
max_val = max(results["Actual"].max(), results["Predicted"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
plt.title("LightGBM: Actual vs Predicted", fontsize=16)
plt.xlabel("Actual CVR")
plt.ylabel("Predicted CVR")
plt.legend()
plt.tight_layout()
plt.show()

# ② Residual Plot (잔차 플롯)
plt.figure(figsize=(8,6))
sns.scatterplot(x="Predicted", y="Residual", data=results, alpha=0.4, color="darkorange")
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.title("LightGBM: Residual Plot", fontsize=16)
plt.xlabel("Predicted CVR")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.show()





# -------------------------------------------------------------------------
# 매체 순위 예측 모델링
# -------------------------------------------------------------------------

# --- group_col 지정 ---
group_col = 'domain_ads3step_os'

for df in [train_df, valid_df, test_df]:
    df["domain_ads3step_os"] = (
        df["domain"].astype(str) + "_" +
        df["ads_3step"].astype(str) + "_" +
        df["ads_os_type"].astype(str)
    )



# --- 모델링 전 준비 ---
# 정렬 
train_sorted = train_df.sort_values(group_col).reset_index(drop=True)
valid_sorted = valid_df.sort_values(group_col).reset_index(drop=True)
test_sorted  = test_df.sort_values(group_col).reset_index(drop=True)

# X, y 분리
X_train_sorted = train_sorted.drop(columns=['week_cvr','ads_idx', group_col])
y_train_sorted = train_sorted['week_cvr']

X_valid_sorted = valid_sorted.drop(columns=['week_cvr','ads_idx', group_col])
y_valid_sorted = valid_sorted['week_cvr']

X_test_sorted  = test_sorted.drop(columns=['week_cvr','ads_idx', group_col])
y_test_sorted  = test_sorted['week_cvr']

# 그룹별 size 계산 (group_col은 원본 정렬된 df에서 가져오기)
train_group_sizes = train_sorted.groupby(group_col).size().to_numpy()
valid_group_sizes = valid_sorted.groupby(group_col).size().to_numpy()
test_group_sizes  = test_sorted.groupby(group_col).size().to_numpy()



# --- 랭킹 성능 지표 함수 ---

def evaluate_ranking_metrics(df, group_col, k_list=[1,3,5,10]):
    results = {}
    
    for group_id, group in df.groupby(group_col):
        y_true = group["true_cvr"].to_numpy()
        y_pred = group["pred_score"].to_numpy()

        # 정답 순서와 예측 순서
        true_order = np.argsort(-y_true)
        pred_order = np.argsort(-y_pred)

        # 각 그룹별 지표 저장
        for k in k_list:
            k_eff = min(k, len(y_true))  # 그룹 크기보다 k가 크면 조정
            if k_eff == 0:
                continue

            # Hit@K
            top_true = set(true_order[:k_eff])
            top_pred = set(pred_order[:k_eff])
            hit = len(top_true & top_pred) / k_eff
            results.setdefault(f"hit@{k}", []).append(hit)

            # NDCG@K
            y_true_sorted = y_true[pred_order[:k_eff]]
            gains = (2**y_true_sorted - 1) / np.log2(np.arange(2, k_eff+2))
            dcg = np.sum(gains)

            ideal_sorted = np.sort(y_true)[::-1][:k_eff]
            ideal_gains = (2**ideal_sorted - 1) / np.log2(np.arange(2, k_eff+2))
            idcg = np.sum(ideal_gains)

            ndcg = dcg / idcg if idcg > 0 else 0
            results.setdefault(f"ndcg@{k}", []).append(ndcg)

    # 그룹 평균 산출
    final_results = {metric: np.mean(vals) for metric, vals in results.items()}
    return final_results




# --- 파라미터 기본 모델로 n_bins 정하기 ---

# n_bins개로 구간 나누어서 등급화
def binarize_labels(y, n_bins):
    # y를 0~1 범위로 정규화 후 n_bins 등급으로 매핑
    y_min, y_max = y.min(), y.max()
    if y_max == y_min:
        return np.zeros_like(y, dtype=int)  # 모두 같은 값일 경우 0으로
    y_scaled = (y - y_min) / (y_max - y_min)
    return np.floor(y_scaled * (n_bins - 1)).astype(int)

def run_ranker_with_bins(n_bins):
    # Train/Valid만 등급화
    y_train_rank = binarize_labels(y_train_sorted, n_bins)
    y_valid_rank = binarize_labels(y_valid_sorted, n_bins)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=42,
        n_jobs=-1
    )

    ranker.fit(
        X_train_sorted,
        y_train_rank,
        group=train_group_sizes,
        eval_set=[(X_valid_sorted, y_valid_rank)],
        eval_group=[valid_group_sizes],
        eval_at=[5, 10],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Test 예측 (테스트셋은 연속값 그대로 사용)
    preds = ranker.predict(X_test_sorted)

    test_result_df = test_sorted[[group_col]].copy()
    test_result_df["true_cvr"] = y_test_sorted.values
    test_result_df["pred_score"] = preds

    # 랭킹 지표 계산
    metrics = evaluate_ranking_metrics(test_result_df, group_col=group_col)
    return metrics

# n_bins 후보 실험
bins_list = [10, 20, 30]
results = {}

print("📊 n_bins 최종 결과 비교")
for n, metrics in results.items():
    print(f"n_bins={n} → {metrics}")



# --- optuna로 파라미터 찾기 ---

# 1) Label 이산화 (n_bins=10 고정)
def binarize_labels(y, n_bins=10):
    return np.floor(y * (n_bins - 1)).astype(int)

y_train_rank = binarize_labels(y_train_sorted, n_bins=10)
y_valid_rank = binarize_labels(y_valid_sorted, n_bins=10)
y_test_rank  = binarize_labels(y_test_sorted,  n_bins=10)


# 2) Optuna Objective 함수
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 700, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 70, 120),
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 70),
        "subsample": trial.suggest_float("subsample", 0.8, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.95),
        "objective": "lambdarank",
        "metric": "ndcg",
        "random_state": 42,
        "n_jobs": -1
    }

    model = lgb.LGBMRanker(**params)

    model.fit(
        X_train_sorted, 
        y_train_rank,
        group=train_group_sizes,
        eval_set=[(X_valid_sorted, y_valid_rank)],
        eval_group=[valid_group_sizes],
        eval_at=[10],   # NDCG@10 기준으로 평가
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    preds = model.predict(X_valid_sorted)

    valid_result_df = valid_sorted[[group_col]].copy()
    valid_result_df["true_cvr"] = y_valid_sorted.values
    valid_result_df["pred_score"] = preds

    metrics = evaluate_ranking_metrics(valid_result_df, group_col, k_list=[10])
    return metrics["ndcg@10"] * -1  # Optuna는 최소화 → 음수로 변환

# 3) Optuna 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

print("Best Trial:")
print(study.best_trial.params)





# --- optuna에서 찾은 최적의 파라미터로 최종 Test 평가 ---

# 최종 모델 
best_params = {
    'n_estimators': 791,
    'learning_rate': 0.054364478530019016,
    'num_leaves': 98,
    'max_depth': 6,
    'min_child_samples': 59,
    'subsample': 0.8714967440697998,
    'colsample_bytree': 0.8312152195436527,
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'random_state': 42,
    'n_jobs': -1
}

# Label 변환 (n_bins=10 기준)
def binarize_labels(y, n_bins=10):
    return np.floor(y * (n_bins - 1)).astype(int)

y_train_rank = binarize_labels(y_train_sorted, n_bins=10)
y_valid_rank = binarize_labels(y_valid_sorted, n_bins=10)

# 모델 학습
ranker_best = lgb.LGBMRanker(**best_params)

ranker_best.fit(
    X_train_sorted,
    y_train_rank,
    group=train_group_sizes,
    eval_set=[(X_valid_sorted, y_valid_rank)],
    eval_group=[valid_group_sizes],
    eval_at=[10],  # NDCG@10 모니터링
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# Test 데이터 예측 및 평가
test_preds = ranker_best.predict(X_test_sorted)

test_result_df = test_sorted[[group_col]].copy()
test_result_df["true_cvr"] = y_test_sorted.values  
test_result_df["pred_score"] = test_preds         

# Hit@K, NDCG@K 계산
final_metrics = evaluate_ranking_metrics(test_result_df, group_col, k_list=[1,3,5,10])
print("\n📊 Hit@K, NDCG@K 최종 Test 성능")
print(final_metrics)





# -------------------------------------------------------------------------
# 모델 평가 지표_전환율 성능 평가
# -------------------------------------------------------------------------

# --- 기존 전환율 대비, 모델 전환율 비교 --- 
# 시각화에 필요한 데이터만 필터링
cvr_graph = model_df.copy()
cvr_graph = cvr_graph.drop(['ads_idx'], axis=1)

# 1주차 광고만 필터링
week1_df = cvr_graph[cvr_graph["week"] == 1].copy()

cat_cols = ['domain','ads_rejoin_type','ads_os_type','mda_idx','ads_3step', 'domain_ads3step', 'domain_mda', 'ads3step_mda', 'domain_os', 'ads3step_os', 'mda_os']
for col in cat_cols:
    if col in week1_df.columns:
        week1_df[col] = week1_df[col].astype("category")

# 모델 예측 전환율
feature_names = lgbm_model.feature_name_  
X_features = week1_df[feature_names]
week1_df['cvr_pred'] = lgbm_model.predict(X_features)

# 전체 평균
mean_actual = week1_df['week_cvr'].mean()
mean_model = week1_df['cvr_pred'].mean()

# 개선율 계산
improvement = (mean_model - mean_actual) / mean_actual * 100

# 값 (% 변환)
mean_actual_pct = mean_actual * 100
mean_model_pct = mean_model * 100
improvement = (mean_model - mean_actual) / mean_actual * 100

# 그래프
fig, ax = plt.subplots(figsize=(6,5))
bars = ax.bar(["기존", "모델"], 
              [mean_actual_pct, mean_model_pct], 
              color=["gray", "#E9353E"])

# 막대 위에 값 표시
for bar, value in zip(bars, [mean_actual_pct, mean_model_pct]):
    ax.text(bar.get_x() + bar.get_width()/2, value, 
            f"{value:.1f}%", ha="center", va="bottom", 
            fontsize=11, fontweight="bold")

# 제목에 개선율 표시
ax.set_title(f"1주차 신규 광고 전환율 비교",
             fontsize=13, fontweight="bold")
ax.set_ylabel("전환율 (%)")

# y축 범위 (0~100%)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))

plt.show()



# --- 개선 vs 악화 비율 ---

improved_ratio = (week1_df["improvement"] > 0).mean()
worsened_ratio = (week1_df["improvement"] < 0).mean()

labels = ["개선", "악화"]
sizes = [improved_ratio, worsened_ratio]
colors = ["#E9353E", "gray"]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct="%.1f%%", startangle=90,
    colors=colors, wedgeprops={"edgecolor": "white"}
)

# 퍼센트 숫자 스타일 (두껍게 + 윤곽선)
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("black")      # 가장 두껍게
    autotext.set_fontsize(12)             # 크기는 그대로
    # 윤곽선 효과 추가 (검정 테두리)
    autotext.set_path_effects([
        path_effects.Stroke(linewidth=0.8, foreground="white"),
        path_effects.Normal()
    ])

plt.title("1주차 광고 개선/악화 비율", fontsize=13, fontweight="bold")
plt.show()





# -------------------------------------------------------------------------
# 모델 평가 지표_랭킹 성능 지표
# -------------------------------------------------------------------------

# --- 랭킹 지표 함수 --- 
def evaluate_ranking(
    df,
    group_col='domain_ads3step_os',
    y_true_col='true_cvr',
    pred_col='pred_score',
    ks=(1,3,5,10),
    compute_spearman=True
):
    # 안전장치: 필수 컬럼 확인
    for c in [group_col, y_true_col, pred_col]:
        assert c in df.columns, f"Missing column: {c}"
    
    rows = []
    weights = []
    
    for gid, g in df.groupby(group_col, observed=False):
        if len(g) < 2:
            # 랭킹이 성립 안 되는 너무 작은 그룹은 스킵(또는 계속 포함 원하면 처리 변경)
            continue
        
        # numpy 형태
        y_true = g[y_true_col].to_numpy()
        y_pred = g[pred_col].to_numpy()
        
        # 정렬은 필요없고 ndcg_score에는 row shape 필요
        y_true_row = y_true.reshape(1, -1)
        y_pred_row = y_pred.reshape(1, -1)
        
        # Spearman (선택)
        sp = np.nan
        if compute_spearman:
            # 순위로 변환해 스피어만
            tr = pd.Series(y_true).rank(ascending=False, method='first')
            pr = pd.Series(y_pred).rank(ascending=False, method='first')
            sp, _ = spearmanr(tr, pr)
        
        # Hit@K: “실제 1등”이 예측 Top-K 안에 있는지
        true_top_idx = g[y_true_col].idxmax()
        # 예측 점수 내림차순 K개
        g_sorted_pred = g.sort_values(pred_col, ascending=False)
        
        row = {'group_id': gid, 'spearman': sp}
        for k in ks:
            topk_idx = set(g_sorted_pred.head(k).index)
            row[f'hit@{k}'] = 1 if true_top_idx in topk_idx else 0
            row[f'ndcg@{k}'] = ndcg_score(y_true_row, y_pred_row, k=k)
        
        rows.append(row)
        weights.append(len(g))
    
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid groups for evaluation.")
    
    # Macro (그룹 균등 평균)
    macro = out.drop(columns=['group_id']).mean(numeric_only=True).to_dict()
    
    # Weighted (그룹 크기 가중 평균)
    w = np.array(weights)
    w = w / w.sum()
    weighted = {}
    for c in out.columns:
        if c in ('group_id',):
            continue
        weighted[c] = np.average(out[c].to_numpy(), weights=w)
    
    return {'macro': macro, 'weighted': weighted, 'by_group': out}



# --- 광고 조합별 모델 성능 --- 
test_eval_df = test_df[[ 'domain_ads3step_os' ]].copy()
test_eval_df['true_cvr'] = y_test.values
test_eval_df['pred_score'] = ranker_best.predict(X_test)  # 랭커 raw 점수

res = evaluate_ranking(
    test_eval_df,
    group_col='domain_ads3step_os',
    y_true_col='true_cvr',
    pred_col='pred_score',
    ks=(1,3,5,10)
)

print("== Macro (headline) ==")
macro_metrics = {k: round(v,4) for k,v in res['macro'].items()}
print(macro_metrics)



# --- 광고 조합별 모델 성능 시각화 --- 
# 성능 결과
overall = {'hit@10': 0.4538, 'ndcg@10': 0.6020}
macro   = {'hit@10': 0.6154, 'ndcg@10': 0.6148}

metrics = list(overall.keys())
x = range(len(metrics))

fig, ax = plt.subplots(figsize=(6,4))

# 막대 그리기
bars1 = ax.bar([i - 0.2 for i in x], [overall[m] for m in metrics], 
               width=0.4, label="모델 전체", color="gray")
bars2 = ax.bar([i + 0.2 for i in x], [macro[m] for m in metrics], 
               width=0.4, label="광고조합별 평균", color="#E9353E")

# 막대 위에 숫자 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f"{height:.3f}", ha="center", va="bottom", fontsize=9)

# 꾸미기
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0,1.1)
ax.set_ylabel("Score")
ax.set_title("전체 vs 광고조합별 성능 비교")
ax.legend()

plt.show()