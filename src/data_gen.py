import os
import json
import time
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # 안전 설정용
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# ==========================================
# 1. 설정
# ==========================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OUTPUT_FILE = "data/merged_train.jsonl"
NUM_SAMPLES_PER_AGENT = 20 # 에이전트당 생성할 데이터 갯수
MAX_WORKERS = 5 # 멀티스레딩 동시 실행 갯수

# 파일 쓰기 충돌방지 락
file_lock = threading.Lock()

# ==========================================
# 2. Safety Settings
# ==========================================
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ==========================================
# 3. 데이터 스키마 정의 (강제 Dict 변환용)
# ★ 모델에게 "이 구조 아니면 뱉지 마"라고 지정하는 설계도
# ==========================================
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "instruction": {"type": "STRING"},
        "input": {"type": "STRING"},
        "output": {"type": "STRING"}
    },
    "required": ["instruction", "input", "output"]
}

# ==========================================
# 4. Prompt & Examples (동일)
# ==========================================
EXAMPLES = {
    "strategy": """
    <Example>
    [Input]
    질문: 2023년 싱가포르 그랑프리에서 카를로스 사인츠(55번)의 전략과 레이스 페이스를 분석해줘
    [Context Retrieved]
    {'driver': 'Carlos Sainz', 'number': 55, 'result': 1, 'strategy': '1-stop', 'start_tyre': 'Medium', 'pit_lap': 20, 'end_tyre': 'Hard', 'sc_lap': 20, 'deg_value': -0.0111, 'mini_sector_fastest': 'Hamilton (60%)', 'tactic': 'DRS sharing with Norris'}
    [Output]
    2023년 싱가포르 그랑프리에서 카를로스 사인츠(55번)는 완벽한 타이어 관리와 전략적 페이스 조절을 통해 우승을 차지했습니다.
    1. 레이스 전략: 원-스탑(One-Stop) 전략의 성공
    * **피트스탑 타이밍**: 20랩 SC 상황을 활용해 하드 타이어로 교체했습니다.
    2. 레이스 페이스: '관리'와 '방어'에 초점을 맞춘 영리한 주행
    * **타이어 마모 최소화**: `Tire_Degradation_Analyzer` 분석 결과, -0.0111로 극히 낮았습니다.
    </Example>
    """,
    "circuit": """
    <Example>
    [Input]
    질문: 레드불링 서킷 설명해줘.
    [Context Retrieved]
    {'circuit_name': 'Red Bull Ring', 'type': 'Rollercoaster High-Speed', 'corners': 10, 'key_corners': ['T1', 'T3'], 'downforce': 'Medium-Low'}
    [Output]
    레드불링 서킷은 오스트리아 그랑프리가 열리는 곳으로, 다음과 같은 특징을 가집니다.
    1. 서킷 개요:
    * **롤러코스터형 고속 서킷**: 고저차가 극심합니다.
    2. 주요 코너:
    * **T3 (Remus)**: 가장 강력한 제동 구간이자 추월 포인트입니다.
    </Example>
    """,
    "briefing": """
    <Example>
    [Input]
    질문: 2024년 벨기에 그랑프리 메르세데스 리뷰해줘.
    [Context Retrieved]
    {'team': 'Mercedes', 'p1_driver': 'Hamilton', 'p1_result': 'Win', 'p2_driver': 'Russell', 'p2_result': 'DNF', 'p2_reason': 'Crash'}
    [Output]
    2024년 벨기에 그랑프리에서 메르세데스 팀은 희비가 엇갈렸습니다.
    * **루이스 해밀턴: 우승 (1위)**
      * 압도적인 기량을 선보이며 우승을 차지했습니다.
    * **조지 러셀: 리타이어 (DNF)**
      * 레이스 초반 충돌로 인해 경기를 중단해야 했습니다.
    </Example>
    """
}

PROMPTS = {
    "strategy": "당신은 F1 수석 전략 엔지니어입니다. 답변은 [1. 레이스 전략, 2. 레이스 페이스, 결론] 구조를 가지며 구체적 수치를 인용하세요.",
    "circuit": "당신은 트랙 분석가입니다. [1. 개요, 2. 주요 코너, 3. 엔지니어링] 목차로 전문 용어를 사용해 설명하세요.",
    "briefing": "당신은 F1 공보 담당관입니다. 팀원별 결과와 사고 원인, 인터뷰 내용을 포함하여 기사처럼 요약하세요."
}

def get_generation_prompt(agent_type):
    return f"""
    목표: '{agent_type}' 페르소나 데이터 생성.
    Target Persona: {PROMPTS[agent_type]}
    Golden Sample: {EXAMPLES[agent_type]}
    """

def clean_json_string(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

# ==========================================
# 5. 실행 로직 (리스트 방어 로직 추가)
# ==========================================
def generate_data():
    if not GEMINI_API_KEY:
        print(" Error: API Key가 없습니다.")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    
    # 2.0 Flash 사용
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    

    print(f" [Structured Mode] 데이터 생성 시작 (Schema 강제 적용)...")
    
    total_generated = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for agent_type in PROMPTS.keys():
            print(f"\nCreating data for: [{agent_type.upper()}]...")
            
            # 진행바 다시 부활 (이제 안정적일 테니까요)
            for i in tqdm(range(NUM_SAMPLES_PER_AGENT)):
                try:
                    prompt = get_generation_prompt(agent_type)
                    
                    response = model.generate_content(
                        prompt + f"\n\n(Seed: {agent_type}_{i}) Generate ONE new sample.",
                        generation_config={
                            "response_mime_type": "application/json",
                            "response_schema": RESPONSE_SCHEMA # ★ 핵심: 스키마 강제!
                        },
                        safety_settings=SAFETY_SETTINGS
                    )
                    
                    if not response.text:
                        continue
                        
                    # JSON 파싱
                    raw_data = json.loads(clean_json_string(response.text), strict=False)
                    
                    # ★ [방어 로직] 리스트로 오면 첫 번째 놈만 꺼낸다 (User Issue Fix)
                    if isinstance(raw_data, list):
                        if len(raw_data) > 0:
                            data = raw_data[0]
                        else:
                            continue # 빈 리스트면 패스
                    else:
                        data = raw_data

                    # 키 확인
                    if not all(k in data for k in ["instruction", "input", "output"]):
                        continue
                    
                    # 저장
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    total_generated += 1
                    
                    # Rate Limit 방지
                    time.sleep(0.5)
                    
                except Exception as e:
                    # 에러 나면 그냥 조용히 넘어가고 다음 거 시도 (tqdm 안 깨지게)
                    time.sleep(1)
                    continue

    print(f"\n 최종 확인: {OUTPUT_FILE} ({total_generated}개 저장됨)")

if __name__ == "__main__":
    generate_data()