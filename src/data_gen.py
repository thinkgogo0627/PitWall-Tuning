## 지도 학습을 위한 데이터셋 생성

## 관련 데이터셋을 구하기는 힘드니 Gemini 사용하여 합성 데이터 생성

import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# 저장할 파일 경로, 제작 샘플 갯수
OUTPUT_FILE = "data/train_dataset.jsonl"
NUM_SAMPLES = 20

# 입출력 쌍
## 입력: 사용자의 질문 -> Raw data
## 출력: 데이터를 분석해서 사용자에게 건네는 최종 답변

# Data Generator에게 시킬 일
## 1. 가상의 질문 제작
## 2. 그 질문에 대한 SQL실행 결과라고 볼 수 있는 가상의 데이터(JSON) 제작
## 3. 그 데이터를 보고 사용자가 만족할 만한 답변 작성


# 프롬프트 엔지니어링

# (A) 전략 에이전트 (Strategy Agent)
PROMPT_STRATEGY = """
당신은 F1 팀의 수석 전략 엔지니어(Chief Strategy Officer)입니다.
사용자의 질문에 대해 오직 **데이터와 수치(랩타임, 타이어 수명, 피트스탑 타이밍)**에 근거해서만 답변하세요.

[행동 강령]
1. 감정 배제: "아쉽게도" 같은 말 금지.
2. 결과 중심: 순위, 갭(Gap), 피트스탑 랩 수 등 팩트 제시.
3. 드라이버 이름은 반드시 번호로 변환해서 생각할 것 (예: 베르스타펜 -> 1, 르클레르 -> 16).
"""


# (B) 서킷 에이전트 (Circuit Agent)
PROMPT_CIRCUIT = """
당신은 F1 팀의 '레이스 엔지니어'이자 '트랙 분석가'입니다.
사용자에게 이번 그랑프리 서킷의 **기술적, 전략적 특징**을 브리핑해야 합니다.

[행동 강령]
1. 전문성 과시: '더티에어', '그레인/블리스터링', '트랙션', '다운포스' 등 전문 용어 사용.
2. 데이터 기반: "분석 결과, 소프트 타이어가 랩당 0.1초씩 느려지는 High Deg 성향입니다" 처럼 구체적으로.
3. 서킷의 섹터별 특징(고속/저속)을 명확히 구분하여 설명.
"""


# (C) 브리핑 에이전트 (Briefing Agent)
PROMPT_BRIEFING = """
당신은 F1 전문 저널리스트이자 팀의 '공보 담당관(Press Officer)'입니다.
경기가 끝난 후, 사용자에게 **이번 경기의 핵심 내용과 비하인드 스토리**를 종합적으로 브리핑해야 합니다.

[행동 강령]
1. **Fact & Story 결합**: 단순히 순위만 나열하지 말고, 그 결과가 나온 '이유(맥락)'를 덧붙일 것.
2. 말투: 객관적인 사실 전달과 현장감 있는 묘사를 섞어, 한 편의 '레이스 리포트' 기사처럼 작성.
3. 포디움, 리타이어 원인, 오늘의 드라이버 등을 중점적으로 다룸.
"""


# 에이전트 설정 딕셔너리
AGENTS = {
    "strategy": {
        "system_prompt": PROMPT_STRATEGY,
        "context_hint": "가상의 레이스 전략 데이터 (랩타임, 타이어 마모도, 피트스탑 델타 등)",
        "task_desc": "특정 드라이버의 전략 분석 요청"
    },
    "circuit": {
        "system_prompt": PROMPT_CIRCUIT,
        "context_hint": "가상의 서킷 분석 데이터 (섹터별 속도, 타이어 데그라데이션 수치, 날씨 등)",
        "task_desc": "서킷의 기술적 특징 및 타이어 관리법 질문"
    },
    "briefing": {
        "system_prompt": PROMPT_BRIEFING,
        "context_hint": "가상의 경기 결과(순위) 및 뉴스/인터뷰 내용 (사고 원인, 페널티 등)",
        "task_desc": "경기 결과 요약 및 비하인드 스토리 질문"
    }
}


# ==========================================
# 3. 데이터 생성용 메타 프롬프트 (Teacher Model 지시)
# ==========================================
def get_generation_prompt(agent_type, config):
    return f"""
    당신은 LLM 학습을 위한 합성 데이터 생성기(Synthetic Data Generator)입니다.
    
    목표: '{agent_type}' 역할을 수행하는 AI를 학습시키기 위한 [질문-컨텍스트-답변] 쌍을 생성하세요.
    
    1. **Target Persona (Instruction):** {config['system_prompt']}
    
    2. **Task Type:** {config['task_desc']}
    
    3. **생성 요구사항 (JSON 포맷):**
       - **instruction**: 위 Target Persona 내용을 그대로 넣으세요.
       - **input**: 
          (1) 사용자의 질문 (User Query)
          (2) **[Context]**: 해당 질문을 해결하기 위해 도구(Tool)가 가져왔을 법한 **가상의 데이터**를 포함하세요. ({config['context_hint']})
          *형식: "사용자 질문\\n\\n[Context Retrieved]\\n(가상의 도구 조회 결과 JSON/Text)..."*
       - **output**: 위 Context를 보고 Target Persona의 말투와 행동 강령에 맞춰 작성한 **최종 답변**.
    
    4. **Output Example (JSON Only):**
    {{
        "instruction": "당신은 F1 팀의 수석 전략 엔지니어...",
        "input": "2025 모나코에서 르클레르 전략 어때?\\n\\n[Context Retrieved]\\n{{'driver': 'Leclerc', 'grid': 1, 'tire_deg': 'High', 'rain_prob': '80%'}}",
        "output": "데이터 분석 결과, 현재 강수 확률이 80%이므로 폴포지션인 르클레르(16번)는 스타트 타이어 유지 후 웨트 전환 전략이 유효합니다..."
    }}
    
    **주의:** 한국어로 작성하세요. JSON 포맷을 엄격히 지키세요.
    """

# ==========================================
# 4. 유틸리티
# ==========================================
def clean_json_string(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

# ==========================================
# 5. 메인 로직
# ==========================================

NUM_SAMPLES_PER_AGENT = 20
def generate_data():
    if not GEMINI_API_KEY:
        print(" Error: GEMINI_API_KEY가 없습니다.")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # 속도 위주

    print(f" [Multi-Persona] 데이터 생성 시작 (Agent당 {NUM_SAMPLES_PER_AGENT}개)")
    
    total_generated = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        
        # 3가지 에이전트를 순회하며 생성
        for agent_type, config in AGENTS.items():
            print(f"\nCreating data for: [{agent_type.upper()}] Agent...")
            
            for i in tqdm(range(NUM_SAMPLES_PER_AGENT)):
                try:
                    meta_prompt = get_generation_prompt(agent_type, config)
                    response = model.generate_content(
                        meta_prompt + f"\n\n(Seed: {agent_type}_{i}) 새로운 다양한 상황의 데이터를 하나 생성해줘.",
                        generation_config={"response_mime_type": "application/json", "temperature": 0.9}
                    )
                    
                    raw_text = clean_json_string(response.text)
                    data = json.loads(raw_text, strict=False)
                    
                    # 필수 키 확인
                    if not all(k in data for k in ["instruction", "input", "output"]):
                        continue
                        
                    # 파일 저장
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    total_generated += 1
                    time.sleep(0.5) # API 속도 조절
                    
                except Exception as e:
                    print(f" Error ({agent_type}): {e}")
                    time.sleep(1)
                    continue

    print(f"전체 완료! 총 {total_generated}개 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")
    print(" 이제 이 파일을 Git에 올리고 Colab에서 학습을 시작하세요.")

if __name__ == "__main__":
    generate_data()