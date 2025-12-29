## 지도 학습을 위한 데이터셋 생성

## 관련 데이터셋을 구하기는 힘드니 Gemini 사용하여 합성 데이터 생성

import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
import os

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
SYSTEM_PROMPT = """
당신은 F1 레이싱 데이터를 분석하여 사용자에게 통찰력을 제공하는 'PitWall AI'의 데이터 생성 엔진입니다.

다음 3단계로 구성된 학습 데이터를 생성해야 합니다:
1. [Question]: F1에 관한 사용자의 질문 (단순 조회부터 비교/분석까지 다양하게)
2. [Context]: 위 질문을 해결하기 위해 DB에서 조회된 결과 데이터 (JSON 형식의 가상 데이터)
3. [Response]: 위 Context 데이터를 바탕으로 사용자에게 답변하는 자연스러운 한국어 문장. 

**필수 조건:**
- Context 데이터는 질문에 맞는 적절한 필드(드라이버명, 랩타임, 타이어, 순위 등)를 포함해야 함.
- 질문하는 그랑프리 연도는 2022년 ~ 2025년 사이의 그랑프리여야 함.
- 질문하는 그랑프리 장소는 어느 한 곳만 질문하지 말고, 여러 곳을 골고루 돌아가면서 질문하도록 함.
- [호주, 중국, 일본, 바레인, 헝가리, 오스트리아, 영국, 이탈리아(몬차, 에밀리아-로마냐), 네덜란드, 싱가포르, 아제르바이잔, 라스베이거스, 벨기에, 카타르, 아부다비, 상파울루, 멕시코, 마이애미, 사우디아라비아, 모나코, 스페인, 캐나다] 중에서 한 개씩 골고루 질문할것
- Response는 단순히 데이터를 나열하지 말고, "베르스타펜이 2위와 10초 차이로 압도적인 우승을 차지했습니다" 처럼 인사이트를 담을 것.
- **출력 형식은 반드시 아래 JSON 포맷을 따를 것.**

Example Output Format:
{
    "instruction": "주어진 F1 데이터(Context)를 바탕으로 사용자의 질문에 답변하세요.",
    "input": "질문: 2023 모나코 GP 우승자 기록이 어때?\n데이터: [{'driver': 'Max Verstappen', 'team': 'RedBull', 'time': '1:48:51.980', 'gap': '+0.000', 'tyre': 'Hard'}]",
    "output": "2023년 모나코 그랑프리의 우승자는 레드불의 막스 베르스타펜입니다. 그는 하드 타이어를 사용하여 1시간 48분 51초의 기록으로 결승선을 통과했습니다."
}
"""

## JSON 포맷 맞추기
def clean_json_string(text):
    '''
    JSON 파싱 방해금지
    '''
    text = text.replace("```json", "").replace("```", "")

    # 공백 제거
    text = text.strip()

    return text



# 데이터 생성
def generate_data():

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    print(f"데이터 생성 시작: {NUM_SAMPLES}개 목표")

    generated_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        
        while generated_count < NUM_SAMPLES:
            try:
                response = model.generate_content(
                    SYSTEM_PROMPT + f"\n\n새로운 예시 데이터 1개를 JSON으로 생성해줘. (현재 {generated_count + 1}번째)",
                    generation_config={"response_mime_type": "application/json", "temperature": 0.85}
                )
                
                raw_text = response.text
                cleaned_text = clean_json_string(raw_text)
                
                # strict=False 옵션을 주어 제어 문자(줄바꿈 등)에 조금 더 관대하게 파싱
                data = json.loads(cleaned_text, strict=False)
                
                # 데이터 유효성 검사
                if "input" not in data or "output" not in data:
                    print(f"키 누락 발생 (Skip)")
                    continue

                # 저장 (한 줄에 하나의 JSON 객체)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                generated_count += 1
                pbar.update(1)
                
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시, 그냥 넘어가고 다시 시도 (멈추지 않음)
                # print( JSON 파싱 실패 (Retrying...): {e}") 
                pass 
            except Exception as e:
                print(f"알 수 없는 에러 (Skip): {e}")
                time.sleep(1)
                continue

    print(f"\n 데이터셋 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_data()