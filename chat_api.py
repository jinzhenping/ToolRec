import openai
import argparse
import time
import os

# OpenAI 예외 클래스 import (버전 호환성)
try:
    from openai import APITimeoutError, APIConnectionError, APIError
except ImportError:
    # 구버전 호환성을 위한 예외 클래스 정의
    class APITimeoutError(Exception):
        pass
    class APIConnectionError(Exception):
        pass
    class APIError(Exception):
        pass

# OpenAI API 버전 호환성 처리
try:
    # openai >= 1.0.0
    from openai import OpenAI
    _openai_v1 = True
except ImportError:
    # openai < 1.0.0
    _openai_v1 = False

def llm_chat(User_message, stop='12', max_retries=3, timeout=60):
    """
    LLM API 호출 함수 (타임아웃 및 재시도 로직 포함)
    
    Args:
        User_message: 사용자 메시지
        stop: 중지 토큰
        max_retries: 최대 재시도 횟수
        timeout: API 호출 타임아웃 (초)
    """
    if len(stop) < 3:
        stop = None
    else:
        stop = [stop]
    
    our_messages = [
        {'role': 'user', 'content': User_message}
    ]
    
    for attempt in range(max_retries):
        try:
            if _openai_v1:
                # OpenAI API 1.0.0+ 사용
                # API 키는 환경 변수 OPENAI_API_KEY에서 읽거나 직접 입력
                api_key = os.getenv("OPENAI_API_KEY", "")  # 환경 변수에서 읽기, 없으면 빈 문자열
                if not api_key:
                    # 환경 변수에 없으면 여기에 직접 입력 (보안상 권장하지 않음)
                    api_key = ""  # 여기에 API 키를 직접 입력할 수 있지만, 환경 변수 사용을 권장합니다
                
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.openai.com/v1",
                    timeout=timeout  # 타임아웃 설정
                )
                if stop:
                    response = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=our_messages,
                        stop=stop,
                        timeout=timeout
                    )
                else:
                    response = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=our_messages,
                        timeout=timeout
                    )
                llm_response = response.choices[0].message.content
            else:
                # OpenAI API < 1.0.0 사용
                # API 키는 환경 변수 OPENAI_API_KEY에서 읽거나 직접 입력
                api_key = os.getenv("OPENAI_API_KEY", "")  # 환경 변수에서 읽기, 없으면 빈 문자열
                if not api_key:
                    # 환경 변수에 없으면 여기에 직접 입력 (보안상 권장하지 않음)
                    api_key = ""  # 여기에 API 키를 직접 입력할 수 있지만, 환경 변수 사용을 권장합니다
                
                openai.api_key = api_key
                openai.api_base = "https://api.openai.com/v1"
                if stop:
                    response = openai.ChatCompletion.create(
                        model='gpt-4o-mini',
                        messages=our_messages,
                        stop=stop,
                        timeout=timeout
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model='gpt-4o-mini',
                        messages=our_messages,
                        timeout=timeout
                    )
                llm_response = response['choices'][0]['message'].to_dict()['content']
            
            return llm_response
            
        except (APITimeoutError, APIConnectionError, APIError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 지수 백오프: 5초, 10초, 15초
                print(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                print(f"{wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print(f"API 호출 최종 실패: {str(e)}")
                raise
        except Exception as e:
            print(f"예상치 못한 오류 발생: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"{wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise
    
    return None


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--role", "-r", type=str, default="Movie recommender", help="llm role")
#     parser.add_argument("--message", "-m", type=str, default="Please recommend some movie for user", help="Messages send to chatgpt.")
#     parser.add_argument("--stop", "-s", type=str, default="1", help="Stop Words")

#     args, _ = parser.parse_known_args()
#     print(llm_chat(args.role + args.message, stop=args.stop))
#     # print(llm_chat_renmin(args.message))