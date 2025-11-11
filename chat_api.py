import openai
import argparse

# OpenAI API 버전 호환성 처리
try:
    # openai >= 1.0.0
    from openai import OpenAI
    _openai_v1 = True
except ImportError:
    # openai < 1.0.0
    _openai_v1 = False

def llm_chat(User_message, stop='12'):
    if len(stop) < 3:
        stop = None
    else:
        stop = [stop]
    
    our_messages = [
        {'role': 'user', 'content': User_message}
    ]
    
    if _openai_v1:
        # OpenAI API 1.0.0+ 사용
        client = OpenAI(
            api_key="",  # API 키 설정 필요
            base_url="https://api.openai.com/v1"
        )
        if stop:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=our_messages,
                stop=stop
            )
        else:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=our_messages
            )
        llm_response = response.choices[0].message.content
    else:
        # OpenAI API < 1.0.0 사용
        openai.api_key = ""
        openai.api_base = "https://api.openai.com/v1"
        if stop:
            response = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=our_messages,
                stop=stop
            )
        else:
            response = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=our_messages
            )
        llm_response = response['choices'][0]['message'].to_dict()['content']
    
    return llm_response


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--role", "-r", type=str, default="Movie recommender", help="llm role")
#     parser.add_argument("--message", "-m", type=str, default="Please recommend some movie for user", help="Messages send to chatgpt.")
#     parser.add_argument("--stop", "-s", type=str, default="1", help="Stop Words")

#     args, _ = parser.parse_known_args()
#     print(llm_chat(args.role + args.message, stop=args.stop))
#     # print(llm_chat_renmin(args.message))