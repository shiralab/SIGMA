import openai
import requests
import json
import time
from typing import Optional
import re
from openai import OpenAI, RateLimitError, APIError

def call_llm(prompt, model_name='gpt-5-nano', port=8000, api_key=None,
             api_base=None, max_retries=3, temperature=0.7, top_p=0.95):
    """
    Call LLM to generate feature engineering code.
    
    Supports both:
    1. OpenAI GPT models (gpt-4-turbo, gpt-3.5-turbo)
    2. Local LLM models (via OpenAI-compatible API)
    
    Args:
        prompt (str): The prompt to send to the LLM
        model_name (str): Model name (e.g., 'gpt-4-turbo' or 'local-model')
        port (int): Port for local LLM server (default: 8000)
        api_key (str): OpenAI API key (if None, read from OPENAI_API_KEY env var)
        api_base (str): API base URL for local LLM (e.g., 'http://localhost:8000/v1')
        max_retries (int): Maximum number of retry attempts (default: 3)
    
    Returns:
        str: Generated code or response from LLM
    """
    
    # Determine if using OpenAI or local LLM
    is_openai = model_name.startswith('gpt')
    
    if is_openai:
        return _call_openai_llm(prompt, model_name, api_key, max_retries, temperature, top_p)
    else:
        return _call_local_llm(prompt, model_name, port, api_base, max_retries, temperature, top_p)


def _call_openai_llm(prompt, model_name, api_key=None, max_retries=3, temperature=0.7, top_p=0.95):
    """
    Call OpenAI GPT models.
    
    Args:
        prompt (str): The prompt to send
        model_name (str): OpenAI model name (e.g., 'gpt-4-turbo', 'gpt-3.5-turbo')
        api_key (str): OpenAI API key
        max_retries (int): Maximum retry attempts
    
    Returns:
        str: Generated response
    """
    import os
    
    # Set API key
    if api_key:
        openai_api_key = api_key
    else:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=openai_api_key)

    # Retry mechanism with exponential backoff
    for attempt in range(max_retries):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data science expert specializing in feature engineering. Generate clean, working Python code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }

            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if top_p is not None:
                request_kwargs["top_p"] = top_p

            response = client.chat.completions.create(**request_kwargs)
            
            # Extract response text
            generated_text = response.choices[0].message.content
            print(f"[SUCCESS] OpenAI API call succeeded on attempt {attempt + 1}")
            return generated_text
            
        except RateLimitError as e:
            print(f"[RATE_LIMIT] Attempt {attempt + 1}/{max_retries}: Rate limited. Waiting...")
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
            
        except APIError as e:
            print(f"[API_ERROR] Attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Unexpected error - {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError(f"Failed to call OpenAI API after {max_retries} attempts")


def _call_local_llm(prompt, model_name, port=8001, api_base=None, max_retries=3,
                    temperature=0.7, top_p=0.95):
    """
    Call local LLM server (OpenAI-compatible API).
    
    Args:
        prompt (str): The prompt to send
        model_name (str): Model name registered on local server
        port (int): Port of local LLM server
        api_base (str): Full base URL (if None, constructed from port)
        max_retries (int): Maximum retry attempts
    
    Returns:
        str: Generated response
    """
    
    # Construct API base URL if not provided
    if api_base is None:
        api_base = f"http://localhost:{port}/v1"
    
    # Retry mechanism with exponential backoff
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data science expert specializing in feature engineering. Generate clean, working Python code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4096,
            }

            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p

            response = requests.post(
                f"{api_base}/chat/completions",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                print(f"[SUCCESS] Local LLM call succeeded on attempt {attempt + 1}")
                return generated_text
            else:
                print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Status code {response.status_code}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Local LLM returned status {response.status_code}: {response.text}")
                    
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] Attempt {attempt + 1}/{max_retries}: Request timeout")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
                
        except requests.exceptions.ConnectionError as e:
            print(f"[CONNECTION_ERROR] Attempt {attempt + 1}/{max_retries}: Cannot connect to {api_base}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries}: Unexpected error - {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError(f"Failed to call local LLM after {max_retries} attempts")

def extract_code_block(response_text: str) -> str:
    """
    智能提取 LLM 回复中的 Python 代码。
    支持：1. 标准 Markdown 块；2. 带有前置文字的裸代码。
    """
    # --- 策略 1: 提取标准 Markdown 代码块 ---
    markdown_pattern = r'```(?:python|py)?\n(.*?)```'
    matches = re.findall(markdown_pattern, response_text, re.DOTALL)
    
    if matches:
        return "\n\n".join([m.strip() for m in matches if m.strip()])
    
    # --- 策略 2: 处理没有 Markdown 标签的裸代码 ---
    # 定义 Python 代码的起始特征词
    anchors = [
        r'^import\s', 
        r'^from\s', 
        r'^def\s', 
        r'^class\s',
        r'\nimport\s', 
        r'\nfrom\s', 
        r'\ndef\s',
        r'\nclass\s'
    ]
    
    # 寻找第一个出现的特征词位置
    first_idx = float('inf')
    found = False
    
    for anchor in anchors:
        match = re.search(anchor, response_text, re.MULTILINE)
        if match:
            # 如果是 \n 开头的锚点，索引需要 +1 避开换行符本身
            start_pos = match.start() if not anchor.startswith(r'\n') else match.start() + 1
            if start_pos < first_idx:
                first_idx = start_pos
                found = True
    
    if found:
        # 从第一个特征词位置截取到最后
        raw_code = response_text[first_idx:].strip()
        
        # 清理：如果结尾有明显的非代码解释性文字（如 "Hope this helps"），
        # 通常代码后面会有连续两个换行符，这里可以根据需求做进一步截断，
        # 但通常直接取到底部对执行影响不大，因为 Python 解释器会忽略末尾的非语法内容
        return raw_code

    return ""


def validate_generated_code(code: str) -> bool:
    """
    Validate if generated code is valid Python.
    
    Args:
        code (str): Python code to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"[SYNTAX_ERROR] Generated code has syntax error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error validating code: {e}")
        return False
