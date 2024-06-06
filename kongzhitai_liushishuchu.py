from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import dashscope

dashscope.api_key = 'sk-cb788e9cc36349e7bde1f56b84edcbe0'

messages = []

while True:
    message = str(input('user:'))
    messages.append({'role': Role.USER, 'content': message})
    whole_message = ''
    responses = Generation.call(Generation.Models.qwen_max, messages=messages, result_format='message', stream=True, incremental_output=True)
    print('system:',end='')
    for response in responses:
        whole_message += response.output.choices[0]['message']['content']
        print(response.output.choices[0]['message']['content'], end='')
    print()
    messages.append({'role': 'assistant', 'content': whole_message})
