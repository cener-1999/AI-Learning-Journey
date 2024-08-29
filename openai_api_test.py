from openai import OpenAI
from settings import OPENAI_API_KEY, SERPAPI_API_KEY, LANGSMITH_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)

# import requests
# import socket
#
# # 获取本机的局域网IP地址
# local_ip = socket.gethostbyname(socket.gethostname())
#
# # 获取本机的公共IP地址
# public_ip = requests.get('https://api.ipify.org').text
#
# # 使用IP地理位置服务来查询IP的地理位置
# response = requests.get(f'https://ipinfo.io/{public_ip}/json')
# location_data = response.json()
#
# # 输出IP地址和所在地信息
# print(f"本机的局域网IP地址: {local_ip}")
# print(f"本机的公共IP地址: {public_ip}")
# print(f"所在地: {location_data.get('city', '未知城市')}, {location_data.get('region', '未知地区')}, {location_data.get('country', '未知国家')}")
