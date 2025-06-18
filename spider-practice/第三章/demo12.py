import requests

url = 'https://www.baidu.com/s?wd=%E7%BE%8E%E5%A5%B3&base_query=%E7%BE%8E%E5%A5%B3&pn=80&oq=%E7%BE%8E%E5%A5%B3&ie=utf-8&usm=2&fenlei=256&rsv_idx=1&rsv_pq=cac8353501c8e2ef&rsv_t=a250oB%2FvLb6FVQA3eneAzus%2FzBg%2BrJWUQhdhvBoNol3460x9rrO2NkNb7f0'
resp = requests.get(url)
json_data = resp.json()
print(json_data)
