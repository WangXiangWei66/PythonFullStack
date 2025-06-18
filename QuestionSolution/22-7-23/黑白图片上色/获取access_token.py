
import requests

host ='https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=3iOPMNJ4WI8MRXLzjBHQBrkT&client_secret=rofLAEXlFgAdNJkAyupxBrtWKTAoWicY'
response = requests.get(host)
if response:
    print(response.json())