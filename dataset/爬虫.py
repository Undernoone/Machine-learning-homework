from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import requests
import time
import os

# 设置 WebDriver 路径
driver_path = r'D:\Software\msedgedriver.exe'

# 创建 EdgeOptions 对象
edge_options = Options()

# 创建 Service 对象
service = Service(executable_path=driver_path)

# 初始化 WebDriver
driver = webdriver.Edge(service=service, options=edge_options)

# 访问网站
driver.get('https://thispersondoesnotexist.com/')

# 创建保存图像的目录
save_path = r'D:\Study Date\Machine learning homework\dataset\train\generatedAI'
if not os.path.exists(save_path):
    os.makedirs(save_path)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

for i in range(10000):
    # 等待页面加载
    time.sleep(0.1)

    # 获取图片元素
    img = driver.find_element(By.TAG_NAME, 'img')
    img_url = img.get_attribute('src')

    # 设置图片保存路径
    img_path = os.path.join(save_path, f'image_{i}.jpg')

    # 下载图片
    try:
        response = requests.get(img_url, headers=headers)
        with open(img_path, 'wb') as file:
            file.write(response.content)
    except requests.HTTPError as e:
        print(f'Failed to download {img_url}: {e}')

    # 刷新页面
    driver.refresh()

driver.quit()
