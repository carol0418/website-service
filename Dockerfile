# 使用官方 Python 映像作為基礎映像
FROM python:3.9-slim-buster

# 設定工作目錄
WORKDIR /app

# 將 requirements.txt 複製到容器中
COPY requirements.txt .

# 安裝所有依賴
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式碼複製到容器中
COPY . /app

# Render 會自動設定 PORT 環境變數，應用程式需要監聽此埠
ENV PORT 10000 # Render 的預設服務埠，您的應用程式應監聽此埠。
              # Flask 的 app.run() 會自動使用 PORT 環境變數。

# 設定 Python 環境變數，確保 stdout/stderr 不被緩衝
ENV PYTHONUNBUFFERED True

# 曝露 Flask 應用程式將監聽的埠
EXPOSE 10000

# 執行啟動命令
CMD ["python", "app.py"]