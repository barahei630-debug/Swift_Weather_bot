import os
import re
import io
import uuid
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, ReplyMessageRequest, ImageMessage, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, LocationMessageContent

app = FastAPI()

# 生成した画像を一時保存するディレクトリを作成し、静的ファイルとして公開
os.makedirs("/tmp/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="/tmp/static"), name="static")

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


def generate_emagram(lat: float, lon: float) -> bytes:
    """Open-Meteoから高層データを取得しエマグラム(Skew-T)画像を生成"""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,"
        f"dewpoint_1000hPa,dewpoint_925hPa,dewpoint_850hPa,dewpoint_700hPa,dewpoint_500hPa&"
        f"timezone=Asia%2FTokyo"
    )
    res = requests.get(url, timeout=10).json()
    hourly = res.get('hourly', {})

    pressures = np.array([1000, 925, 850, 700, 500]) * units.hPa
    temps = np.array([
        hourly.get('temperature_1000hPa', [0])[0],
        hourly.get('temperature_925hPa', [0])[0],
        hourly.get('temperature_850hPa', [0])[0],
        hourly.get('temperature_700hPa', [0])[0],
        hourly.get('temperature_500hPa', [0])[0]
    ]) * units.degC

    dews = np.array([
        hourly.get('dewpoint_1000hPa', [0])[0],
        hourly.get('dewpoint_925hPa', [0])[0],
        hourly.get('dewpoint_850hPa', [0])[0],
        hourly.get('dewpoint_700hPa', [0])[0],
        hourly.get('dewpoint_500hPa', [0])[0]
    ]) * units.degC

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    
    ax.plot(temps, pressures, 'r-o', linewidth=2, label='Temperature (°C)')
    ax.plot(dews, pressures, 'g-o', linewidth=2, label='Dewpoint (°C)')
    
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_yticks([1000, 925, 850, 700, 500])
    ax.set_yticklabels(['1000', '925', '850', '700', '500'])
    
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_title(f'Sounding Profile ({lat:.2f} N, {lon:.2f} E)')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


@handler.add(MessageEvent, message=LocationMessageContent)
def handle_location(event, request: Request):
    lat = event.message.latitude
    lon = event.message.longitude
    process_and_reply(event, request, lat, lon)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event, request: Request):
    text = event.message.text.strip()
    match = re.match(r'^([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)$', text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        process_and_reply(event, request, lat, lon)
    else:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text="位置情報を送信するか、「34.4, 132.7」のように緯度と経度を入力してください。"
                        )
                    ]
                )
            )


def process_and_reply(event, request: Request, lat: float, lon: float):
    try:
        # 1. エマグラム画像生成
        img_bytes = generate_emagram(lat, lon)
        
        # 2. ランダムなファイル名で Render サーバ上に保存
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join("/tmp/static", filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
            
        # 3. 自分のRenderサーバーのURLを取得して画像URLを構築
        base_url = str(request.base_url).rstrip('/')
        img_url = f"{base_url}/static/{filename}"
        
        # 4. LINEへ返信
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        ImageMessage(
                            original_content_url=img_url,
                            preview_image_url=img_url
                        )
                    ]
                )
            )
    except Exception as e:
        print(f"Error occurred: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(text=f"エラーが発生しました:\n{e}")
                    ]
                )
            )
