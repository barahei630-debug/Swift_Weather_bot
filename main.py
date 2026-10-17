import os
import re
import io
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIなしのサーバー環境用
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.units import units

from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, ReplyMessageRequest, ImageMessage, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, LocationMessageContent

app = FastAPI()

# 環境変数の読み込み
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)


def generate_emagram(lat: float, lon: float) -> bytes:
    """
    Open-Meteo APIから高層データを取得し、エマグラム(Skew-T Log-P)画像を生成してバイトデータを返す
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,"
        f"dewpoint_1000hPa,dewpoint_925hPa,dewpoint_850hPa,dewpoint_700hPa,dewpoint_500hPa&"
        f"timezone=Asia%2FTokyo"
    )
    res = requests.get(url, timeout=10).json()
    hourly = res.get('hourly', {})

    # 1000hPa〜500hPaのデータを取得（最新時刻: index=0）
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

    # グラフ描画 (Skew-T 簡易表示)
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

    # メモリ上のPNG形式バイナリとして出力
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    """
    生成した画像バイナリをImgur APIにアップロードして公開URLを取得する
    """
    headers = {'Authorization': f'Client-ID {client_id}'}
    response = requests.post(
        'https://api.imgur.com/3/image',
        headers=headers,
        files={'image': image_bytes},
        timeout=15
    )
    res_data = response.json()
    if response.status_code == 200 and res_data.get('success'):
        return res_data['data']['link']
    else:
        raise RuntimeError(f"Imgur Upload Failed: {res_data}")


@app.post("/callback")
async def callback(request: Request):
    """LINE Messaging API からの Webhook を受けるエンドポイント"""
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


@handler.add(MessageEvent, message=LocationMessageContent)
def handle_location(event):
    """スマホから「位置情報」が送信された場合の処理"""
    lat = event.message.latitude
    lon = event.message.longitude
    process_and_reply(event, lat, lon)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    """「34.4, 132.7」のようなテキストで座標が送信された場合の処理"""
    text = event.message.text.strip()
    match = re.match(r'^([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)$', text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        process_and_reply(event, lat, lon)
    else:
        # 該当しないメッセージの場合は案内を表示
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


def process_and_reply(event, lat: float, lon: float):
    """画像生成・アップロード・LINE返信を一括処理"""
    try:
        # 1. 画像生成
        img_bytes = generate_emagram(lat, lon)
        
        # 2. 画像ホスティングへアップロード
        img_url = upload_to_imgur(img_bytes, IMGUR_CLIENT_ID)
        
        # 3. LINEへ画像メッセージ返信
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
                        TextMessage(text=f"処理中にエラーが発生しました:\n{e}")
                    ]
                )
            )
