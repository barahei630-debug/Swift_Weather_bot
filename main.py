import os
import re
import io
import uuid
import requests
import numpy as np
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語文字化け防止

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, ReplyMessageRequest,
    ImageMessage, TextMessage, FlexMessage, FlexContainer
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, LocationMessageContent

app = FastAPI()

os.makedirs("/tmp/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="/tmp/static"), name="static")

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# ユーザーごとの選択状態を記憶 (lat, lon, name)
user_locations = {}

# 事前登録した14拠点
LOCATIONS = {
    "佐賀": (33.259161, 130.244743),
    "都城": (31.796475, 131.093485),
    "西条": (34.404989, 132.713485),
    "三次": (34.811307, 132.860611),
    "邑久": (34.677947, 134.077216),
    "東近江": (35.083163, 136.196634),
    "鈴鹿": (34.845328, 136.537272),
    "佐久": (36.249108, 138.431620),
    "渡良瀬": (36.253481, 139.655928),
    "茂木": (36.532622, 140.228140),
    "小山": (36.302975, 139.778813),
    "一関": (38.934295, 141.158861),
    "大崎": (38.664064, 141.162129),
    "上士幌": (43.237823, 143.277613)
}

# ==========================================
# 気象計算・画像描画
# ==========================================

def get_dir_label(deg):
    if deg is None or np.isnan(deg): return "-"
    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return labels[int((deg + 11.25) / 22.5) % 16]

def p_to_ft(p, qnh):
    if p is None or qnh is None or qnh <= 0: return 0.0
    return (1.0 - (p / qnh) ** 0.190284) * 145366.45

def calculate_dewpoint(t, rh):
    if t is None or rh is None: return None
    a, b = 17.625, 243.04
    alpha = np.log(max(rh, 1) / 100.0) + (a * t) / (b + t)
    return (b * alpha) / (a - alpha)

def find_intersection(h_arr, t_env, t_parcel):
    diff = t_parcel - t_env
    for j in range(len(h_arr) - 1):
        if diff[j] >= 0 and diff[j+1] < 0:
            p = diff[j] / (diff[j] - diff[j+1])
            return h_arr[j] + p * (h_arr[j+1] - h_arr[j])
    return None

def estimate_thermal_strength(t_env, t_parcel, h_arr):
    diff = t_parcel - t_env
    valid_mask = diff > 0
    if not np.any(valid_mask): return 0.0, 0.0

    g = 9.81
    buoyancy_energy = 0.0
    for k in range(len(h_arr) - 1):
        if diff[k] > 0 and diff[k+1] >= 0:
            dz_m = (h_arr[k+1] - h_arr[k]) * 0.3048
            t_kelvin = t_env[k] + 273.15
            avg_diff = (diff[k] + diff[k+1]) / 2.0
            buoyancy_energy += g * (avg_diff / t_kelvin) * dz_m

    w_max = np.sqrt(2 * max(0, buoyancy_energy))
    avg_thermal_ms = w_max * 0.5
    avg_thermal_kt = avg_thermal_ms * 1.94384
    return avg_thermal_ms, avg_thermal_kt

def check_wind_shear(v_h, v_ws, v_wd):
    shear_alerts = []
    for j in range(len(v_h) - 1):
        dh = v_h[j+1] - v_h[j]
        if dh <= 0: continue

        ws1_kt, ws2_kt = v_ws[j] * 0.539957, v_ws[j+1] * 0.539957
        rad1, rad2 = np.deg2rad(v_wd[j]), np.deg2rad(v_wd[j+1])

        u1, v1 = ws1_kt * np.sin(rad1), ws1_kt * np.cos(rad1)
        u2, v2 = ws2_kt * np.sin(rad2), ws2_kt * np.cos(rad2)

        du, dv = u2 - u1, v2 - v1
        shear_kt = np.sqrt(du**2 + dv**2)
        shear_per_1000ft = (shear_kt / dh) * 1000.0

        if shear_per_1000ft >= 10.0:
            shear_alerts.append((v_h[j] + dh / 2.0, shear_per_1000ft))

    return shear_alerts

def extract_sounding_at_idx(r, idx, ground_alt_ft, p_lv):
    qnh = r['hourly']['pressure_msl'][idx]
    if qnh is None: return None

    v_h, v_t, v_td, v_rh, v_ws, v_wd = [], [], [], [], [], []

    t2 = r['hourly']['temperature_2m'][idx]
    rh2 = r['hourly']['relative_humidity_2m'][idx]
    ws2 = r['hourly']['windspeed_10m'][idx]
    wd2 = r['hourly']['winddirection_10m'][idx]

    if t2 is not None and rh2 is not None:
        v_h.append(ground_alt_ft)
        v_t.append(t2)
        v_td.append(calculate_dewpoint(t2, rh2))
        v_rh.append(rh2)
        v_ws.append(ws2 if ws2 is not None else 0)
        v_wd.append(wd2 if wd2 is not None else 0)

    for p in p_lv:
        alt = p_to_ft(p, qnh)
        if alt > ground_alt_ft + 50:
            tp = r['hourly'][f'temperature_{p}hPa'][idx]
            rhp = r['hourly'][f'relative_humidity_{p}hPa'][idx]
            wsp = r['hourly'][f'windspeed_{p}hPa'][idx]
            wdp = r['hourly'][f'winddirection_{p}hPa'][idx]

            if tp is not None and rhp is not None:
                v_h.append(alt)
                v_t.append(tp)
                v_td.append(calculate_dewpoint(tp, rhp))
                v_rh.append(rhp)
                v_ws.append(wsp if wsp is not None else 0)
                v_wd.append(wdp if wdp is not None else 0)

    if len(v_h) < 2: return None

    return {
        'h': np.array(v_h),
        't': np.array(v_t),
        'td': np.array(v_td),
        'rh': np.array(v_rh),
        'ws': np.array(v_ws),
        'wd': np.array(v_wd),
        'qnh': qnh
    }

def generate_full_sounding_image(lat: float, lon: float, date_str: str, place_name: str = "") -> bytes:
    target_hours_jst = [0, 3, 6, 9, 12, 15, 18, 21]
    models = ["jma_msm", "ecmwf_ifs04"]
    p_lv = np.array([1000, 975, 950, 925, 900, 850, 800, 700])

    params = ["pressure_msl", "temperature_2m", "relative_humidity_2m", "windspeed_10m", "winddirection_10m"]
    for p in p_lv:
        params.extend([f"temperature_{p}hPa", f"relative_humidity_{p}hPa", f"windspeed_{p}hPa", f"winddirection_{p}hPa"])

    r = None
    used_model = ""

    for m in models:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly={','.join(params)}&windspeed_unit=kmh&timezone=Asia%2FTokyo"
               f"&start_date={date_str}&end_date={date_str}&models={m}")
        try:
            res = requests.get(url, timeout=12).json()
            if 'hourly' in res:
                r = res
                used_model = m
                ground_elevation_m = res.get('elevation', 0)
                break
        except Exception:
            continue

    if not r:
        raise RuntimeError(f"{date_str} のデータ取得に失敗しました。")

    ground_alt_ft = ground_elevation_m * 3.28084

    all_temps = []
    for idx in range(min(24, len(r['hourly']['temperature_2m']))):
        try:
            t2 = r['hourly']['temperature_2m'][idx]
            rh2 = r['hourly']['relative_humidity_2m'][idx]
            if t2 is not None:
                all_temps.append(t2)
                if rh2 is not None: all_temps.append(calculate_dewpoint(t2, rh2))
            for p in p_lv:
                tp = r['hourly'][f'temperature_{p}hPa'][idx]
                rhp = r['hourly'][f'relative_humidity_{p}hPa'][idx]
                if tp is not None:
                    all_temps.append(tp)
                    if rhp is not None: all_temps.append(calculate_dewpoint(tp, rhp))
        except Exception: continue

    global_xmin = np.nanmin(all_temps) - 4 if all_temps else -20
    global_xmax = np.nanmax(all_temps) + 6 if all_temps else 40

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    axes = axes.flatten()

    for i, h_jst in enumerate(target_hours_jst):
        ax = axes[i]
        try:
            idx = h_jst
            cur_data = extract_sounding_at_idx(r, idx, ground_alt_ft, p_lv)
            if not cur_data:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
                continue

            qnh = cur_data['qnh']
            h_arr, t_arr, td_arr = cur_data['h'], cur_data['t'], cur_data['td']
            rh_arr, ws_arr, wd_arr = cur_data['rh'], cur_data['ws'], cur_data['wd']

            prev_idx = idx - 3
            prev_data = extract_sounding_at_idx(r, prev_idx, ground_alt_ft, p_lv) if prev_idx >= 0 else None

            is_surface_inversion = t_arr[1] > t_arr[0]
            t_dalr = t_arr[0] - (h_arr - h_arr[0]) * (3.0 / 1000)
            t_iso_w = td_arr[0] - (h_arr - h_arr[0]) * (1.0 / 1000)

            for j in range(len(h_arr)-1):
                dz = h_arr[j+1] - h_arr[j]
                dt = t_arr[j+1] - t_arr[j]
                lapse_rate_per_1000ft = -(dt / dz) * 1000 if dz > 0 else 0

                if dt > 0: color = '#ffd1d1'
                elif lapse_rate_per_1000ft < 3.0: color = '#e6f3ff'
                else: color = '#e6ffed'

                ax.fill_betweenx([h_arr[j], h_arr[j+1]], -50, 80, color=color, alpha=0.35)

            for j in range(len(h_arr)-1):
                if -20.0 <= t_arr[j] <= 0.0 and rh_arr[j] >= 75.0:
                    ax.fill_betweenx([h_arr[j], h_arr[j+1]], global_xmin, global_xmax,
                                     color='purple', alpha=0.12, hatch='//')

            ax.plot(t_arr, h_arr, 'r-o', lw=2.5, markersize=5, label='Temperature', zorder=6)
            ax.plot(td_arr, h_arr, 'b--o', lw=1.5, markersize=4, label='Dew Point', zorder=5)

            for j in range(len(h_arr) - 1):
                dz_ft = h_arr[j+1] - h_arr[j]
                dz_m = dz_ft * 0.3048
                dt = t_arr[j+1] - t_arr[j]

                if dz_m > 0:
                    lapse_rate_per_100m = -(dt / dz_m) * 100.0
                    mid_h = (h_arr[j] + h_arr[j+1]) / 2.0
                    mid_t = (t_arr[j] + t_arr[j+1]) / 2.0

                    if lapse_rate_per_100m > 0:
                        lapse_rate_per_1000ft = -(dt / dz_ft) * 1000.0
                        if lapse_rate_per_1000ft >= 3.0: lr_color = '#b80000'
                        elif lapse_rate_per_1000ft >= 1.5: lr_color = '#007a15'
                        else: lr_color = '#0041a8'

                        ax.text(mid_t - 0.8, mid_h, f"{lapse_rate_per_100m:.2f}°C/100m",
                                color=lr_color, fontsize=7.5, va='center', ha='right', fontweight='bold', zorder=12,
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

            if prev_data:
                prev_h = prev_data['h']
                prev_t = prev_data['t']
                for cur_t_val, cur_h_val in zip(t_arr, h_arr):
                    idx_match = np.argmin(np.abs(prev_h - cur_h_val))
                    if np.abs(prev_h[idx_match] - cur_h_val) < 50.0:
                        dt_3h = cur_t_val - prev_t[idx_match]
                        if np.abs(dt_3h) >= 1.0:
                            diff_color = '#cc0000' if dt_3h > 0 else '#0033cc'
                            sign_str = f"+{dt_3h:.1f}°C" if dt_3h > 0 else f"{dt_3h:.1f}°C"
                            ax.text(cur_t_val + 0.6, cur_h_val, f"Δ3h:{sign_str}",
                                    color=diff_color, fontsize=7.5, va='center', ha='left', fontweight='bold', zorder=12,
                                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))

            if not is_surface_inversion:
                ax.plot(t_dalr, h_arr, color='orange', ls=':', alpha=0.6, label='DALR', zorder=4)
                ax.plot(t_iso_w, h_arr, color='blue', ls=':', alpha=0.5, label='Iso-mixing Ratio', zorder=4)

            additional_ticks, additional_labels = [], []
            text_x_pos = global_xmax - 1.0

            if is_surface_inversion:
                inversion_top_idx = 1
                for j in range(1, len(t_arr)-1):
                    if t_arr[j+1] < t_arr[j]:
                        inversion_top_idx = j
                        break

                t_inv_top, h_inv_top = t_arr[inversion_top_idx], h_arr[inversion_top_idx]
                break_temp = t_inv_top + ((h_inv_top - ground_alt_ft) * (3.0 / 1000))

                ax.vlines(break_temp, ground_alt_ft, h_inv_top + 500, color='forestgreen', lw=1.5, ls='--', zorder=7)
                ax.text(text_x_pos, ground_alt_ft + 500,
                        f"Break @{break_temp:.1f}°C",
                        color='crimson', fontsize=9, ha='right', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='crimson'))
            else:
                lcl_ft = ground_alt_ft + (t_arr[0] - td_arr[0]) * 400
                thermal_top_ft = find_intersection(h_arr, t_arr, t_dalr) or ground_alt_ft
                cloud_top_ft = min(thermal_top_ft + 1500, 10000) if thermal_top_ft > lcl_ft else None

                thermal_ms, thermal_kt = estimate_thermal_strength(t_arr, t_dalr, h_arr)

                if ground_alt_ft < lcl_ft < 10500:
                    ax.axhline(lcl_ft, color='cyan', lw=1.5, ls='--', zorder=7)
                    ax.text(text_x_pos, lcl_ft + 100, f"Cloud Base (LCL): {int(lcl_ft)}ft",
                            color='darkcyan', fontsize=8, ha='right', va='bottom', fontweight='bold')
                    additional_ticks.append(lcl_ft)
                    additional_labels.append(f"LCL:{int(lcl_ft)}")

                if ground_alt_ft < thermal_top_ft < 10500:
                    ax.axhline(thermal_top_ft, color='orange', lw=2, ls='--', zorder=7)
                    label_text = f"Thermal Top: {int(thermal_top_ft)}ft"
                    if thermal_ms > 0.1:
                        label_text += f" ({thermal_ms:.1f}m/s / {thermal_kt:.1f}kt)"
                    ax.text(text_x_pos, thermal_top_ft + 100, label_text,
                            color='darkorange', fontsize=8, ha='right', va='bottom', fontweight='bold')
                    additional_ticks.append(thermal_top_ft)
                    additional_labels.append(f"Thm:{int(thermal_top_ft)}")

                if cloud_top_ft and ground_alt_ft < cloud_top_ft < 10500:
                    ax.axhline(cloud_top_ft, color='gray', lw=1.5, ls='-', zorder=7)
                    ax.text(text_x_pos, cloud_top_ft - 150, f"Cloud Top: {int(cloud_top_ft)}ft",
                            color='dimgray', fontsize=8, ha='top', va='bold')
                    additional_ticks.append(cloud_top_ft)
                    additional_labels.append(f"Top:{int(cloud_top_ft)}")

            shear_alerts = check_wind_shear(h_arr, ws_arr, wd_arr)
            for alert_h, shear_val in shear_alerts:
                if ground_alt_ft <= alert_h <= 10000:
                    ax.plot(global_xmin + 1.0, alert_h, marker='<', color='purple', markersize=8, zorder=10)
                    ax.text(global_xmin + 2.0, alert_h, f"Shear:{shear_val:.0f}kt/kft",
                            color='purple', fontsize=7, va='center', fontweight='bold')

            ax.set_ylim(ground_alt_ft, 10500)
            ax.set_xlim(global_xmin, global_xmax)

            all_ticks = sorted(list(set([t for t in [ground_alt_ft, 2000, 4000, 6000, 8000, 10000] + additional_ticks if ground_alt_ft <= t <= 10500])))
            custom_labels = []
            for t in all_ticks:
                if t == ground_alt_ft: custom_labels.append(f"Sfc:{int(ground_alt_ft)}")
                elif t in additional_ticks: custom_labels.append(additional_labels[additional_ticks.index(t)])
                else: custom_labels.append(f"{t:,}ft")

            ax.set_yticks(all_ticks)
            ax.set_yticklabels(custom_labels, fontsize=8)

            ax.grid(True, which='both', ls=':', alpha=0.5)
            ax.axhline(ground_alt_ft, color='brown', lw=2.5, zorder=8)

            if i % 2 == 0: ax.set_ylabel("Altitude (AMSL)", fontsize=10)
            if i >= 6: ax.set_xlabel("Temperature (°C)", fontsize=10)

            title_suffix = " [INVERSION]" if is_surface_inversion else ""
            ax.set_title(f"{h_jst:02d}:00 JST (Sfc: {t_arr[0]:.1f}°C){title_suffix}",
                         fontweight='bold', loc='left', fontsize=11,
                         color='crimson' if is_surface_inversion else 'black')

            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            show_p = [p for p in p_lv if p_to_ft(p, qnh) > ground_alt_ft + 50]
            ax2.set_yticks([p_to_ft(p, qnh) for p in show_p])
            ax2.set_yticklabels([f"{p}hPa" for p in show_p], color='blue', fontsize=8)

            ax3 = ax.twiny()
            ax3.barh(h_arr, ws_arr, color='#4682b4', alpha=0.35, height=250, align='center')
            ax3.set_xlim(0, max(np.max(ws_arr), 30) + 25)
            ax3.tick_params(axis='x', colors='steelblue', labelsize=8)

            for s, d, alt in zip(ws_arr, wd_arr, h_arr):
                rad = np.deg2rad(d + 180)
                ax3.quiver(s + 3, alt, np.sin(rad), np.cos(rad), color='navy', alpha=0.8,
                           scale=25, scale_units='width', width=0.015, zorder=9)
                ax3.text(s + 6, alt, f"{get_dir_label(d)}", fontsize=8, color='navy', va='center')

        except Exception as e:
            ax.text(0.5, 0.5, f"Data Error\n{str(e)}", ha='center', va='center', color='red')

    # 場所表示の判定
    title_place = f"({place_name})" if place_name else f"({lat:.3f} N, {lon:.3f} E)"

    # 1. タイトル部分
    plt.suptitle(
        f"Swift-gram (B.C.Amatsubame) {title_place} 【{date_str} JST】\n"
        f"model : {used_model.upper()} | Wind [from / km/h] Temp [℃] Altitude [ft, hPa]",
        fontsize=15, y=0.98, fontweight='bold', color='darkslategray'
    )

    # 2. 画像最下部に権利表記・免責事項を集約
    notice_text = (
        "【無断転載・他者への転送・二次利用を厳禁とします】\n"
        "※ 本データは計算値であり大きな誤差を含む可能性があります。フライト判断は各種気象情報を総合的に確認し、自己責任で行ってください。"
    )
    fig.text(
        0.5, 0.005,
        notice_text,
        ha='center', va='bottom', fontsize=9, color='crimson', fontweight='bold',
        linespacing=1.3
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.96))

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ==========================================
# ボタンメニュー (Flex Message) 生成
# ==========================================

def get_menu_flex_message(current_place="佐賀"):
    buttons_contents = []
    
    # 14拠点をボタングリッド化
    loc_keys = list(LOCATIONS.keys())
    for i in range(0, len(loc_keys), 2):
        row_buttons = []
        for name in loc_keys[i:i+2]:
            is_selected = (name == current_place)
            btn_style = "primary" if is_selected else "secondary"
            row_buttons.append({
                "type": "button",
                "action": {"type": "message", "label": f"📍 {name}", "text": name},
                "style": btn_style,
                "size": "sm",
                "height": "sm"
            })
        buttons_contents.append({
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "contents": row_buttons
        })

    flex_json = {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1DB446",
            "contents": [
                {"type": "text", "text": "🎈 日時・場所選択メニュー", "weight": "bold", "color": "#FFFFFF", "size": "md"}
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": f"現在選択中: 【{current_place}】", "weight": "bold", "size": "sm", "color": "#111111"},
                {"type": "separator"},
                {"type": "text", "text": "1. 日時を選択", "weight": "bold", "size": "xs", "color": "#666666"},
                {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {"type": "button", "action": {"type": "message", "label": "📅 今日", "text": "今日"}, "style": "primary", "color": "#007bff"},
                        {"type": "button", "action": {"type": "message", "label": "📅 明日", "text": "明日"}, "style": "primary", "color": "#17a2b8"}
                    ]
                },
                {"type": "separator"},
                {"type": "text", "text": "2. 場所を変更", "weight": "bold", "size": "xs", "color": "#666666"},
                {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "xs",
                    "contents": buttons_contents
                }
            ]
        }
    }
    return FlexMessage(alt_text="日時・場所選択メニュー", contents=FlexContainer.from_dict(flex_json))

# ==========================================
# FastAPI & LINE Webhook ハンドラー
# ==========================================

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
def handle_location(event):
    user_id = event.source.user_id
    lat, lon = event.message.latitude, event.message.longitude
    user_locations[user_id] = (lat, lon, "現在地")
    
    jst = timezone(timedelta(hours=9))
    today_str = datetime.now(jst).strftime("%Y-%m-%d")
    process_and_reply(event, lat, lon, today_str, "現在地")

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst)

    # 現在の記憶場所 (デフォルトは佐賀)
    current_lat, current_lon, current_name = user_locations.get(user_id, (33.2529, 130.2453, "佐賀"))

    # A. 登録拠点ボタンが押された場合
    if text in LOCATIONS:
        lat, lon = LOCATIONS[text]
        user_locations[user_id] = (lat, lon, text)
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(text=f"地点を【{text}】に設定しました！初期値はこちらで確認してください→ https://open-meteo.com/en/docs/model-updates"),
                        get_menu_flex_message(current_place=text)
                    ]
                )
            )
        return

    # B. 座標直接入力の場合
    match_coord = re.match(r'^([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)$', text)
    if match_coord:
        lat, lon = float(match_coord.group(1)), float(match_coord.group(2))
        user_locations[user_id] = (lat, lon, "カスタム位置")
        today_str = now_jst.strftime("%Y-%m-%d")
        process_and_reply(event, lat, lon, today_str, "カスタム位置")
        return

    # C. 日時切り替えボタンの場合
    target_date_str = None
    if text in ["今日", "きょう", "0"]:
        target_date_str = now_jst.strftime("%Y-%m-%d")
    elif text in ["明日", "あした", "翌日", "1"]:
        target_date_str = (now_jst + timedelta(days=1)).strftime("%Y-%m-%d")
    elif text in ["明後日", "あさって", "2"]:
        target_date_str = (now_jst + timedelta(days=2)).strftime("%Y-%m-%d")

    if target_date_str:
        process_and_reply(event, current_lat, current_lon, target_date_str, current_name)
    else:
        # メニュー表示
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[get_menu_flex_message(current_place=current_name)]
                )
            )

def process_and_reply(event, lat: float, lon: float, date_str: str, place_name: str = ""):
    try:
        img_bytes = generate_full_sounding_image(lat, lon, date_str, place_name)
        
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join("/tmp/static", filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
            
        base_url = "https://swift-weather-bot.onrender.com"
        img_url = f"{base_url}/static/{filename}"
        
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
                        TextMessage(text=f"[{place_name} - {date_str}] の解析エラーが発生しました:\n{e}")
                    ]
                )
            )
