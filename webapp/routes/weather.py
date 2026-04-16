from datetime import datetime
from collections import defaultdict
import os
import requests
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

weather_bp = Blueprint("weather", __name__, url_prefix="/api")

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

DISTRICTS = {
    "sankhuwasabha": {"lat": 27.5333, "lon": 87.1833, "name": "Sankhuwasabha"},
    "ilam":          {"lat": 26.9125, "lon": 87.9258, "name": "Ilam"},
    "taplejung":     {"lat": 27.3548, "lon": 87.6694, "name": "Taplejung"},
    "panchthar":     {"lat": 27.1452, "lon": 87.7958, "name": "Panchthar"},
}


# ── Disease risk from single weather reading ──────────────────────────────
def calc_risk(temp, humidity, rain):
    risks = []

    # Chhirke risk (aphid-spread virus → hot + dry)
    if temp > 24 and humidity < 70:
        risks.append({
            "disease": "Chhirke (छिर्के रोग)", "level": "High", "color": "red",
            "message": f"Hot {temp:.0f}°C + low humidity {humidity:.0f}% — ideal for aphids.",
            "action":  "Apply neem oil spray. Check for aphids daily."
        })
    elif temp > 20 and humidity < 80:
        risks.append({
            "disease": "Chhirke (छिर्के रोग)", "level": "Medium", "color": "orange",
            "message": "Moderate risk. Monitor plants for aphid presence.",
            "action":  "Inspect plants 2-3 times per week."
        })
    else:
        risks.append({
            "disease": "Chhirke (छिर्के रोग)", "level": "Low", "color": "green",
            "message": "Low aphid activity expected.",
            "action":  "Continue regular monitoring."
        })

    # Leaf Blight risk (fungal → warm + humid + rain)
    if 20 <= temp <= 28 and humidity > 85 and rain > 5:
        risks.append({
            "disease": "Leaf Blight (पात झुल्सा)", "level": "High", "color": "red",
            "message": f"Warm {temp:.0f}°C + high humidity {humidity:.0f}% + rain — ideal for fungus.",
            "action":  "Apply mancozeb fungicide. Improve drainage immediately."
        })
    elif humidity > 75 and temp > 18:
        risks.append({
            "disease": "Leaf Blight (पात झुल्सा)", "level": "Medium", "color": "orange",
            "message": "Humid conditions may encourage fungal growth.",
            "action":  "Apply preventive copper fungicide spray."
        })
    else:
        risks.append({
            "disease": "Leaf Blight (पात झुल्सा)", "level": "Low", "color": "green",
            "message": "Conditions not favourable for Leaf Blight.",
            "action":  "Continue regular monitoring."
        })

    levels = [r["level"] for r in risks]
    overall = "High" if "High" in levels else (
        "Medium" if "Medium" in levels else "Low")
    return {"risks": risks, "overall": overall}


# ── Disease risk summary from forecast list ───────────────────────────────
def calc_forecast_risk(forecast_list):
    summary = {
        "chhirke":     {"High": 0, "Medium": 0, "Low": 0},
        "leaf_blight": {"High": 0, "Medium": 0, "Low": 0},
    }
    for item in forecast_list:
        temp = item["main"]["temp"]
        humidity = item["main"]["humidity"]
        rain = item.get("rain", {}).get("3h", 0)

        if temp > 24 and humidity < 70:
            summary["chhirke"]["High"] += 1
        elif temp > 20 and humidity < 80:
            summary["chhirke"]["Medium"] += 1
        else:
            summary["chhirke"]["Low"] += 1

        if 20 <= temp <= 28 and humidity > 85 and rain > 3:
            summary["leaf_blight"]["High"] += 1
        elif humidity > 75:
            summary["leaf_blight"]["Medium"] += 1
        else:
            summary["leaf_blight"]["Low"] += 1

    def get_level(counts):
        if counts["High"] > 0:
            return "High"
        if counts["Medium"] > 0:
            return "Medium"
        return "Low"

    return {
        "chhirke":     get_level(summary["chhirke"]),
        "leaf_blight": get_level(summary["leaf_blight"]),
    }


# ── Group forecast items by day ───────────────────────────────────────────
def group_by_day(forecast_list):
    days = defaultdict(list)
    for item in forecast_list:
        date = item["dt_txt"].split(" ")[0]
        days[date].append(item)

    result = []
    for date, items in list(days.items())[:3]:
        avg_temp = sum(i["main"]["temp"] for i in items) / len(items)
        avg_humidity = sum(i["main"]["humidity"] for i in items) / len(items)
        result.append({
            "date":         date,
            "avg_temp":     round(avg_temp, 1),
            "avg_humidity": round(avg_humidity, 1),
        })
    return result


# ── Risk level per day ────────────────────────────────────────────────────
def daily_risk_timeline(forecast_list):
    days = defaultdict(list)
    for item in forecast_list:
        date = item["dt_txt"].split(" ")[0]
        days[date].append(item)

    timeline = []
    for date, items in list(days.items())[:3]:
        avg_temp = sum(i["main"]["temp"] for i in items) / len(items)
        avg_humidity = sum(i["main"]["humidity"] for i in items) / len(items)
        total_rain = sum(i.get("rain", {}).get("3h", 0) for i in items)
        risk = calc_risk(avg_temp, avg_humidity, total_rain)
        timeline.append({"date": date, "overall": risk["overall"]})
    return timeline


# ── Generate top-level alert ──────────────────────────────────────────────
def generate_alert(risk_timeline):
    for day in risk_timeline:
        if day["overall"] == "High":
            return f"⚠️ High disease risk on {day['date']}. Take preventive action!"
    return "✅ No major disease risk in the coming days."


# ═════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════

@weather_bp.route("/weather", methods=["GET"])
@jwt_required()
def weather():
    district = request.args.get("district", "ilam").lower()
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat and lon:
        location = f"{lat:.2f}, {lon:.2f}"
    elif district in DISTRICTS:
        info = DISTRICTS[district]
        lat, lon = info["lat"], info["lon"]
        location = info["name"]
    else:
        return jsonify({"error": f"District '{district}' not found"}), 400

    # Demo mode — no API key configured
    if not WEATHER_API_KEY:
        temp, humidity, rain = 24.5, 82.0, 8.2
        return jsonify({
            "demo":        True,
            "location":    location,
            "temperature": temp,
            "humidity":    humidity,
            "rain_mm":     rain,
            "description": "Light rain (demo data)",
            "icon":        "10d",
            "wind_kmh":    12,
            "risk":        calc_risk(temp, humidity, rain),
            "note":        "Set WEATHER_API_KEY environment variable for live data",
        }), 200

    try:
        r = requests.get(WEATHER_URL,
                         params={"lat": lat, "lon": lon,
                                 "appid": WEATHER_API_KEY, "units": "metric"},
                         timeout=5)
        r.raise_for_status()
        d = r.json()
        temp = d["main"]["temp"]
        humidity = d["main"]["humidity"]
        rain = d.get("rain", {}).get("1h", 0)
        return jsonify({
            "demo":        False,
            "location":    location,
            "temperature": round(temp, 1),
            "humidity":    humidity,
            "rain_mm":     round(rain, 1),
            "description": d["weather"][0]["description"].capitalize(),
            "icon":        d["weather"][0]["icon"],
            "wind_kmh":    round(d["wind"]["speed"] * 3.6, 1),
            "risk":        calc_risk(temp, humidity, rain),
        }), 200

    except requests.exceptions.HTTPError:
        return jsonify({"error": "Weather API error (invalid key or quota exceeded)"}), 502
    except requests.exceptions.RequestException:
        return jsonify({"error": "Unable to connect to weather service"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@weather_bp.route("/forecast", methods=["GET"])
@jwt_required()
def forecast():
    district = request.args.get("district", "ilam").lower()
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat and lon:
        location = f"{lat:.2f}, {lon:.2f}"
    elif district in DISTRICTS:
        info = DISTRICTS[district]
        lat, lon = info["lat"], info["lon"]
        location = info["name"]
    else:
        return jsonify({"error": f"District '{district}' not found"}), 400

    # Demo mode
    if not WEATHER_API_KEY:
        demo_items = [
            {"main": {"temp": 25, "humidity": 80}, "rain": {"3h": 5}},
            {"main": {"temp": 26, "humidity": 85}, "rain": {"3h": 8}},
            {"main": {"temp": 24, "humidity": 78}, "rain": {"3h": 2}},
            {"main": {"temp": 23, "humidity": 82}, "rain": {"3h": 4}},
            {"main": {"temp": 25, "humidity": 88}, "rain": {"3h": 10}},
        ]
        risk_timeline = [
            {"date": "Day 1", "overall": "Medium"},
            {"date": "Day 2", "overall": "High"},
            {"date": "Day 3", "overall": "Low"},
        ]
        daily_summary = [
            {"date": "Day 1", "avg_temp": 25.0, "avg_humidity": 80.0},
            {"date": "Day 2", "avg_temp": 26.0, "avg_humidity": 85.0},
            {"date": "Day 3", "avg_temp": 24.0, "avg_humidity": 78.0},
        ]
        return jsonify({
            "demo":          True,
            "location":      location,
            "daily_summary": daily_summary,
            "risk_timeline": risk_timeline,
            "alert":         generate_alert(risk_timeline),
            "risk_summary":  calc_forecast_risk(demo_items),
        }), 200

    try:
        resp = requests.get(FORECAST_URL,
                            params={"lat": lat, "lon": lon,
                                    "appid": WEATHER_API_KEY, "units": "metric"},
                            timeout=5)
        resp.raise_for_status()
        data = resp.json()
        forecast_list = data.get("list", [])

        daily_summary = group_by_day(forecast_list)
        risk_timeline = daily_risk_timeline(forecast_list)
        alert = generate_alert(risk_timeline)
        risk_summary = calc_forecast_risk(forecast_list[:8])

        return jsonify({
            "demo":          False,
            "location":      location,
            "daily_summary": daily_summary,
            "risk_timeline": risk_timeline,
            "alert":         alert,
            "risk_summary":  risk_summary,
        }), 200

    except requests.exceptions.HTTPError:
        return jsonify({"error": "Forecast API error (invalid key or quota exceeded)"}), 502
    except requests.exceptions.RequestException:
        return jsonify({"error": "Unable to connect to forecast service"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@weather_bp.route("/weather/districts", methods=["GET"])
def districts():
    return jsonify({
        "districts": [{"key": k, "name": v["name"]} for k, v in DISTRICTS.items()]
    }), 200
