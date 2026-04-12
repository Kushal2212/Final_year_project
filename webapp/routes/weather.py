import os
import requests
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

weather_bp = Blueprint("weather", __name__, url_prefix="/api")

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WEATHER_URL     = "https://api.openweathermap.org/data/2.5/weather"

DISTRICTS = {
    "ilam":      {"lat": 26.9125, "lon": 87.9258, "name": "Ilam"},
    "taplejung": {"lat": 27.3548, "lon": 87.6694, "name": "Taplejung"},
    "panchthar": {"lat": 27.1452, "lon": 87.7958, "name": "Panchthar"},
    "dhankuta":  {"lat": 26.9833, "lon": 87.3333, "name": "Dhankuta"},
    "kathmandu": {"lat": 27.7172, "lon": 85.3240, "name": "Kathmandu"},
}


def calc_risk(temp, humidity, rain):
    risks = []

    # chirke risk (aphid-spread virus → hot + dry)
    if temp > 24 and humidity < 70:
        risks.append({"disease": "chirke (चिर्के रोग)", "level": "High",
            "color": "red",
            "message": f"Hot {temp:.0f}°C + low humidity {humidity:.0f}% — ideal for aphids.",
            "action":  "Apply neem oil spray. Check for aphids daily."})
    elif temp > 20 and humidity < 80:
        risks.append({"disease": "chirke (चिर्के रोग)", "level": "Medium",
            "color": "orange",
            "message": "Moderate risk. Monitor plants for aphid presence.",
            "action":  "Inspect plants 2-3 times per week."})
    else:
        risks.append({"disease": "chirke (चिर्के रोग)", "level": "Low",
            "color": "green",
            "message": "Low aphid activity expected.",
            "action":  "Continue regular monitoring."})

    # Leaf Blight risk (fungal → warm + humid + rain)
    if 20 <= temp <= 28 and humidity > 85 and rain > 5:
        risks.append({"disease": "Leaf Blight (पात झुल्सा)", "level": "High",
            "color": "red",
            "message": f"Warm {temp:.0f}°C + high humidity {humidity:.0f}% + rain — ideal for fungus.",
            "action":  "Apply mancozeb fungicide. Improve drainage immediately."})
    elif humidity > 75 and temp > 18:
        risks.append({"disease": "Leaf Blight (पात झुल्सा)", "level": "Medium",
            "color": "orange",
            "message": "Humid conditions may encourage fungal growth.",
            "action":  "Apply preventive copper fungicide spray."})
    else:
        risks.append({"disease": "Leaf Blight (पात झुल्सा)", "level": "Low",
            "color": "green",
            "message": "Conditions not favourable for Leaf Blight.",
            "action":  "Continue regular monitoring."})

    levels  = [r["level"] for r in risks]
    overall = "High" if "High" in levels else ("Medium" if "Medium" in levels else "Low")
    return {"risks": risks, "overall": overall}


@weather_bp.route("/weather", methods=["GET"])
@jwt_required()
def weather():
    district = request.args.get("district", "ilam").lower()
    lat      = request.args.get("lat", type=float)
    lon      = request.args.get("lon", type=float)

    if lat and lon:
        location = f"{lat:.2f}, {lon:.2f}"
    elif district in DISTRICTS:
        info     = DISTRICTS[district]
        lat, lon = info["lat"], info["lon"]
        location = info["name"]
    else:
        return jsonify({"error": f"District '{district}' not found"}), 400

    # Demo mode if no API key
    if not WEATHER_API_KEY:
        temp, humidity, rain = 24.5, 82.0, 8.2
        return jsonify({
            "demo":        True,
            "location":    location,
            "temperature": temp,
            "humidity":    humidity,
            "rain_mm":     rain,
            "description": "Light rain (demo)",
            "icon":        "10d",
            "wind_kmh":    12,
            "risk":        calc_risk(temp, humidity, rain),
            "note":        "Set WEATHER_API_KEY env variable for real data",
        }), 200

    try:
        r    = requests.get(WEATHER_URL,
                            params={"lat": lat, "lon": lon,
                                    "appid": WEATHER_API_KEY, "units": "metric"},
                            timeout=5)
        r.raise_for_status()
        d        = r.json()
        temp     = d["main"]["temp"]
        humidity = d["main"]["humidity"]
        rain     = d.get("rain", {}).get("1h", 0)
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
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@weather_bp.route("/weather/districts", methods=["GET"])
def districts():
    return jsonify({
        "districts": [{"key": k, "name": v["name"]} for k, v in DISTRICTS.items()]
    }), 200