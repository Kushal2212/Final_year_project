import os
import logging
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
 
log = logging.getLogger(__name__)
 
# ── lazy imports so missing packages don't crash the app ──────────────────
def _get_db():
    from webapp.extensions import db
    return db
 
def _get_models():
    from webapp.models import Farmer, Prediction, User
    return Farmer, Prediction, User
 
 
sms_bp = Blueprint('sms', __name__, url_prefix='/api/sms')
 
 
# ════════════════════════════════════════════════════════════════════════════
#  SMS SENDER  — swap provider by setting SMS_PROVIDER in .env
# ════════════════════════════════════════════════════════════════════════════
def send_sms(phone: str, message: str) -> dict:
    """
    Send SMS via configured provider.
    Returns {'success': True/False, 'info': str}
    """
    provider = os.getenv('SMS_PROVIDER', 'textbelt').lower()
 
    # Normalise phone to E.164 for Nepal (+977...)
    if not phone.startswith('+'):
        phone = '+977' + phone.lstrip('0')
 
    if provider == 'twilio':
        return _send_twilio(phone, message)
    elif provider == 'sparrow':
        return _send_sparrow(phone, message)
    else:
        return _send_textbelt(phone, message)
 
 
def _send_twilio(phone, message):
    try:
        from twilio.rest import Client
        client = Client(
            os.getenv('TWILIO_SID'),
            os.getenv('TWILIO_TOKEN')
        )
        msg = client.messages.create(
            body=message,
            from_=os.getenv('TWILIO_FROM'),
            to=phone
        )
        return {'success': True, 'info': msg.sid}
    except Exception as e:
        log.error(f'Twilio error to {phone}: {e}')
        return {'success': False, 'info': str(e)}
 
 
def _send_sparrow(phone, message):
    """Sparrow SMS — Nepal's local gateway (cheap, Nepali number friendly)"""
    try:
        import requests
        # Strip +977 prefix for Sparrow (wants 98XXXXXXXX format)
        local = phone.replace('+977', '')
        r = requests.post(
            'http://api.sparrowsms.com/v2/sms/',
            data={
                'token':  os.getenv('SPARROW_TOKEN'),
                'from':   os.getenv('SPARROW_FROM', 'Cardamom'),
                'to':     local,
                'text':   message,
            },
            timeout=10
        )
        d = r.json()
        ok = d.get('response_code') == 200
        return {'success': ok, 'info': str(d)}
    except Exception as e:
        log.error(f'Sparrow SMS error to {phone}: {e}')
        return {'success': False, 'info': str(e)}
 
 
def _send_textbelt(phone, message):
    """TextBelt — 1 free SMS per day per IP. Good for testing."""
    try:
        import requests
        r = requests.post('https://textbelt.com/text', {
            'phone':   phone,
            'message': message,
            'key':     os.getenv('TEXTBELT_KEY', 'textbelt'),  # 'textbelt' = free tier
        }, timeout=10)
        d = r.json()
        return {'success': d.get('success', False), 'info': str(d)}
    except Exception as e:
        return {'success': False, 'info': str(e)}
 
 
# ════════════════════════════════════════════════════════════════════════════
#  MESSAGE BUILDER — Nepali + English
# ════════════════════════════════════════════════════════════════════════════
RISK_NE = {'High': 'उच्च', 'Medium': 'मध्यम', 'Low': 'कम'}
RISK_EMOJI = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
DISEASE_NE = {
    'chirke':      'चिर्के रोग',
    'chhirke':     'चिर्के रोग',
    'leaf_blight': 'पात झुल्सा',
    'healthy':     'स्वस्थ',
}
 
 
def build_weekly_sms(farmer, weather_data, recent_predictions, lang='ne'):
    """Build the weekly SMS message for a farmer."""
    district_name = farmer.district.title()
    now = datetime.utcnow().strftime('%Y-%m-%d')
 
    if lang == 'ne':
        # Nepali message (short — SMS max 160 chars)
        risk = weather_data.get('risk', {}).get('overall', 'Low')
        risk_ne = RISK_NE.get(risk, 'कम')
        risk_emoji = RISK_EMOJI.get(risk, '🟢')
 
        # Top disease risk
        risks = weather_data.get('risk', {}).get('risks', [])
        top_risk_line = ''
        for r in risks:
            if r['level'] == 'High':
                top_risk_line = f"\n⚠️ {r['disease']}: {r['action'][:50]}"
                break
 
        # Recent prediction summary
        pred_line = ''
        if recent_predictions:
            last = recent_predictions[0]
            disease_name = DISEASE_NE.get(last['disease'], last['disease'])
            pred_line = f"\n📊 पछिल्लो स्क्यान: {disease_name} ({last['confidence']}%)"
 
        temp = weather_data.get('temperature', '?')
        humidity = weather_data.get('humidity', '?')
 
        msg = (
            f"🌿 अलैँची साप्ताहिक सतर्कता ({now})\n"
            f"📍 {district_name}: {temp}°C, आर्द्रता {humidity}%\n"
            f"{risk_emoji} रोग जोखिम: {risk_ne}"
            f"{top_risk_line}"
            f"{pred_line}\n"
            f"cardamomdx.com मा जानकारीका लागि"
        )
    else:
        # English message
        risk = weather_data.get('risk', {}).get('overall', 'Low')
        risks = weather_data.get('risk', {}).get('risks', [])
        top = next((r for r in risks if r['level'] == 'High'), None)
 
        pred_line = ''
        if recent_predictions:
            last = recent_predictions[0]
            pred_line = f"\nLast scan: {last['disease'].replace('_',' ')} ({last['confidence']}%)"
 
        msg = (
            f"🌿 CardamomDx Weekly Alert ({now})\n"
            f"📍 {district_name}: {weather_data.get('temperature','?')}°C\n"
            f"{RISK_EMOJI.get(risk,'🟢')} Disease risk: {risk}"
            + (f"\n⚠️ {top['disease']}: {top['action'][:50]}" if top else '')
            + pred_line
            + "\ncardamomdx.com"
        )
 
    # Truncate to 160 chars if needed (standard SMS limit)
    return msg[:160]
 
 
# ════════════════════════════════════════════════════════════════════════════
#  WEEKLY JOB  — called by scheduler every Sunday at 8 AM Nepal time
# ════════════════════════════════════════════════════════════════════════════
def run_weekly_alerts(app):
    """
    Main weekly job. Call this from the scheduler.
    Fetches weather for each district, gets active farmers, sends SMS.
    """
    with app.app_context():
        Farmer, Prediction, User = _get_models()
        db = _get_db()
 
        farmers = Farmer.query.filter_by(is_active=True).all()
        if not farmers:
            log.info('Weekly SMS: no active farmers')
            return
 
        log.info(f'Weekly SMS: sending to {len(farmers)} farmers')
 
        # Cache weather per district to avoid repeated API calls
        weather_cache = {}
 
        sent = 0
        failed = 0
        for farmer in farmers:
            try:
                # Get weather for farmer's district
                district = farmer.district or 'ilam'
                if district not in weather_cache:
                    weather_cache[district] = _fetch_weather(app, district)
                weather = weather_cache[district]
 
                # Get farmer's last 3 predictions (via linked user if exists)
                recent = []
                # Try matching phone to user (optional link)
                preds = Prediction.query.filter(
                    Prediction.created_at >= datetime.utcnow() - timedelta(days=7)
                ).order_by(Prediction.created_at.desc()).limit(3).all()
                recent = [p.to_dict() for p in preds]
 
                # Build message
                message = build_weekly_sms(farmer, weather, recent, farmer.language)
 
                # Send
                result = send_sms(farmer.phone, message)
 
                if result['success']:
                    farmer.last_sms_at = datetime.utcnow()
                    sent += 1
                    log.info(f'SMS sent to {farmer.phone} ({farmer.name})')
                else:
                    failed += 1
                    log.warning(f'SMS failed to {farmer.phone}: {result["info"]}')
 
            except Exception as e:
                failed += 1
                log.error(f'Error sending to farmer {farmer.id}: {e}')
 
        db.session.commit()
        log.info(f'Weekly SMS complete: {sent} sent, {failed} failed')
        return {'sent': sent, 'failed': failed}
 
 
def _fetch_weather(app, district):
    """Fetch weather from your existing weather API."""
    try:
        import requests
        with app.test_client() as client:
            r = client.get(f'/api/weather?district={district}')
            return r.get_json() or {}
    except Exception as e:
        log.error(f'Weather fetch failed for {district}: {e}')
        return {'temperature': '?', 'humidity': '?', 'rain_mm': 0,
                'risk': {'overall': 'Medium', 'risks': []}, 'demo': True}
 
 
# ════════════════════════════════════════════════════════════════════════════
#  SCHEDULER SETUP
# ════════════════════════════════════════════════════════════════════════════
def start_scheduler(app):
    """
    Call this in your create_app() after initialising extensions.
 
    Example in app/__init__.py:
        from webapp.sms_alert_system import start_scheduler
        start_scheduler(app)
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
 
        scheduler = BackgroundScheduler()
 
        # Every Sunday at 8:00 AM Nepal Standard Time (UTC+5:45 → UTC 2:15)
        scheduler.add_job(
            func=run_weekly_alerts,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=15, timezone='UTC'),
            args=[app],
            id='weekly_sms_alert',
            replace_existing=True,
            name='Weekly SMS Disease Alert',
        )
        scheduler.start()
        log.info('✅ SMS scheduler started — weekly alerts every Sunday 8:00 AM NST')
        return scheduler
 
    except ImportError:
        log.warning('APScheduler not installed. Run: pip install apscheduler')
        return None
    except Exception as e:
        log.error(f'Scheduler failed to start: {e}')
        return None
 
 
# ════════════════════════════════════════════════════════════════════════════
#  ADMIN CHECK
# ════════════════════════════════════════════════════════════════════════════
def check_admin():
    _, _, User = _get_models()
    user = User.query.get(int(get_jwt_identity()))
    return user if (user and user.is_admin) else None
 
 
# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC ROUTES — farmer self-registration
# ════════════════════════════════════════════════════════════════════════════
@sms_bp.route('/register', methods=['POST'])
def register_farmer():
    """Farmers can register their phone for weekly alerts."""
    data     = request.get_json(silent=True) or {}
    name     = (data.get('name') or '').strip()
    phone    = (data.get('phone') or '').strip()
    district = (data.get('district') or 'ilam').strip().lower()
    language = (data.get('language') or 'ne').strip().lower()
 
    if not name or not phone:
        return jsonify({'error': 'Name and phone number are required'}), 400
 
    # Basic Nepal phone validation
    clean = phone.replace(' ', '').replace('-', '').lstrip('+977').lstrip('0')
    if not (clean.isdigit() and 9 <= len(clean) <= 10):
        return jsonify({'error': 'Enter a valid Nepal phone number (e.g. 9812345678)'}), 400
 
    Farmer, _, _ = _get_models()
    db = _get_db()
 
    existing = Farmer.query.filter_by(phone=phone).first()
    if existing:
        if existing.is_active:
            return jsonify({'message': f'✅ {existing.name}, you are already registered!'}), 200
        existing.is_active = True
        existing.name      = name
        existing.district  = district
        existing.language  = language
        db.session.commit()
        return jsonify({'message': '✅ Welcome back! SMS alerts re-activated.'}), 200
 
    farmer = Farmer(name=name, phone=phone, district=district, language=language)
    db.session.add(farmer)
    db.session.commit()
 
    # Send welcome SMS
    welcome_ne = f"🌿 नमस्ते {name}! अलैँची साप्ताहिक सतर्कता सेवामा स्वागत छ। हरेक आइतबार रोग र मौसम अलर्ट पाउनुहुनेछ। - CardamomDx"
    welcome_en = f"🌿 Hi {name}! You're registered for CardamomDx weekly disease & weather alerts every Sunday. - cardamomdx.com"
    welcome_msg = welcome_ne if language == 'ne' else welcome_en
    send_sms(phone, welcome_msg[:160])
 
    return jsonify({
        'message': f'✅ Registered successfully! You will receive weekly SMS alerts every Sunday morning.',
        'farmer_id': farmer.id,
    }), 201
 
 
@sms_bp.route('/unregister', methods=['POST'])
def unregister_farmer():
    """Opt out of SMS alerts."""
    data  = request.get_json(silent=True) or {}
    phone = (data.get('phone') or '').strip()
    if not phone:
        return jsonify({'error': 'Phone number is required'}), 400
 
    Farmer, _, _ = _get_models()
    db = _get_db()
 
    farmer = Farmer.query.filter_by(phone=phone).first()
    if not farmer:
        return jsonify({'error': 'Phone number not found in our system'}), 404
 
    farmer.is_active = False
    db.session.commit()
    return jsonify({'message': '✅ You have been unregistered from SMS alerts.'}), 200
 
 
# ════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ════════════════════════════════════════════════════════════════════════════
@sms_bp.route('/farmers', methods=['GET'])
@jwt_required()
def list_farmers():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403
 
    Farmer, _, _ = _get_models()
    page     = request.args.get('page',   1,  type=int)
    per_page = request.args.get('limit',  10, type=int)
    district = request.args.get('district', None)
 
    query = Farmer.query.order_by(Farmer.created_at.desc())
    if district:
        query = query.filter_by(district=district)
 
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    total_active = Farmer.query.filter_by(is_active=True).count()
 
    return jsonify({
        'farmers':      [f.to_dict() for f in paginated.items],
        'total':        paginated.total,
        'total_active': total_active,
        'page':         page,
        'pages':        paginated.pages,
    }), 200
 
 
@sms_bp.route('/farmers/<int:fid>', methods=['DELETE'])
@jwt_required()
def delete_farmer(fid):
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403
 
    Farmer, _, _ = _get_models()
    db = _get_db()
    farmer = Farmer.query.get_or_404(fid)
    db.session.delete(farmer)
    db.session.commit()
    return jsonify({'message': f'Farmer {farmer.name} removed'}), 200
 
 
@sms_bp.route('/send-now', methods=['POST'])
@jwt_required()
def send_now():
    """Admin: trigger weekly alerts immediately (for testing)."""
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403
 
    data     = request.get_json(silent=True) or {}
    district = data.get('district', None)   # None = all districts
 
    Farmer, Prediction, _ = _get_models()
    query = Farmer.query.filter_by(is_active=True)
    if district:
        query = query.filter_by(district=district)
    farmers = query.all()
 
    if not farmers:
        return jsonify({'error': 'No active farmers to send to'}), 400
 
    weather_cache = {}
    sent = 0
    failed = 0
 
    for farmer in farmers:
        try:
            d = farmer.district or 'ilam'
            if d not in weather_cache:
                weather_cache[d] = _fetch_weather(current_app._get_current_object(), d)
            weather = weather_cache[d]
 
            recent = [p.to_dict() for p in
                      Prediction.query.order_by(Prediction.created_at.desc()).limit(3).all()]
            message = build_weekly_sms(farmer, weather, recent, farmer.language)
            result  = send_sms(farmer.phone, message)
 
            if result['success']:
                from webapp.extensions import db
                farmer.last_sms_at = datetime.utcnow()
                sent += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log.error(f'send-now error for farmer {farmer.id}: {e}')
 
    from webapp.extensions import db
    db.session.commit()
 
    return jsonify({
        'message': f'Sent to {sent} farmer(s).',
        'sent':    sent,
        'failed':  failed,
    }), 200
 
 
@sms_bp.route('/send-custom', methods=['POST'])
@jwt_required()
def send_custom():
    """Admin: send a custom one-off SMS to all active farmers."""
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403
 
    data    = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    if len(message) > 160:
        return jsonify({'error': f'Message too long ({len(message)}/160 characters)'}), 400
 
    Farmer, _, _ = _get_models()
    db = _get_db()
    farmers = Farmer.query.filter_by(is_active=True).all()
    if not farmers:
        return jsonify({'error': 'No active farmers'}), 400
 
    sent = 0
    failed = 0
    for farmer in farmers:
        result = send_sms(farmer.phone, message)
        if result['success']:
            farmer.last_sms_at = datetime.utcnow()
            sent += 1
        else:
            failed += 1
 
    db.session.commit()
    return jsonify({'message': f'Sent to {sent} farmer(s).', 'sent': sent, 'failed': failed}), 200
 
 
@sms_bp.route('/stats', methods=['GET'])
@jwt_required()
def sms_stats():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403
 
    Farmer, _, _ = _get_models()
    total        = Farmer.query.count()
    active       = Farmer.query.filter_by(is_active=True).count()
    by_district  = {}
    for f in Farmer.query.filter_by(is_active=True).all():
        by_district[f.district] = by_district.get(f.district, 0) + 1
 
    last_sent = Farmer.query.filter(
        Farmer.last_sms_at.isnot(None)
    ).order_by(Farmer.last_sms_at.desc()).first()
 
    return jsonify({
        'total_farmers':   total,
        'active_farmers':  active,
        'by_district':     by_district,
        'last_sent_at':    last_sent.last_sms_at.isoformat() if last_sent else None,
        'provider':        os.getenv('SMS_PROVIDER', 'textbelt'),
    }), 200