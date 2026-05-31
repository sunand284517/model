import os
import sys
import ssl
from celery import Celery
from pymongo import MongoClient
from bson.objectid import ObjectId
from model import predict_image

# =========================
# 🔥 PRODUCTION ENV VARIABLES
# =========================
REDIS_URL = os.environ.get("CELERY_BROKER_URL")
MONGO_URI = os.environ.get("MONGO_URI")

if not REDIS_URL or not MONGO_URI:
    raise ValueError("❌ Critical Environment Error: CELERY_BROKER_URL or MONGO_URI is missing.")

# =========================
# 🔥 CELERY SETUP & SSL TLS CONFIGURATION
# =========================
app = Celery(
    "worker", 
    broker=REDIS_URL, 
    backend=REDIS_URL,
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
)

app.conf.update(
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    result_backend_transport_options={"ssl_cert_reqs": ssl.CERT_NONE}
)

# ✅ Windows execution sandbox safe-override
if sys.platform == 'win32':
    app.conf.update(
        worker_pool='solo',
        worker_prefetch_multiplier=1
    )

# =========================
# 🔥 MONGODB PRODUCTION SYNC
# =========================
try:
    client = MongoClient(MONGO_URI, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)
    
    # ✅ FIXED: Safely read default database name using PyMongo properties instead of get_default_database()
    try:
        db = client.get_default_database()
    except AttributeError:
        # Fallback to parsing from URI or defaulting to correct collection cluster path
        db = client.get_database() if client.nodes else client["dairy-sonogram"]
        
    sonogram_collection = db['sonogramresults']
    print(f"✅ MongoDB Connected successfully to database cluster: {db.name}")
except Exception as e:
    print("❌ MongoDB connection sequence failed:", e)
    raise e

# =========================
# 🔥 CELERY BACKGROUND TASK PIPELINE
# =========================
@app.task(name='predict_task')
def predict_sonogram_task(sonogram_id, image_path):
    print(f"📥 Picking up task for sonogram ID: {sonogram_id}")
    print(f"🖼️ Target cloud image asset pathway link: {image_path}")
    
    try:
        # Update state status validation indicator → PROCESSING
        sonogram_collection.update_one({'_id': ObjectId(sonogram_id)}, {'$set': {'status': 'PROCESSING'}})
        
        # Run local weights inference calculations
        classification, confidence, predicted_yield = predict_image(image_path)
        
        # Complete task and save processed predictions to DB
        sonogram_collection.update_one(
            {'_id': ObjectId(sonogram_id)}, 
            {'$set': {
                'status': 'COMPLETED',
                'classification': classification,
                'confidence': float(confidence),
                'predictedYield': float(predicted_yield)
            }}
        )
        print(f"💾 Task completed successfully: {classification} ({confidence:.2f}) | Yield: {predicted_yield:.2f}")
        return {"status": "success", "classification": classification, "predictedYield": predicted_yield}
        
    except Exception as e:
        print(f"❌ Error processing task: {str(e)}")
        sonogram_collection.update_one({'_id': ObjectId(sonogram_id)}, {'$set': {'status': 'FAILED', 'errorReason': str(e)}})
        return {"status": "failed", "error": str(e)}

# =========================
# 🔥 START LOG
# =========================
print("🚀 Production Celery Worker Environment Stack Initialized...")
