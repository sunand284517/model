import os
import sys
from celery import Celery
from pymongo import MongoClient
from bson.objectid import ObjectId
from model import predict_image

# Configure Celery to use Redis as broker
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app = Celery('tasks', broker=broker_url, backend=broker_url)

# Windows doesn't support prefork pool, use solo pool instead
if sys.platform == 'win32':
    app.conf.update(
        worker_pool='solo',
        worker_prefetch_multiplier=4,
        worker_max_tasks_per_child=1000,
    )

# MongoDB connection
mongo_uri = os.environ.get('MONGO_URI', 'mongodb://127.0.0.1:27017/dairy-sonogram')
client = MongoClient(mongo_uri)
db = client.get_database('dairy-sonogram')
sonogram_collection = db['sonogramresults']

@app.task(name='tasks.predict')
def predict_sonogram_task(sonogram_id, image_path):
    print(f"Picking up task for sonogram ID: {sonogram_id}, Image Path: {image_path}")
    try:
        # Update status to processing
        sonogram_collection.update_one({'_id': ObjectId(sonogram_id)}, {'$set': {'status': 'PROCESSING'}})
        
        # Run CNN inference
        classification, confidence, predicted_yield = predict_image(image_path)
        
        # Complete task and save results to DB
        sonogram_collection.update_one(
            {'_id': ObjectId(sonogram_id)}, 
            {'$set': {
                'status': 'COMPLETED',
                'classification': classification,
                'confidence': float(confidence),
                'predictedYield': float(predicted_yield)
            }}
        )
        print(f"Task completed successfully: {classification} ({confidence:.2f}) | Yield: {predicted_yield:.2f}")
        return {"status": "success", "classification": classification, "predictedYield": predicted_yield}
    except Exception as e:
        print(f"Error processing task: {str(e)}")
        sonogram_collection.update_one({'_id': ObjectId(sonogram_id)}, {'$set': {'status': 'FAILED'}})
        return {"status": "failed", "error": str(e)}
