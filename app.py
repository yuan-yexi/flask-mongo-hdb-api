from flask import Flask, request
from dotenv import load_dotenv
from flask_restful import Resource, Api
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

import os
import pymongo
import xgboost as xgb

app = Flask(__name__)
api = Api(app)

load_dotenv()
MONGO_USERNAME = os.environ.get('MONGO_USERNAME')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')

db_client = pymongo.MongoClient(f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@db-mongodb-ams3-93766-5977c341.mongo.ondigitalocean.com", tls=True, tlsCAFile='ca-certificate.crt')
db = db_client.hdb_price_db
collection = db['resale_price_collection']

class LatestMonth(Resource):
    def get(self, month):
        query ={ "month": month }
        documents = collection.find(query).limit(5)
        response = [doc for doc in documents]
        return response

class Town(Resource):
    def get(self, town_name):
        query = { "town": town_name }
        documents = collection.find(query).limit(5)
        response = [doc for doc in documents]
        return response

class MultipleParams(Resource):
    def get(self):
        query = {}
        if request.args.get('month'):
            query['month'] = request.args.get('month')
        if request.args.get('town'):
            query['town'] = request.args.get('town')
        documents = collection.find(query).limit(5)
        response = [doc for doc in documents]
        return response

class PriceEstimate(Resource):
    def get(self):
        query = {}
        xgb_pred = xgb.XGBRegressor()
        xgb_pred.load_model("/home/yexi/repos/flask-vue-hdb-price-app/server/xgb_reg_pred.json")
        request.args.get('floor_area_sqm')
        request.args.get('flat_type_mapped')
        request.args.get('storey_mean')
        request.args.get('lease_remain_years')
        
api.add_resource(LatestMonth, '/api/v1/month/<month>')
api.add_resource(Town, '/api/v1/town/<town_name>')
api.add_resource(MultipleParams, '/api/v1/multiple')
api.add_resource(PriceEstimate, '/api/v1/estimate')


if __name__ == '__main__':
    app.run(debug=True)
    