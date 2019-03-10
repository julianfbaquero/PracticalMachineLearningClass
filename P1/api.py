#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from model_price import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API',
    description='Price Prediction API: Julian Baquero, Andrés Lozano, Lucila Noriega, ')

ns = api.namespace('predict',
     description='Price Classifier')
parser = api.parser()

parser.add_argument(
    'Year',
    type=int,
    required=True,
    help='Type year',
    location='args')
parser.add_argument(
    'Mileage',
    type=float,
    required=True,
    help='Type Mileage',
    location='args')
parser.add_argument(
    'State',
    type=str,
    required=True,
    help='Select Stage',
    location='args')
parser.add_argument(
    'Make',
    type=str,
    required=True,
    help='Select Make',
    location='args')
parser.add_argument(
    'Model',
    type=str,
    required=True,
    help='Select Model',
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
