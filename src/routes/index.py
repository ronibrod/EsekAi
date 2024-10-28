import json
from flask import Blueprint, jsonify, request
from .get_lstm_sales import handle_get_sales

routes = Blueprint('routes', __name__)

@routes.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"}), 200

@routes.route('/getLstmSales', methods=['GET'])
def get_sales():
    request_data = json.loads(request.args.to_dict()['query'])
    response = handle_get_sales(request_data)
    return jsonify(response)
