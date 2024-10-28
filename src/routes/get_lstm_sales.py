import json
from flask import Blueprint, jsonify, request
from bson.json_util import dumps
from datetime import datetime
from ..neuran.get_LSTM_output_data import get_LSTM_output_data

def handle_get_sales(request_data):
  user_name = request_data['userName']
  
  list_of_sales = get_LSTM_output_data(user_name, request_data)
  return list_of_sales
