import sys
sys.path = [path for path in sys.path if not path.startswith('/workspace')]
sys.path.append("/workspace/E8IGHT/POC")

import pytz
import torch
import numpy as np
from model.model import *
from utils.util import *
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields

input_size = 12  # 12개의 입력 수치값
hidden_size = 64
num_layers = 4
output_size = 12  # 12개의 출력 수치값
pm10_weight_path = "result/weights/pm10/best_model_weights.pth"
pm25_weight_path = "result/weights/pm25/best_model_weights.pth"

app = Flask(__name__)

@app.route('/')
def main():
    return "1차 PoC : 미세먼지 예측 시뮬레이터"

@app.route('/predict', methods=['POST'])
def post():
    try:
        data = request.json
        # 넘어온 데이터가 없을경우 바로 리턴
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # 예측 횟수 지정 (10 지정시 앞으로 10시간 수치값까지 예측, 값이 없을시 기본 1시간뒤만 예측)        
        if data.get("time_len"):
            time_len = data['time_len']
        else:
            time_len = 1
        
        # pm10 데이터에 대해 예측 수행         
        if data.get('pm10'):
            pm10 = torch.Tensor(np.array(data['pm10']).reshape(-1,1,1)).T
            model_pm10 = MultiInputOutputLSTM(input_size, hidden_size, num_layers, output_size)
            model_pm10.load_state_dict(torch.load(pm10_weight_path))
            pm10_pred_lst = []
            for _ in range(time_len):
                pm10 = model_pm10(pm10).tolist()[0]
                pm10_pred_lst.append(pm10)
                pm10 = torch.tensor(pm10).view(1, 1, 12) # 다시 모델에 넣기위해 변형
        else:
            pm10_pred_lst = None

        # pm25 데이터에 대해 예측 수행
        if data.get('pm25'):
            pm25 = torch.Tensor(np.array(data['pm25']).reshape(-1,1,1)).T
            model_pm25 = MultiInputOutputLSTM(input_size, hidden_size, num_layers, output_size)
            model_pm25.load_state_dict(torch.load(pm25_weight_path))
            pm25_pred_lst = []
            for _ in range(time_len):
                pm25 = model_pm25(pm25).tolist()[0]
                pm25_pred_lst.append(pm25)
                pm25 = torch.tensor(pm25).view(1, 1, 12) # 다시 모델에 넣기위해 변형
        else:
            pm25_pred_lst = None

        # 위치값
        if data.get('fdo_id'):
            fdo_id = data['fdo_id']
        
        # util 적용 부분
        result = make_return_dataset(fdo_id=fdo_id,
                                     pm10_pred_lst=pm10_pred_lst,
                                     pm25_pred_lst=pm25_pred_lst)
        # result = dict(sorted(result.items(), key=lambda x: int(x[0].split('_')[0])))
        
        korea_timezone = pytz.timezone('Asia/Seoul')
        time = datetime.now().astimezone(korea_timezone)
        time = time.strftime("%Y%m%d_%H:%M:%S")
        with open(f"result/prediction/check.json", "w") as json_file:
            json.dump(result, json_file, indent=4)
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# api 지정
api = Api(app, version='1.0', title='1차 PoC 미세먼지 예측 시뮬레이터', description='Swagger Documents', doc="/docs")
test_api = api.namespace('predict', description='예측 API')

# request 모델 정의
request_model = test_api.model('Request', {
    'time_len': fields.Integer(description='시간 길이'),
    'pm10': fields.List(fields.Float(description='PM10 값 리스트 (길이 12 고정)')),
    'pm25': fields.List(fields.Float(description='PM2.5 값 리스트 (길이 12 고정)'))
})

# Response 모델 정의
location_model = test_api.model('location(Response의 Key)', {
    'gps_x': fields.List(fields.Float(description='GPS X 좌표 리스트 (길이 3200 고정)')),
    'gps_y': fields.List(fields.Float(description='GPS Y 좌표 리스트 (길이 3200 고정)'))
})

hour_model = test_api.model('x_hour(Response의 Key)', {
    'pm10': fields.List(fields.Float(description='PM10 값 리스트')),
    'pm25': fields.List(fields.Float(description='PM2.5 값 리스트'))
})

response_model = test_api.model('Response', {
    'location': fields.Nested(location_model, description='위치 정보'),
    '1_hour': fields.Nested(hour_model, description='1시간 예측 데이터'),
    '2_hour': fields.Nested(hour_model, description='2시간 예측 데이터'),
})

@test_api.route('/')
class Test(Resource):
    @test_api.expect(request_model, validate=True)
    @test_api.marshal_with(response_model)
    def post(self):
        response_data = {
            "location": {
                "gps_x": [126.162165, 126.90060284810126, 126.91007, '...', 126.92007],
                "gps_y": [33.2277489, 33.539831, 33.539831, '...', 33.439531]
            },
            "1_hour": {
                "pm10": [0.0, 43.24156234123, 0.0, '...'],
                "pm25": [0.0, 22.42109683618649, 0.0]
            },
            "2_hour": {
                "pm10": [0.0, 43.24156234123, 0.0],
                "pm25": [0.0, 22.42109683618649, 0.0]
            }
        }
        return response_data, 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)