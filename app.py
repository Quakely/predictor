from flask import Flask, jsonify

from model.earthquake_predictor import EarthquakePredictor

app = Flask(__name__)


@app.route('/', methods=['GET'])
def get_status():
    return jsonify({
        "code": 200,
        "message": "The predictor server is up and running"
    })


@app.route('/earthquakes', methods=['GET'])
def get_earthquakes():
    predictor = EarthquakePredictor()
    predictions = predictor.predict()
    predictions_list = [
        {"timestamp": row[0], "latitude": row[1], "longitude": row[2],
         "magnitude": row[3], "depth": row[4]}
        for row in predictions
    ]

    return jsonify({
        "code": 200,
        "message": "Fetched predicted earthquakes for the next week",
        "data": predictions_list
    })


if __name__ == '__main__':
    app.run()
