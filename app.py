import openrouteservice
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load trained AI model
model = tf.keras.models.load_model("RoadieAI.h5")

# Initialize OpenRouteService
ORS_API_KEY = "5b3ce3597851110001cf6248d55a130edd4f4676be9218174c9e2db3"
client = openrouteservice.Client(key=ORS_API_KEY)

@app.route('/safe-route', methods=['POST'])
def get_safe_route():
    data = request.json
    start = data["start"]  # Example: [longitude, latitude]
    end = data["end"]

    # Get route options from ORS
    routes = client.directions(
        coordinates=[start, end], profile="foot-walking", format="geojson"
    )

    safest_score = -1
    safest_route = None

    for route in routes["features"]:
        distance = route["properties"]["segments"][0]["distance"]
        duration = route["properties"]["segments"][0]["duration"]

        # Dummy values (replace with real-world data)
        crowd_density = np.random.uniform(0, 1)
        lighting = np.random.uniform(0, 1)
        crime_rate = np.random.uniform(0, 1)

        # Predict safety score
        input_data = np.array([[crowd_density, lighting, distance, crime_rate]])
        safety_score = model.predict(input_data)[0][0]

        if safety_score > safest_score:
            safest_score = safety_score
            safest_route = route

    return jsonify({"safest_route": safest_route, "safety_score": safest_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
