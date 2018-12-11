import urllib.request
import json
from sklearn.neighbors import KNeighborsClassifier

"""
This is a simple weather prediction app.
It uses the OpenWeatherMap API to get weather data from various different cities and stores it in two arrays:
one with the main numerical data and another containing the string of labels.

This program uses the sklearn library to create a KNearestNeighbor neighbors algorithm.
"""

weather_data = []
weather_labels = []

# Write your API key here.
api_key = ""


def get_weather_data(lat, lon):

    weather_api = urllib.request.urlopen("https://api.openweathermap.org/data/2.5/find?lat="+lat+"&lon="+lon+"&cnt=10&appid="+api_key).read()
    weather_file = json.loads(weather_api)

    for weather_data_point in weather_file["list"]:
        temp = weather_data_point["main"]["temp"]
        pressure = weather_data_point["main"]["pressure"]
        humidity = weather_data_point["main"]["humidity"]
        wind_speed = weather_data_point["wind"]["speed"]
        wind_deg = weather_data_point["wind"]["deg"]
        clouds = weather_data_point["clouds"]["all"]
        weather_type = weather_data_point["weather"][0]["main"]

        weather_data.append([temp, pressure, humidity, wind_speed, wind_deg, clouds])
        weather_labels.append(weather_type)


def predict_weather(city_name, classifier):
    weather_api = urllib.request.urlopen("http://api.openweathermap.org/data/2.5/weather?q=" + city_name + "&appid=" + api_key).read()
    weather = json.loads(weather_api)

    temp = weather["main"]["temp"]
    pressure = weather["main"]["pressure"]
    humidity = weather["main"]["humidity"]
    wind_speed = weather["wind"]["speed"]
    wind_deg = weather["wind"]["deg"]
    clouds = weather["clouds"]["all"]
    weather_name = weather["weather"][0]["main"]

    this_weather = [temp, pressure, humidity, wind_speed, wind_deg, clouds]
    return {"Prediction: " : classifier.predict([this_weather])[0], "Actual: " : weather_name}


# Get data from various cities
get_weather_data("50.5", "0.2")
get_weather_data("56", "3")
get_weather_data("43", "5")
for i in range(10):
    get_weather_data("46", str(i*5))

# Setup the KNeighborsClassifier
AI_machine = KNeighborsClassifier(n_neighbors=5)
AI_machine.fit(weather_data, weather_labels)

# Print predictions for a given city along with a set of the different labels in the data set.
# This allows to verify, if the prediction is incorrect, weather it is due to an inaccurate
# prediction or the actual label missing in the classifier.
print(list(set(weather_labels)))
print(predict_weather("Brussels", AI_machine))