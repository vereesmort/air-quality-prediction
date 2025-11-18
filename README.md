# Air Quality Prediction: Feature, Training, Batch Inference Pipelines

# Helsinki Air Quality 

In this project we will analyze the air quality in different areas of the city of Helsinki using a Machine Learning approach to predict future daily values and serve as an early alarm system to the population interested and affected by polluted air.  

## Structure

![structure of the project in a kanban format](https://i.imgur.com/3b2ZSG8.png)

The project is composed of independent pipelines that transform or process the data, they are independent as they have different tasks and deliver the results from the operations to a feature store for other pipelines to use, in our case we use [Hopsworks](https://www.hopsworks.ai/). The service lets us also upload and deploy a model to run the inference pipeline on it. 

Data is gathered from two main sources these being: [Open-Meteo](https://open-meteo.com/) for past and future weather data given a city or a point with latitude/longitude coordinates using their API and historic data and [AQICN](https://aqicn.org/here/) a repository of air quality sensors distributed worldwide that serves data from 2014 and provides a free API to query new data. 

We use 5 sensors from AQICN situated on different areas of the city to train our model, we consider this information enough and in some cases, with sensors under the same conditions and environment, redundant as the variance between the data read from them is not significant. 

### Pipelines
Next are described the three pipelines of the application.

**Features pipeline** 
This pipeline receives as an input the data from the external services, Open-Meteo and AQICN, process them and uploads it to our feature store on the cloud. As we want predictions for the next days this pipeline is to be run daily as a Github action. 

**Training pipeline**
In the training pipeline we receive as an input the datasets stored in Hopsworks joined together under one View, split them in training and test sets and train our model, a XGBoost tree. The output is a fitted model that is uploaded to Hopsworks.

**Inference pipeline** 
This pipeline is the base for our predictions, as an input we get the newest data on weather from our feature store, pass it to our model and get the values of air quality for the next days. The output is then saved in a "predictions" feature group in Hopsworks. 
  

## Feature groups

The feature groups used for our model and stored on Hopsworks are the following: 

**Weather**

| Name | Type | Description |
| :--- | :--- | :--- |
| `date` | timestamp | Date of measurement |
| `temperature_2m_mean` | float | Mean temperature in Celsius at 2m |
| `precipitation_sum` | float | Total precipitation (rain/snow) in mm |
| `wind_speed_10m_max` | float | Maximum wind speed at 10m above ground |
| `wind_direction_10m_dominant` | float | Dominant wind direction over the day |
| `rain_sum` | float | Sum of rain in mm |
| `city` | string | City where weather is measured |


**Air Quality**

| Name | Type | Description |
| :--- | :--- | :--- |
| `date` | timestamp | Date of measurement |
| `pm25` | float | PM2.5: Particles < 2.5 micrometers in diameter |
| `street` | string | Street of measurement (**PK**) |
| `url` | string | Source URL |
| `country` | string | Country of measurement (**PK**) |
| `city` | string | City of measurement (**PK**) |
| `pm25_lag_1d` | float | PM2.5 value lagged by 1 day |
| `pm25_lag_2d` | float | PM2.5 value lagged by 2 days |
| `pm25_lag_3d` | float | PM2.5 value lagged by 3 days |

**AQ Predictions**

| Name | Type | Description |
| :--- | :--- | :--- |
| `date` | timestamp | Date of the forecast (**PK**) |
| `temperature_2m_mean` | float | Mean temperature on the forecast date |
| `precipitation_sum` | float | Total precipitation on the forecast date |
| `wind_speed_10m_max` | float | Maximum wind speed on the forecast date |
| `wind_direction_10m_dominant` | float | Dominant wind direction on the forecast date |
| `rain_sum` | float | Sum of rain on the forecast date |
| `city` | string | City of the forecast (**PK**) |
| `pm25_lag_1d` | double | PM2.5 value from 1 day prior |
| `pm25_lag_2d` | double | PM25 value from 2 days prior |
| `pm25_lag_3d` | double | PM25 value from 3 days prior |
| `predicted_pm25` | float | The model's predicted PM2.5 value |
| `street` | string | Street of the forecast (**PK**) |
| `country` | string | Country of the forecast |
| `days_before_forecast_day` | bigint | Identifier for the training window (**PK**) |


## Model

The resulting model is able to predict the Air Quality on five areas of the city, corresponding to the 5 sensors used for training and measuring, for the next days with a **R²** of **0.3886** and **MSE** of **74.28**. 

![enter image description here](https://i.imgur.com/Qm7gJg1.png)

![enter image description here](https://i.imgur.com/eeeZjYq.png)