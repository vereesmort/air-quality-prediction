
# Set the environment variables from the file <root_dir>/.env
from mlfs import config
settings = config.HopsworksSettings()

# %% [markdown]
# <span style="font-width:bold; font-size: 3rem; color:#333;">- Part 01: Feature Backfill for Air Quality Data</span>
# 

# %% [markdown]
# ### <span style='color:#ff5f27'> 📝 Imports

# %%
import datetime
import requests
import pandas as pd
import hopsworks
from mlfs.airquality import util
import datetime
from pathlib import Path
import json
import re
import os
import warnings
import sys
warnings.filterwarnings("ignore")

# %% [markdown]
# ---

# %%
project = hopsworks.login()

# %%
today = datetime.date.today()

root_dir = os.getcwd()

csv_file=f"{root_dir}/data/helsinki-air-quality.csv"
util.check_file_path(csv_file)

# taken from ~/.env. You can also replace settings.AQICN_API_KEY with the api key value as a string "...."
if settings.AQICN_API_KEY is None:
    print("You need to set AQICN_API_KEY either in this cell or in ~/.env")
    sys.exit(1)

AQICN_API_KEY = settings.AQICN_API_KEY.get_secret_value() 
aqicn_url = settings.AQICN_URL
country = settings.AQICN_COUNTRY
city = settings.AQICN_CITY
street = settings.AQICN_STREET
# If this API call fails (it fails in a github action), then set longitude and latitude explicitly - comment out next line
# latitude, longitude = util.get_city_coordinates(city)
# Uncomment this if API call to get longitude and latitude
latitude = "60.1733244"
longitude = "24.9410248"


secrets = hopsworks.get_secrets_api()
# Replace any existing secret with the new value
secret = secrets.get_secret("AQICN_API_KEY")
if secret is not None:
    secret.delete()
    print("Replacing existing AQICN_API_KEY")

secrets.create_secret("AQICN_API_KEY", AQICN_API_KEY)

# %% [markdown]
# ### Validate that the AQICN_API_KEY you added earlier works
# 
# The cell below should print out something like:
# 
# ![image.png](attachment:832cc3e9-876c-450f-99d3-cc97abb55b13.png)

# %%
try:
    aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, AQICN_API_KEY)
except hopsworks.RestAPIError:
    print("It looks like the AQICN_API_KEY doesn't work for your sensor. Is the API key correct? Is the sensor URL correct?")

aq_today_df.head()

# %% [markdown]
# ## Read your CSV file into a DataFrame
# 
# The cell below will read up historical air quality data as a CSV file into a Pandas DataFrame

# %%
df = pd.read_csv(csv_file,  parse_dates=['date'], skipinitialspace=True)
df

# %% [markdown]
# ## Data cleaning
# 
# 
# ### Rename columns if needed and drop unneccessary columns
# 
# We want to have a DataFrame with 2 columns - `date` and `pm25` after this cell below:

# %% [markdown]
# ## Check the data types for the columns in your DataFrame
# 
#  * `date` should be of type   datetime64[ns] 
#  * `pm25` should be of type float64

# %%
df_aq = df[['date', 'pm25']]
df_aq['pm25'] = df_aq['pm25'].astype('float32')

df_aq

# %%
# Cast the pm25 column to be a float32 data type
df_aq.info()

# %% [markdown]
# ## Drop any rows with missing data
# It will make the model training easier if there is no missing data in the rows, so we drop any rows with missing data.

# %%
df_aq.dropna(inplace=True)
df_aq

# %% [markdown]
# ## Add country, city, street, url to the DataFrame
# 
# Your CSV file may have many other air quality measurement columns. We will only work with the `pm25` column.
# 
# We add the columns for the country, city, and street names that you changed for your Air Quality sensor.
# 
# We also want to make sure the `pm25` column is a float32 data type.

# %%
# Your sensor may have columns we won't use, so only keep the date and pm25 columns
# If the column names in your DataFrame are different, rename your columns to `date` and `pm25`
df_aq['country']=country
df_aq['city']=city
df_aq['street']=street
df_aq['url']=aqicn_url
df_aq

# %%
df_aq.info()

# %% [markdown]
# ---

# %% [markdown]
# ## <span style='color:#ff5f27'> 🌦 Loading Weather Data from [Open Meteo](https://open-meteo.com/en/docs)

# %% [markdown]
# ## Download the Historical Weather Data
# 
# https://open-meteo.com/en/docs/historical-weather-api#hourly=&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant
# 
# We will download the historical weather data for your `city` by first extracting the earliest date from your DataFrame containing the historical air quality measurements.
# 
# We will download all daily historical weather data measurements for your `city` from the earliest date in your air quality measurement DataFrame. It doesn't matter if there are missing days of air quality measurements. We can store all of the daily weather measurements, and when we build our training dataset, we will join up the air quality measurements for a given day to its weather features for that day. 
# 
# The weather features we will download are:
# 
#  * `temperature (average over the day)`
#  * `precipitation (the total over the day)`
#  * `wind speed (average over the day)`
#  * `wind direction (the most dominant direction over the day)`
# 

# %%
earliest_aq_date = pd.Series.min(df_aq['date'])
earliest_aq_date = earliest_aq_date.strftime('%Y-%m-%d')
earliest_aq_date

weather_df = util.get_historical_weather(city, earliest_aq_date, str(today), latitude, longitude)

# %%
weather_df.info()

# %% [markdown]
# ## Define Data Validation Rules
# 
# We will validate the air quality measurements (`pm25` values) before we write them to Hopsworks.
# 
# We define a data validation rule (an expectation in Great Expectations) that ensures that `pm25` values are not negative or above the max value available by the sensor.
# 
# We will attach this expectation to the air quality feature group, so that we validate the `pm25` data every time we write a DataFrame to the feature group. We want to prevent garbage-in, garbage-out.

# %%
import great_expectations as ge
aq_expectation_suite = ge.core.ExpectationSuite(
    expectation_suite_name="aq_expectation_suite"
)

aq_expectation_suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_min_to_be_between",
        kwargs={
            "column":"pm25",
            "min_value":-0.1,
            "max_value":500.0,
            "strict_min":True
        }
    )
)

# %% [markdown]
# ## Expectations for Weather Data
# Here, we define an expectation for 2 columns in our weather DataFrame - `precipitation_sum` and `wind_speed_10m_max`, where we expect both values to be greater than zero, but less than 1000.

# %%
import great_expectations as ge
weather_expectation_suite = ge.core.ExpectationSuite(
    expectation_suite_name="weather_expectation_suite"
)

def expect_greater_than_zero(col):
    weather_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column":col,
                "min_value":-0.1,
                "max_value":1000.0,
                "strict_min":True
            }
        )
    )
expect_greater_than_zero("precipitation_sum")
expect_greater_than_zero("wind_speed_10m_max")

# %% [markdown]
# ---

# %% [markdown]
# ### Connect to Hopsworks and save the sensor country, city, street names as a secret

# %%
fs = project.get_feature_store() 

# %% [markdown]
# #### Save country, city, street names as a secret
# 
# These will be downloaded from Hopsworks later in the (1) daily feature pipeline and (2) the daily batch inference pipeline

# %%
dict_obj = {
    "country": country,
    "city": city,
    "street": street,
    "aqicn_url": aqicn_url,
    "latitude": latitude,
    "longitude": longitude
}

# Convert the dictionary to a JSON string
str_dict = json.dumps(dict_obj)

# Replace any existing secret with the new value
secret = secrets.get_secret("SENSOR_LOCATION_JSON")
if secret is not None:
    secret.delete()
    print("Replacing existing SENSOR_LOCATION_JSON")

secrets.create_secret("SENSOR_LOCATION_JSON", str_dict)

# %% [markdown]
# ### Create the Feature Groups and insert the DataFrames in them

# %% [markdown]
# ### <span style='color:#ff5f27'> 🌫 Air Quality Data
#     
#  1. Provide a name, description, and version for the feature group.
#  2. Define the `primary_key`: we have to select which columns uniquely identify each row in the DataFrame - by providing them as the `primary_key`. Here, each air quality sensor measurement is uniquely identified by `country`, `street`, and  `date`.
#  3. Define the `event_time`: We also define which column stores the timestamp or date for the row - `date`.
#  4. Attach any `expectation_suite` containing data validation rules

# %%
air_quality_fg = fs.get_or_create_feature_group(
    name='air_quality',
    description='Air Quality characteristics of each day',
    version=1,
    primary_key=['country','city', 'street'],
    event_time="date",
    expectation_suite=aq_expectation_suite
)

# %% [markdown]
# #### Insert the DataFrame into the Feature Group

# %%
air_quality_fg.insert(df_aq)

# %% [markdown]
# #### Enter a description for each feature in the Feature Group

# %%
air_quality_fg.update_feature_description("date", "Date of measurement of air quality")
air_quality_fg.update_feature_description("country", "Country where the air quality was measured (sometimes a city in acqcn.org)")
air_quality_fg.update_feature_description("city", "City where the air quality was measured")
air_quality_fg.update_feature_description("street", "Street in the city where the air quality was measured")
air_quality_fg.update_feature_description("pm25", "Particles less than 2.5 micrometers in diameter (fine particles) pose health risk")

# %% [markdown]
# ### <span style='color:#ff5f27'> 🌦 Weather Data
#     
#  1. Provide a name, description, and version for the feature group.
#  2. Define the `primary_key`: we have to select which columns uniquely identify each row in the DataFrame - by providing them as the `primary_key`. Here, each weather measurement is uniquely identified by `city` and  `date`.
#  3. Define the `event_time`: We also define which column stores the timestamp or date for the row - `date`.
#  4. Attach any `expectation_suite` containing data validation rules

# %%
# Get or create feature group 
weather_fg = fs.get_or_create_feature_group(
    name='weather',
    description='Weather characteristics of each day',
    version=1,
    primary_key=['city'],
    event_time="date",
    expectation_suite=weather_expectation_suite
) 

# %% [markdown]
# #### Insert the DataFrame into the Feature Group

# %%
# Insert data
weather_fg.insert(weather_df, wait=True)

# %% [markdown]
# #### Enter a description for each feature in the Feature Group

# %%
weather_fg.update_feature_description("date", "Date of measurement of weather")
weather_fg.update_feature_description("city", "City where weather is measured/forecast for")
weather_fg.update_feature_description("temperature_2m_mean", "Temperature in Celsius")
weather_fg.update_feature_description("precipitation_sum", "Precipitation (rain/snow) in mm")
weather_fg.update_feature_description("wind_speed_10m_max", "Wind speed at 10m abouve ground")
weather_fg.update_feature_description("wind_direction_10m_dominant", "Dominant Wind direction over the dayd")

# %% [markdown]
# ##  **Next:** Daily Feature Pipeline 
# 

# %% [markdown]
# ---


