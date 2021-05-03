#! python
"""
File: twitterTrendCapture.py
Description:
  This script uses an api to get a download for trending twitter hashtags,
    then adds the data to a local database.  It can be run as a cron job or scheduled task daily.
"""
# import libraries used below
import http.client
import mysql.connector
from mysql.connector import errorcode
import pandas as pd
import json
from datetime import date
from datetime import timedelta

import requests

# Setup api details here
api_conn = http.client.HTTPSConnection("onurmatik-twitter-trends-archive-v1.p.rapidapi.com")
headers = {
    'x-rapidapi-host': "onurmatik-twitter-trends-archive-v1.p.rapidapi.com",
    'x-rapidapi-key': "8ea6d22264mshd97d2ff15c516c5p160f30jsn8a63f7d83682"
}


# Define function to get json response from rapidapi
def get_api_response(captureDate):
    api_conn.request("GET", "/download?date="+str(captureDate)+"&country=US", headers=headers)
    res = api_conn.getresponse()
    data = res.read().decode("utf-8")
    thejson = json.loads(data)
    return thejson


# Connect to the local database
try:
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost',
                                  database='rapidapi_examples')
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
        # Get date to be used in the API call
today = date.today()
daysBack = 1  # set to 1 to get a full set of yesterday's data
captureDate = today - timedelta(days=daysBack)

# Make the API call and return the response to get a link
api_response = get_api_response(captureDate)
resultsURL = api_response['url']

# Download the file specified by the URL returned
# by the API into a pandas dataframe object
df = pd.read_csv(resultsURL)

# Loop through the data from the CSV file
for index, row in df.iterrows():
    # Remove quotes and hashtag symbols from hashtags,
    # and single quotes from other fields
    thisHashTag = row[0].replace("'",'').replace('"','').replace('#','')
    thisLoc = row[1].replace("'",'')
    thisLocType = row[2].replace("'",'')
    thisTime = row[4].replace("'",'')
    thisCount = row[5] #This field is sparsely populated so we will ignore it
    thisDate = row[3]
    thisDT = row[3]+" "+thisTime+":00"
    # add record to database
    sql = ("insert into twitter_popular_tags (hashtag,location,location_type,popular_date_time,popular_date)"
            " values ('"+thisHashTag+"','"+thisLoc+"','"+thisLocType+"','"+thisDT+"','"+thisDate+"')")
    cursor = cnx.cursor()
    try:
        cursor.execute(sql)
        cnx.commit()
    except mysql.connector.DataError as e:
        print("DataError from query: "+sql)
        print(e)
    except mysql.connector.InternalError as e:
        print("InternalError from query: "+sql)
        print(e)
    except mysql.connector.IntegrityError as e:
        print("IntegrityError from query: "+sql)
        print(e)
    except mysql.connector.OperationalError as e:
        print("OperationalError from query: "+sql)
        print(e)
    except mysql.connector.NotSupportedError as e:
        print("NotSupportedError from query: "+sql)
        print(e)
    except mysql.connector.ProgrammingError as e:
        print("ProgrammingError from query: "+sql)
        print(e)
    except :
        print("Unknown error occurred from query: "+sql)
