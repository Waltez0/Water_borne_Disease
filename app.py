import numpy as np
from flask import Flask, request, make_response
import json
import pickle
import mysql.connector as mysql
from flask_cors import cross_origin
import os
import joblib
import Hepa
import typhoid
from Hepa import classifier
from typhoid import clas
app = Flask(__name__)
# model = joblib.load("hepatitis_pickle.pkl")


@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow


@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    # print("Request:")
    # print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):
    if req.get("queryResult").get("action") != "DefaultWelcomeIntent.DefaultWelcomeIntent-yes.DefaultWelcomeIntent-yes-yes.Hepatitis-yes":
        return {}
    result = req.get("queryResult")
    parameters = result.get("parameters")
    AbdominalPain_Yes = parameters.get("AbdominalPain")
    Fever_Yes = parameters.get("Fever")
    Fatigue_Yes = parameters.get("Fatigue")
    LooseStool_Yes = parameters.get("LooseStool")
    age = parameters.get("age")
    steroids = parameters.get("steroids")
    spiders = parameters.get("spiders")
    liver_firm = parameters.get("liver_firm")
    Name = parameters.get("Patient")
    hepa_features = [age, steroids, spiders, liver_firm]
    typh_features = [AbdominalPain_Yes, Fever_Yes, Fatigue_Yes, LooseStool_Yes]

    hepatitis_features = [np.array(hepa_features)]
    typhoid_features = [np.array(typh_features)]

    intent = result.get("intent").get('displayName')

    if (intent == 'Symptoms_check'):
        prediction = clas.predict(typhoid_features)

        output = round(prediction[0], 2)

        if(prediction == 1):
            typo = 'Typhoid result negative'

        if(prediction == 0):
            typo = 'Typhoid result positive'

        predi = classifier.predict(hepatitis_features)

        output = round(predi[0], 2)

        if(predi == 1):
            hepat = 'Hepatitis result Positive'

        if(predi == 0):
            hepat = 'Hepatitis result Negative'

        if(LooseStool_Yes == 1):
            Dia = 'Diarrhea result positive'

        if(LooseStool_Yes == 0):
            Dia = 'Diarrhea result Negative'

        fulfillmentText = "Diagnosis : {} , {} , {} !".format(Dia,typo,hepat)
        # log.write_log(sessionID, "Bot Says: "+fulfillmentText)

        #con = mysql.connect (host='localhost', user='root', port=3306, password='', db='hospital')
        #cursor = con.cursor()
        #cursor.execute("INSERT INTO `PatientDetails` VALUES ('"+ Name +"','"+ AbdominalPain_Yes +"','"+ Fever_Yes +"','"+ Fatigue_Yes +"','"+ LooseStool_Yes +"','"+ age +"','"+ steroids +"','"+ spiders +"','"+ liver_firm +"','"+ hepat +"','"+ typo +"','"+ Dia +"')")
        #cursor.execute("commit");
        #con.close();
        
        return {
            "fulfillmentText": fulfillmentText
        }
    
    
    # else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 85))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')
