import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin

app = Flask(__name__)
model = pickle.load(open('hepatitis_pickle.pkl', 'rb'))

@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):

    #sessionID=req.get('responseId')
    result = req.get("queryResult")
    #user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    age=parameters.get("age")
    steroids = parameters.get("steroids")
    spiders=parameters.get("spiders")
    liver_firm=parameters.get("liver_firm")
    int_features = [age,steroids,spiders,liver_firm]
    
    final_features = [np.array(int_features)]
	 
    intent = result.get("intent").get('displayName')
    
    if (intent=='Hepatitis_Symptoms'):
        prediction = classifier.predict(final_features)
    
        output = round(prediction[0], 2)
    
    	
         
        if(output==1):
            Hepatitis = 'you do not have Hepatitis'
        
        if(output==2):
            Hepatitis = 'you have Hepatitis'
       
        fulfillmentText= "It seems that..  {} !".format(Hepatitis)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    #else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)

if __name__ == '__main__':
    app.run()
#if __name__ == '__main__':
#    port = int(os.getenv('PORT', 5000))
#    print("Starting app on port %d" % port)
#    app.run(debug=False, port=port, host='0.0.0.0')