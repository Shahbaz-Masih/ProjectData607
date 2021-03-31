import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

lrModel = pickle.load(open('lrModel.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictOutput',methods=['POST'])
def Outpredictput():
    '''
    For rendering results on HTML GUI
    Reading the entered parameters and saving them in a list
    '''
    # Index(['Age', 'Price', 'Property_Type_D', 'Property_Type_F', 'Property_Type_O',
    #        'Property_Type_S', 'Property_Type_T', 'Duration_F', 'Duration_L',
    #        'PPDCategory_Type_A', 'PPDCategory_Type_B'],
    #       dtype='object')


    Property_Type_D = 0
    Property_Type_F = 0
    Property_Type_O = 0
    Property_Type_S = 0
    Property_Type_T = 0
    Duration_F = 0
    Duration_L = 0
    PPDCategory_Type_A = 0
    PPDCategory_Type_B = 0

    Price = request.form["Price"]
    property_type = request.form["property_type"]
    duration = request.form["duration"]
    ppd = request.form["ppd"]

    if property_type == "D":
        Property_Type_D = 1
        Property_Type_F = 0
        Property_Type_O = 0
        Property_Type_S = 0
        Property_Type_T = 0
    if property_type == "F":
        Property_Type_D = 0
        Property_Type_F = 1
        Property_Type_O = 0
        Property_Type_S = 0
        Property_Type_T = 0
    if property_type == "O":
        Property_Type_D = 0
        Property_Type_F = 0
        Property_Type_O = 1
        Property_Type_S = 0
        Property_Type_T = 0
    if property_type == "S":
        Property_Type_D = 0
        Property_Type_F = 0
        Property_Type_O = 0
        Property_Type_S = 1
        Property_Type_T = 0
    if property_type == "T":
        Property_Type_D = 0
        Property_Type_F = 0
        Property_Type_O = 0
        Property_Type_S = 0
        Property_Type_T = 1
    if duration == "F":
        Duration_F = 1
        Duration_L = 0
    if duration == "L":
        Duration_F = 0
        Duration_L = 1
    if ppd == "A":
        PPDCategory_Type_A = 1
        PPDCategory_Type_B = 0
    if ppd == "B":
        PPDCategory_Type_A = 0
        PPDCategory_Type_B = 1
    parameters = [Price, Property_Type_D, Property_Type_F, Property_Type_O,
           Property_Type_S, Property_Type_T, Duration_F, Duration_L,
           PPDCategory_Type_A, PPDCategory_Type_B]
# changing the input to float
    float_features = [float(x) for x in parameters]
# changing the list to numpy array
    final_features = [np.array(float_features)]
# Taking the input of the model type and selecting the right model for predicting power output
#     modeltype = request.form["models"]
#
#     if modeltype == "Linear":
    prediction = lrModel.predict(final_features)
    output = round(prediction[0], 2)
    Age = "New"
    if output == 1:
        Age = "Old"
    else:
        Age = "New"

    return render_template('index.html', prediction_text='The Age of the property is predicted to be {}'.format(Age))



    # if modeltype == "DecisionTree":
    #     prediction = dtModel.predict(final_features)
    #     output = round(prediction[0], 2)
    #     return render_template('index.html', prediction_text='With Decision Tree Regressor, Plant Power Output is {} MW'.format(output))
    #
    # if modeltype == "RandomForest":
    #     prediction = rfModel.predict(final_features)
    #     output = round(prediction[0], 2)
    #     return render_template('index.html', prediction_text='With Random Forest Regressor, Plant Power Output is {} MW'.format(output))


if __name__ == "__main__":
    app.run(debug=True)