from flask import Flask, request, render_template
# from requests import request
import numpy as np
import pandas as pd
# from src.logger import lj 
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            fuel_type= request.form.get('fuel_type'),
            gear_type= request.form.get('gear_type'),
            Make= request.form.get('Make').title(),
            Year_of_manufacture= int(request.form.get('Year_of_manufacture')),
            Condition= request.form.get('Condition'),
            Mileage= float(request.form.get('Mileage')),
            Engine_size= float(request.form.get('Engine_size'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predictdat = PredictPipeline()
        results = predictdat.predict(pred_df)
        car_makes = ['Mercedes-Benz', 'BMW', 'Toyota', 'Chevrolet', 'Mini', 'Ford',
       'Lexus', 'Hyundai', 'Peugeot', 'Acura', 'Land Rover', 'Honda',
       'Infiniti', 'Mitsubishi', 'Dodge', 'Cadillac', 'Kia', 'Lincoln',
       'SsangYong', 'Nissan', 'Brabus', 'Renault', 'Geely', 'Jeep',
       'Mazda', 'Volkswagen', 'Pontiac', 'GMC', 'Chrysler', 'JAC',
       'Volvo', 'Audi', 'Subaru', 'Porsche', 'Changan', 'Suzuki',
       'Jaguar', 'Scion', 'Skoda', 'Opel', 'RAM', 'Rover', 'Seat']
        # return render_template("your_template.html", car_makes=car_makes)

        return render_template('home.html',car_makes=car_makes,results=round(int(results[0]), 2))
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)