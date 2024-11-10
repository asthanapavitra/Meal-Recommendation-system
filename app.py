from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and data
model = joblib.load('calorie_model.pkl')
nutrition_df = pd.read_excel('nutrition.xlsx')

# Preprocess the nutrition data if necessary
nutrition_df['calories'] = pd.to_numeric(nutrition_df['calories'], errors='coerce')
nutrition_df['protein'] = pd.to_numeric(nutrition_df['protein'], errors='coerce')
nutrition_df.fillna(0, inplace=True)

def generate_meal_plan(user_data, model):
    # Process input data for prediction
    gender_encoded = 1 if user_data['gender'] == 'female' else 0
    user_features = np.array([[user_data['age'], user_data['height'], user_data['weight'],
                               user_data['duration'], user_data['heart_rate'], user_data['body_temp'], gender_encoded]])
    
    # Scale the user features (fit a scaler only if necessary)
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    # Predict daily calorie intake
    predicted_calories = model.predict(user_features_scaled)[0]
    
    # Generate meal plan
    meal_plan = []
    for day in range(1, user_data['plan_duration'] + 1):
        # Filter meals within the calorie range
        possible_meals = nutrition_df[nutrition_df['calories'] <= predicted_calories]
        if possible_meals.empty:
            return "No meals available within the calorie limit."

        # Sample meals
        selected_breakfast = possible_meals.sample(1).iloc[0]
        selected_lunch = possible_meals.sample(1).iloc[0]
        selected_dinner = possible_meals.sample(1).iloc[0]
        selected_snack = possible_meals.sample(1).iloc[0]
        
        # Add meals to the plan
        meal_plan.append({
            'day': day,
            'breakfast': selected_breakfast,
            'lunch': selected_lunch,
            'dinner': selected_dinner,
            'snack': selected_snack,
            'total_calories': sum([selected_breakfast['calories'],
                                   selected_lunch['calories'],
                                   selected_dinner['calories'],
                                   selected_snack['calories']])
        })
    
    return meal_plan

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    # Collect input from form
    user_data = {
        'age': int(request.form['age']),
        'gender': request.form['gender'].lower(),
        'height': float(request.form['height']),
        'weight': float(request.form['weight']),
        'duration': int(request.form['duration']),
        'heart_rate': int(request.form['heart_rate']),
        'body_temp': float(request.form['body_temp']),
        'plan_duration': int(request.form['plan_duration']),
    }

    # Generate meal plan
    meal_plan = generate_meal_plan(user_data, model)
    
    # Display the plan on result page
    return render_template('result.html', meal_plan=meal_plan)

if __name__ == '__main__':
    app.run(debug=True)
