from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and data
model = joblib.load('calorie_model.pkl')
# API Key for CalorieNinjas
API_KEY = '9lWb65UV1kwSPJZEGoNbSQ==nUxw9QZSZ9ZTG48z'

# Function to load the nutrition data (your food dataset with ingredients and allergens)
nutrition_df = pd.read_csv('updated_food_with_calories_and_categories.csv')







# Meal Recommendation Function based on calorie intake
def get_meals_for_calories(target_calories):
    possible_meals = nutrition_df[nutrition_df['Calories'] <= target_calories]

    if possible_meals.empty:
        raise ValueError("No meals found within the calorie limit.")
    
    # Randomly select meals from the possible options
    selected_meals = possible_meals.sample(4)  # Get 4 random meals: breakfast, lunch, dinner, snack
    
    total_calories = selected_meals['Calories'].sum()

    return selected_meals, total_calories

# Generate meal plan based on user data
def generate_meal_plan(user_data, model):
    # Prepare user input for calorie prediction
    gender_encoded = 1 if user_data['gender'] == 'female' else 0
    
    # Scale user features
    user_features = np.array([[user_data['age'], user_data['height'], user_data['weight'],
                                user_data['duration'], user_data['heart_rate'], user_data['body_temp'], gender_encoded]])
    
    user_scaler = StandardScaler()
    user_scaler.fit(user_features)  # Fit on the single input for consistent shape
    user_features_scaled = user_scaler.transform(user_features)
    
    # Predict daily caloric intake
    predicted_calories = model.predict(user_features_scaled.reshape(1, -1))[0]
    
    # Generate meal plans
    meal_plan = []
    num_days = user_data['plan_duration']
    
    for day in range(1, num_days + 1):
        selected_meals, total_calories = get_meals_for_calories(predicted_calories)
        
        meal_plan.append({
            'day': day,
            'meals': selected_meals[['Food Product', 'Category', 'Calories']].to_dict(orient='records'),
            'total_calories': total_calories
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
