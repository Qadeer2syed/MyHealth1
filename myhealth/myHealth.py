from prettytable import PrettyTable
from tinydb import TinyDB, Query
from tensorflow.keras.preprocessing import image
#from keras.preprocessing import image

import typer
import numpy as np
import tensorflow as tf
from typing import List

app = typer.Typer()
model_file_path = "model/model2.keras"
#class_names = ["hamburger", "hot dog", "pizza"]

class_names = {
    0: 'macaron',
    1: 'beignet',
    2: 'cruller',
    3: 'cockle_food',
    4: 'samosa',
    5: 'tiramisu',
    6: 'tostada',
    7: 'moussaka',
    8: 'dumpling',
    9: 'sashimi',
    10: 'knish',
    11: 'croquette',
    12: 'couscous',
    13: 'porridge',
    14: 'cabbage',
    15: 'salad',
    16: 'Noodles',
    17: 'rigatoni',
    18: 'beef tartare',
    19: 'cannoli',
    20: 'foie gras',
    21: 'cupcake',
    22: 'osso buco',
    23: 'pad thai',
    24: 'poutine',
    25: 'ramen',
    26: 'sandwich',
    27: 'bibimbap',
    28: 'chicken',
    29: 'apple pie',
    30: 'risotto'
}


def get_food_name_from_user() -> str:
    food_name = input("Enter food name (type quit to quit):").lower()
    if food_name == "quit":
        exit()
    return food_name


def get_nutritional_data_for_food(food_name: str) -> List:
    db = TinyDB("data/db.json")
    Food = Query()
    results = db.search(Food.name == food_name)
    if len(results) > 0:
        return results
    # Check secondary db.
    db = TinyDB("data/secondary_db.json")
    for food in db:
        for food_id in food:
            if food_name in food[food_id]["name"]:
                results.append(food[food_id])
    if len(results) > 0:
        return results
    # No food found. Exiting program
    print(f"No nutritional data for '{food_name.title()}'")
    exit()


def print_nutritional_data(nutritional_data: dict):
    # Print food name
    food_name = nutritional_data.get("name", "UNKNOWN FOOD NAME").title()
    print(f"\n{food_name}")
    # Print serving size disclosure
    print("The following metrics are based on a 100g serving size.")
    # Print Nutrients
    table = PrettyTable()
    table.field_names = ["Nutrient", "Amount"]
    for attr in nutritional_data:
        if attr == "name":
            continue
        nutrient_name = attr.title()
        nutrient_value = round(nutritional_data[attr], 4)
        table.add_row([nutrient_name, nutrient_value])
    print(table)

def preprocess_image(image_path,target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.command()
def get_nutrition_from_photo(photo_path: str):
    """get_nutrition_from_photo takes in a path to a photo of food, and prints the nutritional information.

    Args:
        photo_path (str): a path to a photo of food.
    """
    # Running the model on the image that was provided.
    #img = tf.keras.utils.load_img(photo_path)
    image_path = photo_path
    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create a batch
    target_size = (224, 224)
    preprocessed_image = preprocess_image(image_path,target_size)
    model = tf.keras.models.load_model(model_file_path)
    
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    #highest_score_highest_idx = np.argmax(prediction)

    if prediction[0,predicted_class] < 0.8:
        print("Mmmmm... I don't recognize that food by the picture.")
        food_name = get_food_name_from_user()
    else:
        food_name = class_names[predicted_class]
        confidence_score = round(100 * np.max(prediction), 2)
        print(f"This food is a {food_name} with a {confidence_score}% confidence.")
    nutritional_data_results = get_nutritional_data_for_food(food_name)
    for nutritional_data in nutritional_data_results:
        print_nutritional_data(nutritional_data)


if __name__ == "__main__":
    app()
