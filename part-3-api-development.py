import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler


# Load the trained model
with open('./models/linear_reg.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the LabelEncoder objects
label_encoders = {}
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                       'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
for column in categorical_columns:
    with open(f'./models/label_encoder_{column}.pkl', 'rb') as f:
        label_encoders[column] = pickle.load(f)


# Scale numerical columns
scaler = StandardScaler()
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


def predict(symboling, CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, wheelbase, carlength, carwidth, carheight, curbweight, enginetype, cylindernumber, enginesize, fuelsystem, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg):
    # Create a DataFrame
    df = pd.DataFrame({
        # 'car_ID': [car_ID],
        'symboling': [symboling],
        'CarName': [CarName],
        'fueltype': [fueltype],
        'aspiration': [aspiration],
        'doornumber': [doornumber],
        'carbody': [carbody],
        'drivewheel': [drivewheel],
        'enginelocation': [enginelocation],
        'wheelbase': [wheelbase],
        'carlength': [carlength],
        'carwidth': [carwidth],
        'carheight': [carheight],
        'curbweight': [curbweight],
        'enginetype': [enginetype],
        'cylindernumber': [cylindernumber],
        'enginesize': [enginesize],
        'fuelsystem': [fuelsystem],
        'boreratio': [boreratio],
        'stroke': [stroke],
        'compressionratio': [compressionratio],
        'horsepower': [horsepower],
        'peakrpm': [peakrpm],
        'citympg': [citympg],
        'highwaympg': [highwaympg]
    })

    # Extract brand and model from CarName

    df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
    print('brand: ', df['brand'])
    df['model'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    # Define categorical and numerical columns
    numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                        'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                        'peakrpm', 'citympg', 'highwaympg']


    # Load the LabelEncoder object and transform the new data
    for column in categorical_columns:
        df[column] = label_encoders[column].transform(df[column])


    # Feature engineering
    df['power_to_weight_ratio'] = df['horsepower'] / df['curbweight']
    for column in numerical_columns:
        df[f'{column}_squared'] = df[column] ** 2
    df['log_enginesize'] = np.log(df['enginesize'] + 1)


    df[numerical_columns] = scaler.transform(df[numerical_columns])

    df.drop(['CarName'], axis=1, inplace=True)

    # Make a prediction
    prediction = model.predict(df)

    # return prediction
    return round(prediction[0], 2)



# demo = gr.Interface(
#     predict,
#     [
#         gr.Number(label="Car ID"),
#         gr.Number(label="Symboling"),
#         gr.Textbox(label="Car Name"),
#         gr.Dropdown(label="Fuel Type", choices=["gas", "diesel"]),
#         gr.Dropdown(label="Aspiration", choices=["std", "turbo"]),
#         gr.Dropdown(label="Door Number", choices=["two", "four"]),
#         gr.Dropdown(label="Car Body", choices=["convertible", "hatchback", "sedan", "wagon"]),
#         gr.Dropdown(label="Drive Wheel", choices=["rwd", "fwd", "4wd"]),
#         gr.Dropdown(label="Engine Location", choices=["front", "rear"]),
#         gr.Number(label="Wheel Base"),
#         gr.Number(label="Car Length"),
#         gr.Number(label="Car Width"),
#         gr.Number(label="Car Height"),
#         gr.Number(label="Curb Weight"),
#         gr.Dropdown(label="Engine Type", choices=["dohc", "ohc", "ohcv"]),
#         gr.Dropdown(label="Cylinder Number", choices=["four", "six", "eight"]),
#         gr.Number(label="Engine Size"),
#         gr.Dropdown(label="Fuel System", choices=["mpfi", "2bbl", "4bbl"]),
#         gr.Number(label="Bore Ratio"),
#         gr.Number(label="Stroke"),
#         gr.Number(label="Compression Ratio"),
#         gr.Number(label="Horsepower"),
#         gr.Number(label="Peak RPM"),
#         gr.Number(label="City MPG"),
#         gr.Number(label="Highway MPG")
#     ],
#     "text",
#     title="Car Price Prediction",
#     description="Enter the car's features to predict its price"
# )

if __name__ == "__main__":

    title=(
    """
    <center> 

    <h1> Car Price Prediction  🛰️ </h1>
    </center>
    """
    )

    # more theme: https://huggingface.co/spaces/gradio/theme-gallery
    with gr.Blocks(theme='NoCrypt/miku') as demo:
        
        gr.HTML(title)

        with gr.Row():
            # with gr.Column(scale=1, min_width=300):
            #     gr.Label("**Car Information**")
                # car_id = gr.Number(label="Car ID", info = "Indentification Number for Each Car")
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Car Details**")
                car_name = gr.Textbox(label="Car Name", )
                fuel_type = gr.Dropdown(["gas", "diesel"], label="Fuel Type")
                symboling = gr.Number(label="Symboling", info = "Safety rating of the car, -3 to 3")
                aspiration = gr.Dropdown(["std", "turbo"], label="Aspiration", info = "Aspiration Used in the Car")
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Car Body and Drive**")
                door_number = gr.Dropdown(["two", "four"], label="Door Number", info = "Number of Doors")
                car_body = gr.Dropdown(["convertible", "hatchback", "sedan", "wagon"], label="Car Body", info = "Type of Car Body")
                drive_wheel = gr.Dropdown(["rwd", "fwd", "4wd"], label="Drive Wheel", info = "Type of Drive Wheel")
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Car Dimensions**")
                wheel_base = gr.Slider(label="Wheel Base", info = "Distance from Front to Rear Axle", minimum=86, maximum=120, step = 1)
                car_length = gr.Slider(label="Car Length", minimum=141, maximum=208, step=1)
                car_width = gr.Slider(label="Car Width", minimum=60, maximum=70, step=1)
                car_height = gr.Slider(label="Car Height", minimum=48, maximum=60, step=1)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Engine and Weight**")
                curb_weight = gr.Number(label="Curb Weight")
                engine_type = gr.Dropdown(["dohc", "ohc", "ohcv"], label="Engine Type", info = "Type of Engine")
                engine_location = gr.Dropdown(["front", "rear"], label="Engine Location", info = "Location of Engine")
                cylinder_number = gr.Dropdown(["four", "six", "eight"], label="Cylinder Number", info = "Number of Cylinders")
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Engine Performance**")
                engine_size = gr.Number(label="Engine Size", minimum=61, maximum=320, step=10)
                fuel_system = gr.Dropdown(["mpfi", "2bbl", "4bbl"], label="Fuel System")
                bore_ratio = gr.Number(label="Bore Ratio", minimum=2.5, maximum=4, step=0.5)
                stroke = gr.Number(label="Stroke", minimum=2, maximum=4, step=0.5)
                compression_ratio = gr.Number(label="Compression Ratio")
            with gr.Column(scale=1, min_width=300):
                gr.Label("**Performance and Fuel Efficiency**")
                horsepower = gr.Number(label="Horsepower", minimum=50, maximum=280, step=10)
                peak_rpm = gr.Number(label="Peak RPM", minimum=4100, maximum=6500, step=100)
                city_mpg = gr.Number(label="City MPG", minimum=13, maximum=35, step=1)
                highway_mpg = gr.Number(label="Highway MPG", minimum=16, maximum=54, step=1)
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                predict_button = gr.Button("Predict")

        output = gr.Textbox(label="Predicted Price", text_align="center")

        predict_button.click(
            fn = predict,
            inputs = [
                # car_id,
                symboling,
                car_name,
                fuel_type,
                aspiration,
                door_number,
                car_body,
                drive_wheel,
                engine_location,
                wheel_base,
                car_length,
                car_width,
                car_height,
                curb_weight,
                engine_type,
                cylinder_number,
                engine_size,
                fuel_system,
                bore_ratio,
                stroke,
                compression_ratio,
                horsepower,
                peak_rpm,
                city_mpg,
                highway_mpg     
            ], 
            outputs = output
        )



    demo.launch(server_port=8080)