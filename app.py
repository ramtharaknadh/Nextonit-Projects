import streamlit as st
import pickle

scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def classify(num):
    if num == 1:
        return 'Rainfall'
    else:
        return 'No Rainfall'
    
def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Rainfall Dectection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input boxes for user to enter values
    Temperature = st.text_input('Enter Temperature (°C)', value="0.0")
    Humidity = st.text_input('Enter Humidity (%)', value="0.0")
    Windspeed = st.text_input('Enter Wind Speed (km/h)', value="0.0")
    Pressure = st.text_input('Enter Pressure (hPa)', value="0.0")

    # Ensure inputs are numeric
    try:
        Temperature = float(Temperature)
        Humidity = float(Humidity)
        Windspeed = float(Windspeed)
        Pressure = float(Pressure)
    except ValueError:
        st.error("Please enter valid numeric values.")
        return
    
    inputs=[[Temperature,Humidity,Windspeed,Pressure]]

    scaler_inputs = scaler.transform(inputs)

    prediction = model.predict(scaler_inputs)

    if st.button('Classify'):
        st.success(classify(prediction[0]))

if __name__ == '__main__':
    main()