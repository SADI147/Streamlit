# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:36:50 2020

@author: Sourav
"""
import streamlit as st
import joblib
import pandas as pd
import os.path
import datetime
import socket as sc

#TO STORE IP ADDRESS OF DEVICE IN THE DATABASE
global hn, ip
hn = sc.gethostname()
ip = sc.gethostbyname(hn)

#TO STORE THE DATE & TIME OF THE INPUT IN THE DATABSE
global time
time = datetime.datetime.now()

#LOADING THE MODEL
model = open("models/model.pkl", "rb")
analyzer = joblib.load(model)

#GETTING MEAN AND STANDARD DEVIATION OF DATASET FOR SCALING
file = pd.read_csv('Dataset\Dataset.csv')
global mean, sd
mean = file.mean(numeric_only='float')
sd = file.std()

html_temp ="""
<style>
body {
  height: 722px;
  width: 4px;
  background-image: url('https://img.freepik.com/free-photo/3d-geometric-abstract-cuboid-wallpaper-background_1048-9891.jpg?size=626&ext=jpg');
  background-size: cover;
}
</style>
"""
    
st.markdown(html_temp, unsafe_allow_html=True)

#PREDICTING STATUS AND RETURNING THE SAME
def predict_status(mx, my, h, w):
    scaled = [[mx, my, h, w]]
    #scaled = np.reshape(scaled,(1,-1))
    result = analyzer.predict(scaled)
    return result


def main():
    st.title("Stability Analyzer App")  
    
    if(st.button('Draw')):
        paint = 'mspaint'
        os.system(paint)
    
    if(st.button('Get Values')):
        batch = r'E:\stream_rf\additional.py'
        os.system(batch+ ' >Values.txt')
        
    if st.button('Display Values'):
        path = r"E:\stream_rf\Values.txt"
        os.system(path)
        
        
        
    status = st.radio('Got values?', ("No", "Yes"))
    if status == 'Yes':      
        Mx = str(st.number_input('Enter x corordinate of center'))
        My = str(st.number_input('Enter y corordinate of center'))
        H = str(st.number_input('Enter the height'))
        W = str(st.number_input('Enter the Width'))
        if st.button('Predict'):
            st.text('You have entered')
            st.text('Center_x {}'.format(Mx.title()))
            st.text('Center_y {}'.format(My.title()))
            st.text('Height {}'.format(H.title()))
            st.text('Width {}'.format(W.title()))
            mean_x = mean[1]
            mean_y = mean[2]
            h =  mean[3]
            w = mean[4]
            Mx = (float(Mx)-mean_x)/sd[1]
            My = (float(My)-mean_y)/sd[2]
            H = (float(H)-h)/sd[3]
            W = (float(W)-w)/sd[4]
            result = predict_status(Mx, My, H, W)
            
            if(os.path.isfile('Dataset\Random.csv') == False):
                random = pd.DataFrame(columns=["Host Name", "Host IP","Date & Time", "Mx", "My", "Height", "Width","Status"])
                random = random.append({"Host Name":hn, "Host IP":ip,"Date & Time":time, "Mx":Mx, "My":My, "Height":H, "Width":W, "Status":result[0]}, 
                                             ignore_index = True)
                random.to_csv('Dataset\Random.csv', index=False)
            else:
                extra = pd.DataFrame(columns=["Host Name", "Host IP","Date & Time", "Mx", "My", "Height", "Width","Status"])
                extra = extra.append({"Host Name":hn, "Host IP":ip,"Date & Time":time, "Mx":Mx, "My":My, "Height":H, "Width":W, "Status":result[0]}, 
                                             ignore_index = True)
                with open('Dataset\Random.csv', 'a') as fd:
                    extra.to_csv(fd, header=False, index=False)
                st.text('Successfully added to database')
                
            if result[0] == 0:
                prediction = 'Unstable'
                st.warning('Image with above values is {}'.format(prediction.title()))
            else:
                prediction = 'Stable'
                st.balloons()
                st.success('Image with above values is {}'.format(prediction.title()))
    else:
        st.subheader('Kindly check input')
        
if __name__ == '__main__':
    main()