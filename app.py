import streamlit as st
import youtube_download
import main

st.title('VIDEO MOMENT RETRIEVAL')
st.write('You can pass a youtube video url and use a query to find the top three related moments in the video')
url = st.text_input('Enter youtube url')
query = st.text_input('Enter your query')
if st.button('Find'):
    if url:
        youtube_download.download(url)
        frames = main.find(query)
        st.image(frames)

