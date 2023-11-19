import streamlit as st

def settingFooter():
  hide_streamlit_style = """
              <style>
              #MainMenu {visibility: hidden;}
              footer {visibility: hidden;}
              </style>
              """
  st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

  # Add custom information at the bottom of the page with fixed positioning
  st.markdown(
      """
      <style>
      .custom-footer {
          position: fixed;
          bottom: 0;
          left: 0;
          width: 100%;
          background-color: #f5f5f5;
      }
      .custom-footer p {
          text-align: center;
          font-size: 12px;
          color: #555;
          padding: 5px;
          margin: 0;
      }
      </style>
      <div class="custom-footer">
          <p>Design by Phan Ben (潘班) ID: P76127051</p>
      </div>
      """,
      unsafe_allow_html=True
  )