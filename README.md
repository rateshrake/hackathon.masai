# masaihackthon

# For Video Presentation <br>
https://drive.google.com/file/d/1v0c36TtF8vpndTHJPf-t_ACUC8U1dDUW/view?usp=sharing
<br>

## One
  open folder with vs code
  ->go to terminal 
  then
 for installing all installation dependencies <br>
->pip install -r requirements.txt 

### Two
now run the following command , it will open the webpage automatically<br>
streamlit run app/dashboard.py <br>

now upload the test file it response to it

## Sample File for Testing
for uploading a sample file , it is alreay have a sample file in  <data> folder with in the directory ,
use test.csv for sample test the model!


____________________________________________________<br> 
about the model used 
we choosen xgboost from all the model , because it finds best output and % accuracy in context to the given requirment(binary classification)
we use labeled encoding for categorical data  to make model more accurate and we
analysis the relation patterns ,and then train and test the model using AUC ROC method and plot those thing in the streamlit!
