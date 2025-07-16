# Bank statement analyser
A web application that allows users to upload bank statements (CSV/PDF) and get the total credit, total debit and the net cash flows from the information. 

# Key Features
1. Upload CSV or PDF bank statements as individual files or in the form of zip files. 
2. Automatic credit/debit detection
3. AI-powered data cleaning
4. API built using FastAPI

# Installation
Open the terminal of your preferred code editor and type the following commands 
1. git clone https://github.com/arnavdesai6143/bank_statement_project.git
2. cd bank_statement_project
3. pip install -r requirements.txt


# How to run the model
1. Visit http://0.0.0.0:8000/docs for API documentation. 
2. You would see two posts: analyse_statements and analyse_pdf_statement
3. Open them and click on 'Try it out'. Upload your files in the form of a csv or an excel file or a zip of csv or excel files and run the commnands 
4. The commands might take a while to run depending on the size of the files. You should be able to see your output on the screen
5. Similarly, open the analyse_pdf_statement post and add a pdf file or a zip containing multiple pdf files and run them. 
6. You should be able to see the file name, the total credit, total debit and the total cash flows. 
