"""
This script is an Azure Function, which is activated when a file(in .json format) is uploaded
as a blob in a specified container allocated in the cloud, it manages the blob and upload another
blob named 'summary_report.txt' that contains a summary of the uploaded file in the same container
"""


#import the azure.functions module
import azure.functions as func
#to log messages during the execution
import logging

#Library to import the json
import json
#Lbrary that will allow us to clean symbols(not relevant)
import nltk
#Library for the stopwords
from nltk.corpus import stopwords

#Just in case we need to download stopwords
nltk.download('stopwords')

#Library for regular expression
import re

#Libraries for use the resource of OpenAI that will do the summary
from openai import AzureOpenAI
#To manage environment variables
import os


#To manage blobs
from azure.storage.blob import BlobServiceClient


def filter_of_confidence(doc, threshold):
    #I create function thet removes the part of the data that we feel that
    #hasn't been extracted in a right way with the OCR, because of its confidence

    #PRE: doc is one of the components(dictionary) of the main list (composed of many documents) and 
    #threshold the minimum "confidence" of the "words" we want to mantain
    #POST: returns the document in same format with the "words" that have the key of "confidence" more than threshold, 
    #deleting the keys "selection marks", "width", "height" and "unit" from the "content" key of each page because aren't relevant

   #Initialize a new dictionary for the filtered document, maintaining the format
    filtered_doc = {
        'doc_id': doc['doc_id'],  #Preserve doc_id
        'content': []
    }
    
    #Iterate over the pages in the document
    for page in doc['content']:
        #Initialize a new page structure with page_number
        filtered_page = {
            'page_number': page['page_number'],  #Preserve page_number
            'words': []  #This will hold filtered words
        }

        #Filter words based on confidence
        for word in page['words']:
            if word.get('confidence', 0) > threshold:
                #Preserve the whole word structure (content and confidence together)
                filtered_page['words'].append({'content': word['content'],'confidence': word['confidence']})
        
        #Add the filtered page to the content list
        filtered_doc['content'].append(filtered_page)
    
    return filtered_doc

def filter_data_by_confidence(data, threshold):
    #PRE: data is the whole data(composed of dictionaries, each document is one dictionary) we want to filter by confidence
    #POST: returns the data in same format but filtered by confidence
    return [filter_of_confidence(doc, threshold) for doc in data]



def clean_words(words):
    #PRE: words is a list of dictionaries, which every dictionary has at least a 'content' key
    #POST: return a list of strings, without stopwords(words with a stopword in the key 'content') and without
    #words that are not relevant for the summarizer

    stop_words=set(stopwords.words('english'))
    pattern=re.compile(r'^[a-zA-Z]{3,}$')
    return [word['content'] for word in words if (word['content'].lower() not in stop_words) and (pattern.match(word['content'].lower()))]

def filter_of_stopwords(data):
    #PRE: data has a json format and has been filtered by confidence
    #POST: returns data in json format, but now 'words' is returned as a list of 
    #strings(with all the words of each page filtered)

    filtered_by_stopwords = []
    
    #Iterate over each document
    for doc in data:
        cleaned_content = []
        
        #Iterate over each page in the document
        for page in doc['content']:
            #Clean the words (remove stopwords)
            cleaned_words = clean_words(page['words'])
            
            #Add cleaned data to the content
            cleaned_content.append({'page_number': page['page_number'],'words': cleaned_words})
        
        #Add the cleaned document to the processed list
        filtered_by_stopwords.append({'doc_id': doc['doc_id'],'content': cleaned_content})
    
    return filtered_by_stopwords


def json_to_text_with_metadata(data):
    #We want to transform the json in raw text to be sent to the AI
    #PRE: data must be a json with the next format
    #  [{"doc_id": docNum1, 
    #    "content":[{"page_number": pageNum1,
    #                "words":["word1","workd2",...]},   //end of the page1 of the doc1
    #               {"page_number": pageNum2,
    #                "words":[...]},{...}]  //end of content of the doc1
    #   {"doc_id": docNum2,
    #    "content": ...}
    #   {...}]     //end of the data
    #POST: returns the filtered content of each document and each page 
    #in an string(but specifying document and page content)

    #With this function we want to have the data in linear format, because
    #some NPL services would process better the data in this format
   
    text_content = []

    #Go through all the documents in the data
    for document in data:
        doc_id = document['doc_id'] 
        
        #Go through all the pages on each document
        for page in document['content']:
            page_number = page['page_number'] 
            #Add the number of document and page before the words 
            text_content.append(f"\n[Document {doc_id}, Page {page_number}]\n")
            #Add the words of each page
            text_content.extend(page['words'])
    #Return the list of words into continous text
    return ' '.join(text_content) 

def cleaner_of_data(data):

    #Filter the data by confidence
    filter1=filter_data_by_confidence(data,0.8)
    #Filter the data (that has been filtered by confidence), by stopwords
    filter2=filter_of_stopwords(filter1)
    #Write the json document into continous text, specifying Document and Page numbers
    continuousText=json_to_text_with_metadata(filter2)
    
    return continuousText

def connection_to_data(myblob:func.InputStream):
    #PRE:myblob is the new blob uploaded in a certain container in the cloud
    #POST: returns the json file saved in the blob
   
    try:
        #Read the content of the blob
        blob_data=myblob.read().decode('utf-8')
        
        #Convert the content of the blob into a json
        jsonData=json.loads(blob_data)
        logging.info("Data from the blob obtained correctly")
        return jsonData

    except Exception as e:
        logging.error("Error obtaining the data from the blob")
        #At this point we will see the error in the logs of the Azure Function
        return None
    


#From this Python function, we could create another Azure Function,
#one for the Step1(cleaning) and other for the Step2(summarizing)


def extract_documents_and_pages(importantData):
    #PRE: immportantData has this format:
    #[Document 1, Page 1]
    #...
    #[Document 1, Page 2]
    #...
    #POST: returns the same information but in json format like: 
    #[{"doc_id"=1,"page_number"=1, "text": "all text of the page"},{"doc_id"=1,"page_number"=2, "text":"..."},...,{"doc_id"=2, "page_number"=1,"text": "..."}]
    #so puts each page as one object, that is in a certain document and has a certain text

    
    #With this function we want to know from where is obtaining the openAI resource the information of each point in the summary
    documents = []
    current_document = None
    current_page = None
    current_text = []

    #Regex to detect lines like '[Document X, Page Y]'
    pattern = re.compile(r"\[Document (\d+), Page (\d+)\]")

    for line in importantData.splitlines():
        match = pattern.match(line.strip())
        if match:
            #If we already have accumulated text, save the previous document's content
            if current_document and current_page and current_text:
                documents.append({
                    'doc_id': current_document,
                    'page_number': current_page,
                    'text': ' '.join(current_text)
                })

            #Start a new document/page
            current_document = match.group(1)
            current_page = match.group(2)
            current_text = []
        else:
            #Accumulate text for the current document and page
            current_text.append(line.strip())

    #Add the last document/page to the list if available
    if current_document and current_page and current_text:
        documents.append({
            'doc_id': current_document,
            'page_number': current_page,
            'text': ' '.join(current_text)
        })

    return documents

def summarize_with_openai(text_to_summarize):
    #PRE: immportantData is an array of objects, each object is a page from the whole cleaned data
    #POST: returns the final summary in a sequence of points with metadata(from which document and page it has obtained each point)
    try:
        #Get the endpoint and key from the environment
        endpoint = os.getenv("ENDPOINT_URL","https://gptsummary.openai.azure.com/")
        deployment = os.getenv("DEPLOYMENT_NAME","gpt-4-32k")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY","5c6e773f47fa41d096e338c91a3a5b1f")
        #Initialize Azure OpenAI client with key-based authentication
        client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key = subscription_key,
            api_version = "2024-05-01-preview",
        )

        completion = client.chat.completions.create(
            model=deployment,
            messages= [
            {
                "role": "system",
                "content": "Lawyer that wants to sum up a collection of documents into the most important points."
            },
            {
                "role": "user",
                "content": f"Sum up this collection of documents in 20 different important points and tell me from where you have took each concept:\n{text_to_summarize}"
            }],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        #response will be the whole answer of the gpt-4 service
        response= completion.to_json()
        data = json.loads(response)
        #We want only the summary
        summary=data["choices"][0]["message"]["content"]
        return (summary)

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        #At this point we will see the error in the Logs the Azure function
        return None
    

def validate_summary(summary):
    #PRE: summary is the supposed summary received from the OpenAI service
    #POST: returns true if the summary is divided in points like: 1. (point one text) 2. (point two text) ... N. (point N text)

        #Regular expression to validate next structure in the summary: 1. (...) 2. (...) ... n. (...)
        patron = r'^\d+\.\s.*$'
        
        lines=summary.strip().split('\n')
        for line in lines:
            #Sometimes gpt-4 returns me the summary with empty lines between the points
            if line.strip() == '':
                continue
            if not re.match(patron,line.strip()):
                return False
        
        return True
    

    
#We could create an extra Azure function just to decide what to do with the result,
#in my case I have choosen creating a blob in the same container that THE INITIAL DATA WAS UPLOADED, which
#produce this Azure Function to start

def upload_to_blob(container_name, blob_name, text):
    #PRE: container_name is a container resource created in the Azure cloud, blob_name is a string which indicates the blob where
    #the final summary will be located, text is the final summary
    #POST: returns a message(string) of OK(after creating the blob with text in the container) if the summary is well uploaded to the container(in a blob)
    #and None in other case
    try:
        #Get the connection string from environment variables
        connection_string = os.getenv("Connection_STORAGE")

        if not connection_string:
            raise ValueError("Azure Storage connection string is not configured properly.")

        #Create a BlobServiceClient object to interact with the Blob service
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        #Get the reference to the container
        container_client = blob_service_client.get_container_client(container_name)


        #Create a reference to upload the text in the blob of name
        blob_client = container_client.get_blob_client(blob_name)

        #Upload text to the blob, if it was another blob with the same name, rewrite the content of it 
        #and if it wasn't any blob with that name we create it
        blob_client.upload_blob(text.encode('utf-8'), blob_type="BlockBlob", overwrite=True)  

        
        return f"Summary uploaded successfully to {blob_name} in the container: {container_name}"

    except Exception as e:
        logging.error(f"Error uploading summary to blob: {e}")
        #At this point we will see the error in the logs of the Azure Function
        return None

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="container/{name}",connection="Connection_STORAGE") 

#blob_tigger activates the function when a new blob is uploaded in the specified container located in the cloud(resource)

#arg_name is the variable we are going to use in the function to refer the blob

#path is the route of the blob that will activate the function

#connection is the configuration(defined in local.settings.json) that we are going to use to 
#access to the container in the cloud, we can se in the local settings thet is the Connection String of the 
#resource located in the cloud

def blob_trigger(myblob: func.InputStream):
    
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}")
    
    if "summary_report.txt" in myblob.name:
        #We don't want to repeat the activation of the function uploading the final summary(another blob) in the container
        logging.info(f"Skipping processing for blob: {myblob}")
        return

    #Gets the data of the blob that has been uploaded in json format
    json_data=connection_to_data(myblob)
    if json_data:
        #Clean data
        importantData=cleaner_of_data(json_data)
        #Manage the cleaned data for being in the best format to send it to the OpenAI resource
        data_for_gpt4=extract_documents_and_pages(importantData)
        #Send the data to the OpenAI resource
        SUMMARY=summarize_with_openai(data_for_gpt4)
        
        if SUMMARY:
            logging.info(f"Final summary of the data updated in the container:\n {SUMMARY}")
            if validate_summary(SUMMARY):
            
                #Upload the final summary in a blob named "summary_report.txt"
                finalBlob="summary_report.txt"
                logging.info(upload_to_blob('container',finalBlob,SUMMARY))
            else:
                logging.error("The summary is not schematic and standardised")
        else:
            logging.error('Error summarizing with the OpenAI service')
    else:
        logging.error('Error accessing to the initial data')
    
  
    
    