"""
This script is an Azure Function, which is activated when a file(in .json format) is uploaded
as a blob in a specified container allocated in the cloud, it manages the blob and upload another
blob named 'summary_report.json' that contains a summary of the uploaded file in the same container
"""

#import the azure.functions module
import azure.functions as func
#to log messages during the execution, monitoring all the process
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
from openai import AzureOpenAI#class
#To manage environment variables
import os


#To manage blobs
from azure.storage.blob import BlobServiceClient#class

#To manage the types used along the script
from typing import List, TypedDict, Union





#I define some typed classes I'm going to use to type my code, highlight that all of them inherits from TypedDict
class Word(TypedDict):
    content: str
    confidence: float #in (0,1)

class Page(TypedDict):
    page_number: int
    width: float
    height: float
    unit: str
    words: List[Word]
    selection_marks: List

class Document(TypedDict):
    doc_id: int
    content: List[Page]

#Filtered typed classes
class Filtered1Page(TypedDict):
    page_number: int
    words: List[Word]
class Filtered1Document(TypedDict):
    doc_id: int
    content: List[Filtered1Page]



def filter_of_confidence(doc: Document, threshold: float)->Filtered1Document:
    #I create function thet removes the part of the data that we feel that
    #hasn't been extracted in a right way with the OCR, because of its confidence

    #PRE: doc is one of the components(dictionary) of the main list (composed of many documents) and 
    #threshold the minimum "confidence" of the "words" we want to mantain
    #POST: returns the document in same format with the "words" that have the key of "confidence" more than threshold, 
    #deleting the keys "selection marks", "width", "height" and "unit" from the "content" key of each page because aren't relevant

   #Initialize a new dictionary for the filtered document, maintaining the format
    filtered_doc : Filtered1Document = {
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
            #get the key confidence, if it doesn't exist return 0
            if word.get('confidence', 0) > threshold:
                #Preserve the whole word structure (content and confidence together)
                filtered_page['words'].append({'content': word['content'],'confidence': word['confidence']})
        
        #Add the filtered page to the content list
        filtered_doc['content'].append(filtered_page)
    
    return filtered_doc

def filter_data_by_confidence(data: List[Document], threshold:float)->List[Filtered1Document]:
    #PRE: data is the whole data(composed of dictionaries, each document is one dictionary) we want to filter by confidence
    #POST: returns the data in same format but filtered by confidence
    return [filter_of_confidence(doc, threshold) for doc in data]


#Define the typed classes uses in the filter2
class Filtered2Page(TypedDict):
    page_number: int
    words: List[str]
class Filtered2Document(TypedDict):
    doc_id: int
    content: List[Filtered2Page]

def clean_words(words:List[Word])->List[str]:
    #PRE: words is a list of dictionaries, which every dictionary has at least a 'content' key
    #POST: return a list of strings, without stopwords(words with a stopword in the key 'content') and without
    #words that are not relevant for the summarizer

    stop_words=set(stopwords.words('english'))
    pattern=re.compile(r'^[a-zA-Z]{3,}$')
    return [word['content'] for word in words if (word['content'].lower() not in stop_words) and (pattern.match(word['content']))]

def filter_of_stopwords(data:List[Document])->List[Filtered2Document]:
    #PRE: data has a json format and has been filtered by confidence
    #POST: returns data in json format, but now 'words' is returned as a list of 
    #strings(with all the words of each page filtered)

    filtered_by_stopwords: List[Filtered2Document] = []
    
    #Iterate over each document
    for doc in data:
        cleaned_content:Filtered2Document = []
        
        #Iterate over each page in the document
        for page in doc['content']:
            #Clean the words (remove stopwords)
            cleaned_words = clean_words(page['words'])
            
            #Add cleaned data to the content
            cleaned_content.append({'page_number': page['page_number'],'words': cleaned_words})
        
        #Add the cleaned document to the processed list
        filtered_by_stopwords.append({'doc_id': doc['doc_id'],'content': cleaned_content})
    
    return filtered_by_stopwords


def json_to_text_with_metadata(data: List[Filtered2Document])->str:
    #We want to transform the json in raw text to be sent to the AI
    #PRE: data must have the next format
    #  [{"doc_id": docNum1, 
    #    "content":[{"page_number": pageNum1,
    #                "words":["word1","word2",...]},   //end of the page1 of the doc1
    #               {"page_number": pageNum2,
    #                "words":[...]},{...}]  //end of content of the doc1
    #   {"doc_id": docNum2,
    #    "content": ...}
    #   {...}]     //end of the data
    #POST: returns the filtered content of each document and each page 
    #in an string(but specifying document and page content)

    #With this function we want to have the data in linear format, because
    #some NLP(Natural Processing Language) services would process better the data in this format
   
    text_content: List[str] = []

    #Go through all the documents in the data
    for document in data:
        doc_id = document['doc_id'] 
        
        #Go through all the pages on each document
        for page in document['content']:
            page_number = page['page_number'] 
            #Add to the list the number of document and page before the words 
            text_content.append(f"\n[Document {doc_id}, Page {page_number}]\n")
            #Add to the list all the words of each page
            text_content.extend(page['words'])
    #Return words into continous text, joining all the elements of text_content
    return ' '.join(text_content) 

def cleaner_of_data(data:List[Document])->str:
    #PRE: data is a list of documents we want to clean
    #POST:returns a string with the cleaned data form each page and document


    #Filter the data by confidence
    filter1=filter_data_by_confidence(data,0.8)
    #Filter the data (that has been filtered by confidence), by stop words
    filter2 =filter_of_stopwords(filter1)
    #Write the json document into continous text, specifying Document and Page numbers
    continuousText=json_to_text_with_metadata(filter2)
    
    return continuousText

def connection_to_data(myblob: func.InputStream)->List[Document]:
    #PRE:myblob is the new blob uploaded in a certain container in the cloud
    #POST: returns the json file saved in the blob but in object format
   
    try:
        #Read the content of the blob, given as a string
        blob_data=myblob.read().decode('utf-8')
        
        #Convert the content of the blob into python objects format
        jsonData=json.loads(blob_data)
        logging.info("Data from the blob obtained correctly, from the json file")
        return jsonData

    except Exception as e:
        logging.error(f"Error obtaining the data from the blob: {e}")
        #At this point we will see the error in the logs of the Azure Function
        return None
    


#From this next function on , we could create another Azure Function,
#one for the Step1(cleaning) and other for the Step2(summarizing)


def summarize_with_openai(text_to_summarize: str)->str:
    #PRE: immportantData is an list of objects, each object is a page from the whole cleaned data
    #POST: returns the final summary in a sequence of points with metadata(from which document and page it has obtained each point)
    try:
        #Get the endpoint and key from the virtual  environment
        endpoint = os.getenv("ENDPOINT_URL_AI")
        deployment = os.getenv("DEPLOYMENT_NAME_AI")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

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
                "content": "Lawyer that wants to sum up a collection of documents into the most important points of all the data. We don't want a summary of each page."
            },
            {
                "role": "user",
                "content": f"Sum up this collection of documents in 10 different important concepts and tell me from which certain page you took each concept. The result must be in json format:\n{text_to_summarize}"
            }],
            max_tokens=800,
            temperature=0.7,#more creative response, 0 will be coherence
            top_p=0.95,#probability of take words that are in the 95% of the acumulated probability
            stream=False#give the whole answer at same time
        )
        #response will be the whole answer of the gpt-4 service, in json format
        response= completion.to_json()
        #Obtain the objects from the json
        data = json.loads(response)
        #We want only the summary
        summary=data["choices"][0]["message"]["content"]
        return (summary)

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        #At this point we will see the error in the Logs the Azure function
        return None
    

def validate_summary(summary:str)->Union[Union[list,dict],None]:
    #PRE: summary is the summary received from the OpenAI service
    #POST: returns the summary if it's in json format and None in other case

    try:
        summaryJson=json.loads(summary)
        logging.info('Summary schematic and standardised in json format.')
        return summaryJson
    except Exception as e:
        logging.info(f'The summary is not in json format: {e}')
        return None
    

    
#We could create an extra Azure function just to decide what to do with the result,
#in my case I have choosen creating a blob in the same container that THE INITIAL DATA WAS UPLOADED, which
#produce this Azure Function to start

def upload_to_blob(container_name:str, blob_name:str, text: Union[list,dict]):
    #PRE: container_name is a container resource created in the Azure cloud, blob_name is a string which indicates the blob where
    #the final summary will be located, text is the final summary
    #POST: returns true if the summary has been well uploaded to the container(in a blob) and false in other case

    try:
        #Get the connection string from environment variables
        connection_string = os.getenv("CONNECTION_STORAGE")

        #Create a BlobServiceClient object to interact with the Blob service
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        #Get the reference to the container
        container_client = blob_service_client.get_container_client(container_name)

        #Create a reference to upload the text in the blob of name
        blob_client = container_client.get_blob_client(blob_name)

        #Upload text to the blob, if it was another blob with the same name, rewrite the content of it 
        #and if it wasn't any blob with that name we create it
        summary_to_encode=json.dumps(text,indent=4)

        blob_client.upload_blob(summary_to_encode.encode('utf-8'), blob_type="BlockBlob", overwrite=True)  

        #Upload a message into the loggings to communicate that the result has been uploades
        logging.info(f"Summary uploaded successfully to {blob_name} in the container: {container_name}")

        return 

    except Exception as e:
        logging.error(f"Error uploading summary to blob: {e}")
        #At this point we will see the error in the logs of the Azure Function
        return 

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="container/{name}",connection="CONNECTION_STORAGE") 
#blob_tigger activates the function when a new blob is uploaded in the specified container located in the cloud(resource)
#arg_name is the variable we are going to use in the function to refer the blob
#path is the route of the blob that will activate the function
#connection is the configuration that we are going to use to access to the container in the cloud

def blob_trigger_summarizer(myblob: func.InputStream):
    
    logging.info(f"Python blob trigger function processed blob. Name of the blob: {myblob.name}")
    
    if myblob.name.endswith("summary_report.json"):
        #We don't want to repeat the activation of the function uploading the final summary(another blob) in the container
        logging.info(f"Skipping process for blob: {myblob.name}")
        return

    #Gets the data of the blob that has been uploaded in json format
    json_data=connection_to_data(myblob)
   
    if json_data:
        #Clean data
        importantData=cleaner_of_data(json_data)
       
        #Send the data to the OpenAI resource
        summary=summarize_with_openai(importantData)
        
        if summary:
            logging.info(f"Final summary of the data updated in the container:\n {summary}")
            summaryJSON=validate_summary(summary)
            
            if summaryJSON:
                #Upload the final summary in a blob named "summary_report.json"
                finalBlob="summary_report.json"
                #will return a bool that will say if all the process has run correctly
                upload_to_blob('container', finalBlob, summaryJSON)

    return
                

    
                
            