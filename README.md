# Azure Function

## Context

#### We suppose that we have a suscription of Microsoft Cloud(Azure), in which we have the next resources:
- Storage Account with a container.
- Function App, Blob Trigger(where this proyect will be deployed).
- OpenAI service.

## Specification

#### This Azure Function, developed with Python, runs everytime that a file is uploaded in the container. The file is supposed to be in json format, with data about many documents(part of the same "matter") that have been performed by an OCR, like the following: 
```
[
   {
    "doc_id": 1,
    "content": [
      {
        "page_number": 1,
        "width": 8.2639,
        "height": 11.6806,
        "unit": "inch",
        "words": [{"content": "Submission", "confidence": 0.994},{"content": "to", "confidence": 0.995},{...}],
        "selection_marks"=[]
      },
      {
        "page_number": 2,
        "width": 8.2639,
        "height": 11.6806,
        "unit": "inch",
        ...//words of page 2
      },     //more pages
      {...}]
   },
   {
    "doc_id": 2,
      ...      //content doc 2
   },
    ...     //more documents
 
]
```

#### The function will manage the blob(storage solution of Azure) of data that has been uploaded and will create another blob in the same container. This second blob will contain a .txt file with a summary of the data divided by points(with metadata, from which document and page it has obtained each point).

## Inner functions

#### The Function of Azure(`cleanerFunction()`) is divided in 4 steps: 

  1. Clean the data(which includes connecting to the blob in which is the data). Functions:
`filter_of_confidence()`,`filter_data_by_confidence()`,`clean_words()`,`filter_of_stopwords()`,`json_to_text_with_metadata()`,`cleaner_of_data()`,`connection_to_data()`    
  2. Summarize the cleaned data(which includes connecting to the OpenAI resource). Functions:
    `summarize_with_openai()`
  3. Validate the summary(ensure the summary is in schematic format). Functions:
     `validate_summary()`
  4. Upload the summary at the container(with a blob client). Functions:
      `upload_to_blob()`*
#### *When the second blob(with the summary) is uploaded, the Azure Function won't be invoked to prevent the infinite loop(redundant invocation control).
#### (To be more effective run the function in a Linux OS, will be more robust)

    
      
        
        
          
  

