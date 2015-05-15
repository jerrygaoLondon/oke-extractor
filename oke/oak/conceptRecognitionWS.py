'''
Expose oke-challenge Task 2 as a RESTful Service
    
Class Induction and entity typing for Vocabulary and Knowledge Base enrichment.

@author: jieg
'''
import sys
sys.path.append("../../")

from flask import Flask
from flask import request
from flask import Response

app = Flask(__name__)

@app.route('/extract', methods =['POST'])
def api_extract():
    print("======api extract=====")
    requestedData = request.get_data()
    
    print("requestedData:",requestedData)
    from oke.oak.okeConceptRecogniser import ConceptRecogniser
    conceptRecogniser=ConceptRecogniser()
    
    print("attempting to extact concept and return enriched graph...")
    returnedN3Graph = conceptRecogniser.type_extraction_and_interlink(requestedData)
    
    resp = Response(returnedN3Graph, status=200, mimetype='text/rdf+n3')
    return resp
if __name__ == '__main__':
    app.run(host="localhost", port=5000)