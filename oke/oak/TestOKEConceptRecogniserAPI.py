'''
Created on 15 May 2015

@author: jieg
'''

class TestOKEConceptRecogniserWS(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
if __name__ == '__main__':
    import requests
    test_file_path="../test/task2_test.ttl"
    with open(test_file_path,'r',encoding="utf-8") as f:
        content=f.read()   
            
    r = requests.post("http://localhost:5000/extract", data=content)
    print(r.status_code, r.reason)
    print(r.content)