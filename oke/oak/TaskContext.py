'''
Created on 25 Apr 2015

@author: jieg
'''

class TaskContext(object):
    def __init__(self, contextURI):
        #isString,beginIndex,endIndex,entity
        self.contextURI=contextURI
        self.isString=""
        self.beginIndex=""
        self.endIndex=""
        '''
        if type(entity) is not ContextEntity:
            raise Exception("'entity' must be the oke.oak.ContextEntity")
        '''
        self.entity=None
        
class ContextEntity():
    def __init__(self, entityURI):
        #,anchorOf, referenceContext, beginIndex, endIndex,taIdentRef,entityClasses
        self.entityURI=entityURI
        self.anchorOf=""
        self.referenceContext=""
        self.beginIndex=""
        self.endIndex=""
        #dbpedia URI
        self.taIdentRef=""
        #a list of types the entity belongs to e.g., dbpedia:Brian_Banner a oke:FictionalVillain, oke:Villain
        self.isInstOfEntityClasses=None
    
class EntityClass():
    def __init__(self,uri):  
        #, anchorOf,referenceContext,beginIndex,endIndex, subClassOf      
        self.uri=uri
        self.anchorOf=""
        self.referenceContext=""
        self.beginIndex=""
        self.endIndex=""
        #e.g., subClassOf: http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Person
        self.subClassOf=""      