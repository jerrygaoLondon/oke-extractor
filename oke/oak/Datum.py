class Datum:
    def __init__(self, contextURI, word, label):
        self.contextURI=contextURI
        self.word = word
        #self.beginIndex=beginIndex
        #self.endIndex=endIndex
        self.label = label
        self.guessLabel = ''
        self.previousLabel = ''
        self.features = []
    
