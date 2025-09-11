from k_onda.core import Base


class CustomDataImporter(Base):
 
    def import_data(self, source):
        raise NotImplementedError
    
    def validate_data(self, data):
        raise NotImplementedError
    

class CSVDataImporter(Base):

    pass


class CustomDataCalculator(Base):

    def __init__(self):
        pass

    


    
