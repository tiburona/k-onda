from ..base import Base


class AbstractDataImporter(Base):
 
    def import_data(self, source):
        raise NotImplementedError
    
    def validate_data(self, data):
        raise NotImplementedError
    
    
