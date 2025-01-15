from .partitions import Section, Segment, Series, Split
from .processor import Container

PROCESSOR_MAP = {
            'series': Series,
            'section': Section,
            'split': Split,
            'segment': Segment,
            'container': Container
        }