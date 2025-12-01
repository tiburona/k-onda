from .partitions import Section, Segment, Series
from .processor import Container

PROCESSOR_MAP = {
            'series': Series,
            'section': Section,
            'segment': Segment,
            'container': Container
        }