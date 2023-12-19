import numpy as np


class BlockConstructor:

    @property
    def all_blocks(self):
        return [block for block_type, block_list in self.blocks.items() for block in block_list]

    @property
    def earliest_block(self):
        return sorted(self.all_blocks, key=lambda x: x.onset)[0]

    def prepare_blocks(self):
        for boo, function in zip((False, True), (self.construct_blocks, self.construct_reference_blocks)):
            try:
                block_info = self.block_info
            except AttributeError:
                block_info = self.parent.block_info
            filtered_block_info = {k: v for k, v in block_info.items() if bool(v.get('reference')) == bool(boo)}
            for block_type in filtered_block_info:
                self.blocks[block_type] = function(block_type, filtered_block_info[block_type])

    def construct_blocks(self, block_type, block_info):
        blocks = []
        num_events = len([event for events_list in block_info['events'] for event in events_list])  # all the events for this block type
        if self.data_opts is not None and self.data_opts.get(f"{block_type}_events"):
            events = slice(*self.data_opts.get(f"{block_type}_events"))  # the events used in this data analysis
        else:
            events = slice(*(0, num_events))
        selected_event_indices = list(range(num_events))[events]  # indices of the events used in this data analysis
        block_onsets = block_info['onsets']  # the time stamp of the beginning of a block
        block_events = block_info['events']  # the time stamps of things that happen within the block
        event_ind = 0
        for i, (onset, events) in enumerate(zip(block_onsets, block_events)):
            block_events = np.array([ev for j, ev in enumerate(events) if event_ind + j in selected_event_indices])
            event_ind += len(block_info['events'][i])
            blocks.append(self.block_class(self, i, block_type, block_info, onset, events=block_events))
        return blocks

    def construct_reference_blocks(self, block_type, block_info):
        blocks = []
        shift = block_info['shift']
        duration = block_info.get('duration')
        paired_blocks = self.blocks[block_info['target']]
        for i, paired_block in enumerate(paired_blocks):
            onset = paired_block.onset - shift * self.sampling_rate
            duration = duration if duration else paired_block.duration
            reference_block = self.block_class(self, i, block_type, block_info, onset, paired_block=paired_block,
                                               is_reference=True)
            paired_block.paired_block = reference_block
            blocks.append(reference_block)
        return blocks

