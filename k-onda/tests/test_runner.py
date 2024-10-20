import os
import csv
import json
from runner import Runner


TEST_PSTH_OPTS = {
    'calc_opts': {
        'kind_of_data': 'spike', 'data_type': 'psth',
        'events': {'pretone': {'pre_event': .05, 'post_event': .65}, 'tone': {'pre_event': .05, 'post_event': .65}},
        'bin_size': 0.01, 'adjustment': 'none', 'time_type': 'block', 'data_path': './data', 'row_type': 'event'}
}

units_per_animal = 4
tone_periods_per_animal = 2
num_animals = 4
sampling_rate = 1000


def run_test():
    # Step 2: Execute the Test
    runner = Runner(config_file='./data/exp_config.json')
    runner.run('make_spreadsheet', TEST_PSTH_OPTS, path='./output', filename='test_psth')

    # Step 3: Verify the Results
    verify_output('./output/psth/test_psth.csv')

def skip_lines(file):
    # Skip the metadata line
    next(file)
    # Skip the blank line
    next(file)
    # Yield the rest of the lines (including the header)
    for line in file:
        yield line


def verify_output(output_file):
    # Implement verification logic here
    # This is a simple example to get you started
    with open('./output/psth/test_psth.csv', 'r') as file:
        filtered_lines = skip_lines(file)
        reader = csv.DictReader(filtered_lines)
        data_rows = list(reader)
        # Example assertion: Check if the file is not empty
        assert len(data_rows) > 0, "Output file is empty."

        bin_size = TEST_PSTH_OPTS['calc_opts']['bin_size']
        pre_event = TEST_PSTH_OPTS['calc_opts']['events']['tone']['pre_event']
        post_event = TEST_PSTH_OPTS['calc_opts']['events']['tone']['post_event']
        events_per_period = 30
        num_bins_per_event = int((pre_event + post_event)/bin_size)
        expected_len = num_animals * units_per_animal * tone_periods_per_animal * 2 * events_per_period

        assert len(data_rows) == expected_len, "Unexpected number of rows in data"

    with open('./data/exp_config.json') as f:
        source_data = json.load(f)
        one_unit_data = source_data['animals'][0]['units']['good'][0]['spike_times']
        for event in range(10):
            start = 30000 + (event - pre_event) * sampling_rate
            end = start + (pre_event + post_event) * sampling_rate
            expected_firing_rate = len([spike for spike in one_unit_data if start <= spike < end])/(pre_event +post_event)
            assert data_rows[event]['rate'] == expected_firing_rate, "Unexpected firing rate"


        # More sophisticated checks can be added here
        # For example, checking specific values in rows or columns

if __name__ == "__main__":
    run_test()
    print("Test completed successfully.")
