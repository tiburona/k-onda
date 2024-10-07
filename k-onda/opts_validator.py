class OptsValidator:

    def validate_opts(self, opts):
        # TODO: update animal selection validation
        all_animal_ids = [animal.identifier for animal in self.experiment.all_animals]
        selected_animals = opts.get('selected_animals')
        if selected_animals is not None and not all([id in all_animal_ids for id in selected_animals]):
            raise ValueError("Missing animals")
        if opts.get('kind_of_data') == 'spike' and opts.get('adjustment') != 'none':
            if opts.get('evoked'):
                raise ValueError("It does not make sense to set 'evoked' to True and 'adjustment' to anything other "
                                 "than 'none'.  See Analysis Configuration Reference.")
            if not opts.get('periods'):
                raise ValueError("You picked a value for adjustment other than 'none' and haven't specified which "
                                 "periods to include.  This will result in a nonsensical result.  See the Analysis "
                                 "Configuration Reference.")
       
