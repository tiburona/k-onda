
class ColorbarMixin:

    @property
    def has_colorbar(self):
        return bool(self.colorbar_spec)

    @property
    def colorbar_spec(self):
        return self.spec and self.spec.get('legend', {}).get('colorbar', {})
           
    @property
    def colorbar_for_each_plot(self):
        return self.has_colorbar and self.colorbar_spec.get('share') in ['each', None]

    @property
    def global_colorbar(self):
        return self.has_colorbar and self.colorbar_spec.get('share') == 'global'

    @property
    def colorbar_position(self):
        return self.colorbar_spec.get('position')
    
    def create_outer_and_subgrid(self):
        outer_grid, main_slice, cax_slice = self.make_outer_grid() 
        self.figure = outer_grid[main_slice]
        self.figure.cax = outer_grid[cax_slice].add_subplot(111)
        self.processor.figure = self.figure
    
    def make_outer_grid(self):
       
        i = ['top', 'bottom', 'left', 'right'].index(self.colorbar_position)
        outer_dim = [1, 1]
        outer_dim[i//2] += 2
        ratio_string = ['height', 'width'][i//2]
        ratios = [1, .025, .075]
        if not i % 2:
            ratios.reverse()
        outer_grid = self.subfigures(*outer_dim, **{f'{ratio_string}_ratios': ratios})
        location_of_main_figure = [0, 0]
        location_of_colorbar_ax = [0, 0]

        location_of_colorbar_ax[i//2] += 1
        if not i%2:
            location_of_main_figure[i//2] += 2

        return (outer_grid, tuple(location_of_main_figure), 
                tuple(location_of_colorbar_ax))
    


        





    

        



