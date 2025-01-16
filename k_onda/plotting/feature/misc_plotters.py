import matplotlib.pyplot as plt

from k_onda.base import Base

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 
   

class TextPlotter(Base):
    """
    Very minimal text plotter.
    """
    def plot(self, cell, text_spec):
        # If cell is an AxWrapper, cell.ax is the actual Matplotlib Ax
        ax = cell.ax if hasattr(cell, 'ax') else cell
        ax.text(0.5, 0.5, text_spec['content'], ha='center', va='center')
        ax.set_axis_off()
        


