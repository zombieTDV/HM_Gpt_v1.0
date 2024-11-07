import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class ANIMATOR:
    def __init__(self, list_of_data: list, fps:60, title: str) -> None:
        self.list_of_data = list_of_data
        
        self.fig = plt.figure()
        
        self.title = title
        metadata = dict(title = self.title, artist='LLNN')
        self.writer = FFMpegWriter(fps, metadata=metadata)
        
    def img_plot(self):
        with self.writer.saving(self.fig, f"{self.title}.mp4", 100):
            for data in self.list_of_data:
                plt.imshow(data)
                self.writer.grab_frame()
                
# import numpy as np

# data = []
# for i in range(100):
#     data.append(np.random.randn(9,9))
    
# Animator = ANIMATOR(data, 10, 'random')

# Animator.img_plot()