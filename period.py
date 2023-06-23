import numpy as np
from scipy.signal import correlate2d

def find_period(image):
    # Calculate the autocorrelation of the image
    autocorr = correlate2d(image, image, mode='same')

    # Find the peak in the autocorrelation
    peak_indices = np.unravel_index(np.argmax(autocorr), autocorr.shape)
    peak = autocorr[peak_indices]

    # Determine the period as the distance from the peak to the closest edge
    width, height = image.shape
    period_x = min(peak_indices[1], width - peak_indices[1]) * 2
    period_y = min(peak_indices[0], height - peak_indices[0]) * 2

    return period_x, period_y

# Example usage
from PIL import Image
image = Image.open('img/siggraph_asia/redknitwear.jpg').convert('L').resize((200, 200))
image = np.array(image)
print(image.shape)

# image = np.array([[0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 1, 1, 1],
#                   [1, 1, 1, 0, 0, 0],
#                   [1, 1, 1, 0, 0, 0],
#                   [1, 1, 1, 0, 0, 0]])

period_x, period_y = find_period(image)
print(f"Period along x-axis: {period_x}")
print(f"Period along y-axis: {period_y}")
with open("period.txt", 'a') as savefile:
    savefile.write(f"{period_x}, {period_y}")

