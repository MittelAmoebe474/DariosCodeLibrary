#!/usr/bin/env python
# coding: utf-8

# In[10]:


from skimage import data
import matplotlib.pyplot as plt

# Assign data to varible
coins = data.coins()

# Find relevant attributes
"""
coin_attributes = dir(coins)
for attribute in coin_attributes:
    print(attribute)
"""

# Display relevant information
print("Image shape:", coins.shape)
print("Minimum pixel value:", coins.min())
print("Maximum pixel value:", coins.max())


# Display the coin image
fig, ax = plt.subplots()
image = ax.imshow(coins, cmap='gray')

# Add a title to the figure
ax.set_title('Coin Image')


# Add a colorbar
fig.colorbar(image)

# Add a name for the x-axis
ax.set_xlabel('image width in pixel')
ax.set_ylabel('image height in pixel')

# Show the figure
plt.show()


# In[29]:


import numpy as np
from skimage import data
from matplotlib import gridspec

coins = data.coins()

# Calculate the mean intensity of the coins image
mean_intensity = coins.mean()

# Create a zeros array with the same dimensions as the coins image
zeros_array = np.zeros_like(coins)

# Fill the zeros array with values greater than or equal to the mean intensity
zeros_array[coins >= mean_intensity] = coins[coins >= mean_intensity]

# Print the filled zeros array
# print(zeros_array)

# Print the shape of the zeros array
#print("Zeros array shape:", zeros_array.shape)

# Create the figure and adjust the layout using gridspec
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# Plot the 'coins' image in the first subplot
ax1 = plt.subplot(gs[0])
image1 = ax1.imshow(coins, cmap='gray')
ax1.set_title('Coins Image')

# Plot the filled zeros array in the second subplot
ax2 = plt.subplot(gs[1])
image2 = ax2.imshow(zeros_array, cmap='gray')
ax2.set_title('Filled Zeros Array')

# Add a colorbar to the figure
cbar_ax = plt.subplot(gs[2])
cbar = plt.colorbar(image2, cax=cbar_ax)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import data

coins = data.coins()

# Create a copy of the original 'coins' image
coins_copy = coins.copy()

# Fill each 10th line with the maximum intensity of the image
max_intensity = coins.max()
coins_copy[::10, :] = max_intensity

# Display the original 'coins' image and the modified copy
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original 'coins' image in the first subplot
axes[0].imshow(coins, cmap='gray')
axes[0].set_title('Original Coins Image')

# Plot the modified copy in the second subplot
axes[1].imshow(coins_copy, cmap='gray')
axes[1].set_title('Modified Coins Image')

# Create a new figure for displaying the line-array and intensity profiles
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

# Display the line-array along the y-direction
axes2[0].imshow(coins_copy, cmap='gray')
axes2[0].set_title('Line-Array along Y-Direction')
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
axes2[0].plot([0, coins_copy.shape[1] - 1], [coins_copy.shape[0] // 2, coins_copy.shape[0] // 2], 'r--')

# Display the intensity profile along the x-direction
profile_x = np.mean(coins_copy, axis=0)
axes2[1].plot(profile_x)
axes2[1].set_title('Intensity Profile along X-Direction')
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Intensity')

# Display the intensity profile along the y-direction
profile_y = np.mean(coins_copy, axis=1)
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(profile_y)
ax3.set_title('Intensity Profile along Y-Direction')
ax3.set_xlabel('Y')
ax3.set_ylabel('Intensity')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figures
plt.show()


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import ndimage

coins = data.coins()

# Create a copy of the original 'coins' image
coins_copy = coins.copy()

# Fill each 10th line with the maximum intensity of the image
max_intensity = coins.max()
coins_copy[::10, :] = max_intensity

# Display the original 'coins' image and the modified copy
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original 'coins' image in the first subplot
axes[0].imshow(coins, cmap='gray')
axes[0].set_title('Original Coins Image')

# Plot the modified copy in the second subplot
axes[1].imshow(coins_copy, cmap='gray')
axes[1].set_title('Modified Coins Image')

# Create a new figure for displaying the line-array and intensity profiles
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

# Display the line-array along the y-direction
axes2[0].imshow(coins_copy, cmap='gray')
axes2[0].set_title('Line-Array along Y-Direction')
axes2[0].set_xlabel('X')
axes2[0].set_ylabel('Y')
axes2[0].plot([0, coins_copy.shape[1] - 1], [coins_copy.shape[0] // 2, coins_copy.shape[0] // 2], 'r--')

# Smooth the intensity profile along the x-direction using Gaussian filtering
profile_x = np.mean(coins_copy, axis=0)
smoothed_profile_x = ndimage.gaussian_filter1d(profile_x, sigma=3)

axes2[1].plot(smoothed_profile_x)
axes2[1].set_title('Smoothed Intensity Profile along X-Direction')
axes2[1].set_xlabel('X')
axes2[1].set_ylabel('Intensity')

# Display the intensity profile along the y-direction
profile_y = np.mean(coins_copy, axis=1)
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(profile_y)
ax3.set_title('Intensity Profile along Y-Direction')
ax3.set_xlabel('Y')
ax3.set_ylabel('Intensity')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figures
plt.show()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import data

coins = data.coins()

# Calculate the mean intensity of the coins image
mean_intensity = coins.mean()

# Create arrays based on intensity conditions
less_mean = np.zeros_like(coins)
greater_mean = np.zeros_like(coins)
within_range = np.zeros_like(coins)

# Populate the arrays based on intensity conditions
less_mean[coins < mean_intensity] = coins[coins < mean_intensity]
greater_mean[coins > mean_intensity] = coins[coins > mean_intensity]
within_range[np.abs(coins - mean_intensity) <= 0.1 * mean_intensity] = coins[np.abs(coins - mean_intensity) <= 0.1 * mean_intensity]

# Create the RGB image by stacking the arrays
rgb_image = np.stack((less_mean, greater_mean, within_range), axis=2)

# Display the RGB image
plt.imshow(rgb_image)
plt.axis('off')
plt.title('RGB Image with Intensity Conditions')
plt.show()

