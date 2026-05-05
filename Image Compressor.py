
#Code that uses SVD to take an image and approximate it with specified # of coefficients

from PIL import Image;
import numpy as np;
import tkinter as tk
import os
import platform
import subprocess
from tkinter import filedialog

# This prevents a tiny blank tkinter window from staying open
root = tk.Tk()
root.withdraw()

# Open the file explorer window
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

# Check if the user actually selected a file
if not file_path:
    print("No file selected. Exiting.")
    exit()

img = Image.open(file_path).convert("RGB")
img.thumbnail((800, 800))

#Turning image into matrix
A = np.array(img)

#Express matrix thru SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)


# Number of values to approximate/select from SVD to RGB
input_k = int(input("Number of Coefficients to Approximate Image: "))

channels = []


#Looping for R,G,B channels (0,1,2)
for i in range(3):
    #Slice specific channel: [all rows, all cols, channel_i]
    channel_matrix = A[:,:, i]

    #SVD on individual channel
    U, S, Vt = np.linalg.svd(channel_matrix, full_matrices=False)

    #Ensure K doesn't exceed available singular values
    k= min(input_k, len(S))

    #Reconstruct Channel w/ k coefficients, Ak = U_k * Sigma_k * V^T_k
    Ak = U[:,:k] @ np.diag(S[:k]) @ Vt[:k, :] 
    channels.append(Ak)


#Stack channels together, keep values to valid RGB range (0-255)
compressed = np.stack(channels, axis=2)
compressed = np.clip(compressed,0,255).astype(np.uint8)

#Display the optimized image
Image.fromarray(compressed).show()
compressed_image = Image.fromarray(compressed)


# Ask user where to save the file and what to name it
save_path = filedialog.asksaveasfilename(
    defaultextension=".jpg",
    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")],
    title="Save Compressed Image As"
)

if save_path:
    # Save the image
    final_img = Image.fromarray(compressed)
    final_img.save(save_path)
