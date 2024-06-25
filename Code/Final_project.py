# %%
import tkinter as tk
from tkinter import ttk
from tkinter import Button, Label, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, UnidentifiedImageError
import cv2
import numpy as np

#---------------------Cartoonization 1---------------------------
def smooth_image(img):
    # Apply Gaussian blur to smooth the image
    img_smooth = cv2.GaussianBlur(img, (7, 7), 0)
    # Apply median blur for further smoothing
    img_smooth = cv2.medianBlur(img_smooth, 5)
    # Apply bilateral filter for edge-preserving smoothing
    img_smooth = cv2.bilateralFilter(img_smooth, 5, 80, 80)
    return img_smooth

def edge_detection(img):
    # Apply Laplacian edge detection
    img_edges = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
    # Convert the edge-detected image to grayscale
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)
    # Remove additional noise
    img_edges = cv2.GaussianBlur(img_edges, (5, 5), 0)
    return img_edges

def apply_threshold(blur_img):
    # Apply Otsu's thresholding to segment the image
    _, thresh_img = cv2.threshold(blur_img, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img

def invert_image(thresh_img):
    # Invert the thresholded image
    inverted_img = cv2.subtract(255, thresh_img)
    return inverted_img

def kmeans_color_quantization(img, K=8):
    # Reshape the image for k-means clustering
    img_reshaped = img.reshape((-1, 3))
    img_reshaped = np.float32(img_reshaped)
    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Apply k-means clustering
    _, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # Assign cluster centers to each pixel
    res = center[label.flatten()]
    img_quantized = res.reshape((img.shape))
    return img_quantized

def binarize_image(img, div=64):
    # Binarize the image by quantizing pixel values
    return img // div * div + div // 2

def cartoonization1(imgPath):
    
    img = cv2.imread(imgPath)

    img_smooth = smooth_image(img)
    img_edges = edge_detection(img_smooth)
    thresh_img = apply_threshold(img_edges)
    
    inverted_img = invert_image(thresh_img)
    img_quantized = kmeans_color_quantization(img)    
    img_binarized = binarize_image(img)
    
    # Resize the binarized image to match the size of the inverted image
    img_binarized = cv2.resize(img_binarized, (inverted_img.shape[1], inverted_img.shape[0]))
    
    # Convert the inverted image to RGB format
    inverted_img_rgb = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2RGB)
    
    # Apply bitwise AND operation to combine the inverted image with the binarized image
    cartoon_image = cv2.bitwise_and(inverted_img_rgb, img_binarized)
    
    # Write the cartoonized image to a file
    cv2.imwrite('CartoonImage.png', cartoon_image)
    
    return cartoon_image

#---------------------Cartoonization 2---------------------------
def cartoonization2(imgPath):
    # Parameters for downsampling and bilateral filtering
    downsampleNum = 2
    bilateralNum = 7

    img = cv2.imread(imgPath)
    
    # Ensure the image dimensions are multiples of 4 for consistent downsampling and upsampling
    shape = tuple(list(((i // 4) * 4 for i in img.shape))[:2])[::-1]
    img = cv2.resize(img, dsize=shape)

    # Downsample the image
    imgColor = img
    for _ in range(downsampleNum):
        imgColor = cv2.pyrDown(imgColor)

    # Apply bilateral filter multiple times to smooth the image while keeping edges sharp
    for _ in range(bilateralNum):
        imgColor = cv2.bilateralFilter(imgColor, d=9, sigmaColor=5, sigmaSpace=7)
    
    # Upsample the image back to its original size
    for _ in range(downsampleNum):
        imgColor = cv2.pyrUp(imgColor)

    # Convert the image to grayscale
    imgBinary = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply median blur to the grayscale image
    imgMedianBlur = cv2.medianBlur(imgBinary, 9)

    # Detect edges using adaptive thresholding
    imgContour = cv2.adaptiveThreshold(
        imgMedianBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=7, C=2
    )
    
    # Convert the binary image to RGB format
    imgContour = cv2.cvtColor(imgContour, cv2.COLOR_GRAY2RGB)

    # Combine the smoothed color image with the edge mask
    img_cartoon = cv2.bitwise_and(imgColor, imgContour)

    cv2.imwrite('CartoonImage.png', img_cartoon)
    
    return img_cartoon

# %%
#-------------------------------Front end----------------------------
def upload():
    global result_label, ImagePath

    try:
        # Open a file dialog to select an image file
        ImagePath = askopenfilename()        
        img = Image.open(ImagePath)
        
        # Resize the image to have a minimum width of 400 pixels
        min_width = 400
        img = img.resize((min_width, (min_width * img.size[1] // img.size[0])))
        
        pic = ImageTk.PhotoImage(img)

        # Update the label widget to display the uploaded image
        result_label.config(image=pic)
        result_label.image = pic  # Keep a reference to avoid garbage collection
        result_label.pack()  # Ensure the label is displayed

    except UnidentifiedImageError:
        # Handle the error if the selected file is not a valid image
        messagebox.showinfo(title='Upload Error', message='This is not an image. Try again!')

def cartoonize():
    global ImagePath, combo

    # Check if an image has been uploaded
    if not ImagePath:
        messagebox.showinfo(title='Error', message='No image received yet!')
        return
    
    method = combo.get()

    # Apply the selected cartoonization method
    if method == 'Style 1':
        cartoon = cartoonization1(ImagePath)
    elif method == 'Style 2':
        cartoon = cartoonization2(ImagePath)
    else:
        messagebox.showinfo(title='Error', message='Please select one of the methods!')
        return

    # Convert the cartoonized image from BGR to RGB 
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    
    # Convert the cartoonized image to a PIL image
    im_pil = Image.fromarray(cartoon)

    # Resize the image to have a minimum width of 400 pixels 
    min_width = 400
    im_pil = im_pil.resize((min_width, (min_width * im_pil.size[1] // im_pil.size[0])))
    
    pic = ImageTk.PhotoImage(im_pil)

    # Update the label widget to display the cartoonized image
    result_label.config(image=pic)
    result_label.image = pic 
    result_label.pack(fill=tk.NONE, expand=True)  

def resizer(event):
    global background_image, canvas
    # Adjust the size of the background image to the canvas size
    new_bg = Image.open("background.jpeg")
    new_bg = new_bg.resize((event.width, event.height), Image.LANCZOS)
    background_image = ImageTk.PhotoImage(new_bg)
    canvas.create_image(0, 0, image=background_image, anchor="nw")

def setup_gui():
    global canvas, result_label, background_image, combo

    WIDTH, HEIGHT = 1000, 700
    top = tk.Tk()
    top.geometry(f'{WIDTH}x{HEIGHT}')
    top.title('Cartoonify Your Image!')

    background_image = ImageTk.PhotoImage(Image.open("background.jpeg"))

    canvas = tk.Canvas(top, width=WIDTH, height=HEIGHT, bg='white')
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.create_image(0, 0, image=background_image, anchor="nw")
    canvas.bind('<Configure>', resizer)

    # Display sample image
    eg = Image.open("example.jpg").resize((225,390))
    example_img = ImageTk.PhotoImage(eg)
    example_label = tk.Label(canvas, image=example_img)
    example_label.place(relx=0.03, rely=0.05, anchor='nw')
    
    # Upload button
    upload_button = Button(canvas, text="UPLOAD", command=upload, padx=10, pady=5)
    upload_button.configure(background='#84afcc', foreground='black', font=('calibri', 15, 'bold'))
    upload_button.pack(side=tk.TOP, pady=50)

    # Instruction text box
    text = tk.Text(canvas, height=10, width=33, font=('calibri', 12))
    insert_text = (
        "Steps to cartoonize your photo:\n"
        "1. Upload the image using the UPLOAD button\n"
        "2. Choose a method in the drop-down menu\n"
        "3. Press the START button\n"
        "4. Choose another method in the drop-down menu and press START to try another style\n"
        "5. Press UPLOAD to use another image and repeat the above steps"
    )
    text.insert(tk.END, insert_text)
    text.place(relx=0.03, rely=0.95, anchor='sw')

    # Dropdown Menu for selecting cartoonization method
    combo = ttk.Combobox(canvas, values=["Style 1", "Style 2"])
    combo.set("Select Method")
    combo.pack()

    # Start button
    cartoonize_button = Button(canvas, text="START", command=cartoonize, padx=10, pady=5)
    cartoonize_button.configure(background='#84afcc', foreground='black', font=('calibri', 15, 'bold'))
    cartoonize_button.pack(side=tk.BOTTOM, pady=50)

    result_label = Label(canvas)
    result_label.pack()
    top.mainloop()

if __name__ == "__main__":
    setup_gui()