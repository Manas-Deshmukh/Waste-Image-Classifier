import cv2

# Load Image
image = cv2.imread("C:/Users/Manas Deshmukh/OneDrive/Desktop/Mac_DIP Project/dataset/plastic/plastic1.jpg")  # Change to your image file

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display Image
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the Grayscale Image
cv2.imwrite("grayscale_output.jpg", gray)
print("Grayscale image saved as 'grayscale_output.jpg'")