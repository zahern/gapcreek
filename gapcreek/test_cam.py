import cv2
import csv


# Mouse event callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


clicked_points = []
# Load the image
test_ = "3A"
image = cv2.imread(f'{test_}_input_test.jpg')

# Create a named window for the image
cv2.namedWindow('Image')

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', mouse_callback)

# Display the image
cv2.imshow('Image', image)

# Wait for the user to click points and press 'Esc' to exit
while True:
    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break

# Close all windows
cv2.destroyAllWindows()

csv_file = test_ + 'clicked_points.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])  # Write header row
    writer.writerows(clicked_points)

print(f"Clicked points saved to {csv_file}.")
