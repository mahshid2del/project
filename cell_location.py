import cv2
import os
import pandas as pd

input_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results"
output_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/cell_location"
output_file = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/bb_results.xlsx"

cellresults = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply threshold and find contours
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes and ellipses on image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w/2
            center_y = y + h/2
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(img, ellipse, (0, 0, 255), 2)

            # Extract minor and major axis of ellipse
                (x, y), (MA, ma), angle = ellipse
                if ma > MA:
                    ma, MA = MA, ma
                cellresults.append([filename, MA, ma, w, h, center_x, center_y])
                cv2.putText(img, f"Major axis: {MA:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"Minor axis: {ma:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"Width: {w}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"Height: {h}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

                
            else:
                print(f"Skipping ellipse fitting for contour {contour}, as it has less than 5 points")
  
                
        # Save output image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print("Processed:", img_path)
        
# Save results to Excel file
df = pd.DataFrame(cellresults, columns=["file_name", "major_axis", "minor_axis", "width", "height", "centerX", "centerY"])
df.to_excel(output_file, index=False)
