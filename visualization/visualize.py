import cv2
import numpy as np

def visualize_multiple_images(
    max_cols: int = 5,
    image_paths: np.ndarray = np.empty((0,)),
    texts: np.ndarray = np.empty((0,))) -> None:

    d, m = divmod(len(image_paths), max_cols)
    grid_rows = d if not m else d + 1
    grid_cols = max_cols
    
    # Resize images to fit in the grid cells
    cell_height = 300
    cell_width = 300

    # Create a blank canvas for the grid
    grid_height = grid_rows * cell_height
    grid_width = grid_cols * cell_width
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background

    # Loop through the images and place them in the grid
    for i in range(grid_rows):
        for j in range(grid_cols):
            index = i * grid_cols + j
            if index >= len(image_paths):
                break

            # Load the image
            image = cv2.imread(image_paths[index])
            # Resize the image to fit in the cell
            image = cv2.resize(image, (cell_width, cell_height))
            # Calculate the position for the image in the grid
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            # Place the image in the grid
            grid[y_start:y_end, x_start:x_end] = image

            # Overlay text on the image
            text = f"Classify: {texts[index]}"
            cv2.rectangle(grid, (x_start + 5, y_start + 15), (x_start + len(text) * 9, y_start + 38), (220, 220, 220), -1)
            cv2.putText(grid, text, (x_start + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the grid
    cv2.imshow('Facial Emotion Recognition', grid)
    cv2.waitKey(0)
    cv2.imwrite('results/result_1.png', grid)
    cv2.destroyAllWindows()
    