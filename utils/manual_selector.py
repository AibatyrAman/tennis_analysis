import cv2
import numpy as np

def select_corners_manually(frame):
    """
    Opens a window and allows the user to manually select 4 corners of the court.
    Returns the list of 4 points [(x, y), ...].
    """
    corners = []
    
    # Resize frame for selection if it's too big (optional, but good for UX on smaller screens)
    # Keeping it simple for now and using original size to ensure accuracy mapping back.
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append((x, y))
                print(f"Corner {len(corners)} selected: ({x}, {y})")
                
                # Visual feedback
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                if len(corners) > 1:
                    cv2.line(display_frame, corners[-2], corners[-1], (0, 255, 0), 2)
                if len(corners) == 4:
                    # Close the loop
                     cv2.line(display_frame, corners[-1], corners[0], (0, 255, 0), 2)
                
                cv2.imshow("Select 4 Corners", display_frame)

    cv2.imshow("Select 4 Corners", display_frame)
    cv2.setMouseCallback("Select 4 Corners", mouse_callback)

    print("Please click on the 4 corners of the court in the popup window.")
    print("Press 'q' or 'Enter' when done (must select 4 corners).")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13: # q or Enter
            if len(corners) == 4:
                break
            else:
                print(f"You checked {len(corners)} corners. Please select exactly 4.")
        
        # Check if window was closed manually
        if cv2.getWindowProperty("Select 4 Corners", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    
    if len(corners) == 4:
        return corners
    else:
        print("Selection cancelled or incomplete.")
        return None
