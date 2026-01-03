import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):
    # Iterate using zip to safely handle frame list and dataframe rows together
    # regardless of dataframe index values.
    # Note: player_stats should cover the frames in output_video_frames in order.
    
    # Check if player_stats is a DataFrame, if so verify len match roughly or just iterate
    # We assume simple row-by-row correspondence
    
    for i, (frame, (_, row)) in enumerate(zip(output_video_frames, player_stats.iterrows())):
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        shapes = np.zeros_like(frame, np.uint8)

        width=350
        height=230

        start_x = frame.shape[1]-400
        start_y = frame.shape[0]-500
        end_x = start_x+width
        end_y = start_y+height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text
        text = "     Player 1     Player 2"
        frame = cv2.putText(frame, text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. S. Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        frame = cv2.putText(frame, text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        frame = cv2.putText(frame, text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update the list (though list elements are mutable, we assigned frame back)
        output_video_frames[i] = frame
    
    return output_video_frames