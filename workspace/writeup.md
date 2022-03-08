# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?
filter: Calman Filter is implemented as 'Filter' class. This class has methods for calculation of Kalman Filter parameters, state prediction and motion update.
track management: Track management is implemented as 'Trackmanagement' class. 
                  Depend on measurements and association, the track management add/delete track and change track score.
association: Association is implemented as 'Association' class. The unassigned tracks and unassigned measurements are associated.
             To associate the closest measurement and track, this class use the mahalanobis distance and gating.
camera fusion: To improve object detection, both LiDAR sensor and camera sensor data are used.

Association is the most difficult part to complete.
When I implemented the Step 3 requirements first, boundy box are disappeared.
I spent a lot of time to analysis it.

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
In theory, update times are increased. Since different types of sensors are used, noise can be reduced on one side.
In my results, I cannot see it.

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
The difference of visibility between LiDAR and camera cause uncertain track updates.
In this project, I cannot see it in yaw direction. but, in x direction, camera sensor looks like not be able to measure too far point.
Therefore, RMSE track 1 increased rapidly towards the end.


### 4. Can you think of ways to improve your tracking results in the future?
First, visibility of camera should be extended. Therefore, cameras shuold be put on right and left sides.
Second, the limitation of camera's x direction shuould be determine.



