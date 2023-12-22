import numpy as np
import matplotlib.pyplot as plt
import pygame

class Obstacle():
    # Class representing a rectangular obstacle
    def __init__(self, xyt, size, color='b'):
        """
        Initialize the obstacle.

        Parameters:
        - xyt (list): Position and orientation [x, y, theta] of the obstacle.
        - size (list): Size [width, height] of the obstacle.
        - color (str): Color of the obstacle in the plot (default is 'b' for blue).
        """
        self._xyt = np.array(xyt)  # Obstacle's position and orientation (x, y, theta)
        self.size = np.array(size)  # Obstacle's size (width, height)
        self.corners = np.array([
            [-self.size[0]/2, -self.size[1]/2],
            [-self.size[0]/2, self.size[1]/2],
            [self.size[0]/2, self.size[1]/2],
            [self.size[0]/2, -self.size[1]/2]
        ])  # Corner coordinates relative to the obstacle's center
        self.color = color  # Obstacle's color

        # Rotate and translate the corners to get the actual coordinates in the world
        rot_mat = self.rotation_matrix(xyt[2])
        for i in range(self.corners.shape[0]):
            self.corners[i, :] = rot_mat @ self.corners[i, :]
            self.corners[i, :] += self._xyt[:-1]

    def plot(self, ax, islabel = False):
        """
        Plot the obstacle on the given axes.

        Parameters:
        - ax: Matplotlib axes object.
        """
        if islabel:
            ax.fill(self.corners[:, 0], self.corners[:, 1], self.color, label = "Obstacles")
        else:
            ax.fill(self.corners[:, 0], self.corners[:, 1], self.color)

    def render(self, window, ratio):
        """
        Render the obstacle on the window.

        Parameters:
        - pygame: pygame object
        """
        obstacle_surf = pygame.Surface(self.size*ratio)
        obstacle_surf.set_colorkey((255,255,255))
        obstacle_surf.fill((0,0,255))
        
        # x,y coordinates are flipped..
        obstacle_surf = pygame.transform.rotate(obstacle_surf,self._xyt[2]*180/np.pi)
        rect_center = self._xyt[:2]*ratio
        rect_center[1] = window.get_size()[1] - rect_center[1]
        obstacle_rect = obstacle_surf.get_rect(center = rect_center)
        window.blit(obstacle_surf, obstacle_rect)
        
        return obstacle_surf, obstacle_rect

    def is_collision(self, other):
        """
        Check for collision with another obstacle.

        Parameters:
        - other (Obstacle): Another obstacle to check for collision.

        Returns:
        - bool: True if collision detected, False otherwise.
        """
        
        if isinstance(other,np.ndarray):
            return np.isclose(self.get_min_distance(other),0.0)
        else:
            for axis in np.vstack([self.get_normals(), other.get_normals()]):
                projection1 = self.project(self.corners, axis)
                projection2 = other.project(other.corners, axis)

                if not self.overlap(projection1, projection2):
                    return False

            return True
    
    def get_min_distance(self, point):
        # Rotate and translate the point
        if point.ndim == 2:
            current_point = self._xyt[:2].reshape((2,1))
            transl_point = point - current_point
        else:
            transl_point = np.float64(point) - self._xyt[:-1]
        rotated_point = self.rotation_matrix(-self._xyt[2])@transl_point
        
        if rotated_point.ndim == 2:
            rotated_x = rotated_point[0,:]
            rotated_y = rotated_point[1,:]
            ret_val = np.zeros(point.shape[1])
        else:
            rotated_x = rotated_point[0]
            rotated_y = rotated_point[1]
            ret_val = 0.0

        ret_val += (rotated_x <= -self.size[0] / 2) * ((rotated_x+self.size[0]/2)**2)
        ret_val += (rotated_x >= self.size[0] / 2) * ((rotated_x-self.size[0]/2)**2)
        ret_val += (rotated_y <= -self.size[1] / 2) * ((rotated_y+self.size[1]/2)**2)      
        ret_val += (rotated_y >= self.size[1] / 2) * ((rotated_y-self.size[1]/2)**2)
        

        ret_val = np.sqrt(ret_val)
        return ret_val

    def get_normals(self):
        """
        Calculate normals for each edge of the obstacle.

        Returns:
        - numpy.ndarray: Array of normals.
        """
        normals = np.zeros_like(self.corners)

        for i in range(len(self.corners)):
            edge = self.corners[(i + 1) % len(self.corners), :] - self.corners[i, :]
            normals[i] = np.array([edge[1], -edge[0]])

        return normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    def calculate_potential_field(self, other, weight):
        # Calculate potential field as a function of distance
        distance = np.linalg.norm(self._xyt[:2] - other._xyt[:2])
        potential_field = weight / distance  # Adjust weight and distance according to your needs

        return potential_field        


    @staticmethod
    def rotation_matrix(angle):
        """
        Generate a 2D rotation matrix for the given angle.

        Parameters:
        - angle (float): Rotation angle in radians.

        Returns:
        - numpy.ndarray: 2x2 rotation matrix.
        """
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    @staticmethod
    def rotate_and_translate(vertices, angle, translation):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotated = np.dot(vertices, rotation_matrix.T)
        translated = rotated + translation
        return translated
    
    @staticmethod
    def project(corners, axis):
        """
        Project the corners onto the given axis.

        Parameters:
        - corners (numpy.ndarray): Corner coordinates of the obstacle.
        - axis (numpy.ndarray): Projection axis.

        Returns:
        - numpy.ndarray: Array of projections.
        """
        dot_products = np.dot(corners, axis)
        return np.array([np.min(dot_products), np.max(dot_products)])

    @staticmethod
    def overlap(projection1, projection2):
        """
        Check if two projections overlap.

        Parameters:
        - projection1 (numpy.ndarray): Projection 1.
        - projection2 (numpy.ndarray): Projection 2.

        Returns:
        - bool: True if overlap, False otherwise.
        """
        return (projection1[1] >= projection2[0]) and (projection2[1] >= projection1[0])


class Obstacles():
    # Class representing a collection of obstacles
    def __init__(self, obsts_list = None, fname = None):
        """
        Initialize a collection of obstacles.

        Parameters:
        - obsts_list (list): List of dictionaries, each specifying the xyt and size of an obstacle.
        """
        if obsts_list is None:
            raise ValueError("No obstacles specified.")
        
        self.obsts_list = obsts_list

        self.obstacles = []
        self.obs_sizes = []
        self.obs_xyts = []
        for obst in self.obsts_list:
            self.obstacles.append(Obstacle(obst["xyt"], obst["size"]))
            self.obs_sizes.append(obst["size"])
            self.obs_xyts.append(obst["xyt"])

        self.obs_sizes = np.array(self.obs_sizes)
        self.obs_xyts = np.array(self.obs_xyts)
        self.n = len(self.obsts_list) 
        


    def plot(self, ax = None):
        """
        Plot all obstacles on the given axes.

        Parameters:
        - ax: Matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots()
        # make only one label for obstacles.
        is_label = True
        for obst in self.obstacles:
            obst.plot(ax,is_label)
            is_label = False

        plt.show()
            
            
    def render(self, window, ratio_coord_to_window):
        """
        Plot all obstacles on the window.

        Parameters:
        - ax: Matplotlib axes object.
        """
        obstacle_surfs = []
        obstacle_rects = []
        # make only one label for obstacles.
        for obst in self.obstacles:
            temp_surf, temp_rect = obst.render(window, ratio_coord_to_window)
            obstacle_surfs.append(temp_surf)
            obstacle_rects.append(temp_rect)

        return obstacle_surfs, obstacle_rects
    
    def is_collision(self, other):
        """
        Check for collision with any obstacle in the collection.

        Parameters:
        - other (Obstacle): Another obstacle to check for collision.

        Returns:
        - bool: True if collision detected with any obstacle, False otherwise.
        """
        if isinstance(other,np.ndarray):
            ret_val = np.zeros(other.shape[1],dtype=bool)
            for obst in self.obstacles:
                ret_val = np.logical_or(ret_val,obst.is_collision(other))
        else:
            ret_val = False
            for obst in self.obstacles:
                ret_val = ret_val or obst.is_collision(other)

        return ret_val    
    
    def __getitem__(self, key):
        """
        Get an obstacle from the collection by index.

        Parameters:
        - key: Index of the obstacle.

        Returns:
        - Obstacle: The obstacle at the specified index.
        """
        return self.obstacles[key]


if __name__ == "__main__":
    # Example usage
    obstacles_list = [
        {'xyt': [0, 0, 0], 'size': [1, 1]},
        {'xyt': [1, 0, np.pi/4], 'size': [0.5, 1]},
        {'xyt': [5, 2, np.pi/3], 'size': [2, 1]}
    ]
    a = Obstacles(obsts_list=obstacles_list)
    b = Obstacle([2, 2, 0], [2, 1], 'r')
    fig, ax = plt.subplots()
    ax.axis('equal')
    a.plot(ax)
    b.plot(ax)
    plt.show()

    if a.is_collision(b):
        print("Collision detected!")
    else:
        print("No collision")

        
    # Check point collisions
    point1 = np.array([0, 0])  # Inside the first rectangular obstacle
    point2 = np.array([0, 1])  # Inside the second rectangular obstacle
    point3 = np.array([1, 1])  # Outside any obstacle
        
    print(f"{a.obstacles[0].get_min_distance(point1)}")
    print(f"{a.obstacles[0].get_min_distance(point2)}")
    print(f"{a.obstacles[0].get_min_distance(point3)}")