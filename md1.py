import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Parameters
grid_size = 150
fill_percentage = 0.3
seed = 42  # Set your desired seed here

# Set the seed for reproducibility
random.seed(seed)

def random_clustered_fill(grid, ratio):
    # Calculate total number of cells to fill with True
    total_cells = grid.size
    true_count = int(total_cells * ratio)

    # Randomly pick an initial starting point for clustering
    start_x = random.randint(0, grid.shape[0] - 1)
    start_y = random.randint(0, grid.shape[1] - 1)
    
    # Mark the first point as True
    grid[start_x, start_y] = True
    
    # List of coordinates with True values (starts with the first one)
    true_cells = [(start_x, start_y)]
    
    # Generate True values and cluster around existing True values
    for _ in range(true_count - 1):
        # Pick a random cell that is already True
        x, y = random.choice(true_cells)
        
        # Get neighboring cells (up, down, left, right)
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < grid.shape[0] - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < grid.shape[1] - 1: neighbors.append((x, y + 1))
        
        # Filter out neighbors that are already True
        neighbors = [(nx, ny) for nx, ny in neighbors if not grid[nx, ny]]
        
        # If there are valid neighbors, pick one randomly and set it to True
        if neighbors:
            next_x, next_y = random.choice(neighbors)
            grid[next_x, next_y] = True
            true_cells.append((next_x, next_y))
        else:
            # If no valid neighbors, just pick a random False cell in the grid
            false_cells = list(zip(*np.where(grid == False)))
            if false_cells:
                next_x, next_y = random.choice(false_cells)
                grid[next_x, next_y] = True
                true_cells.append((next_x, next_y))
    
    return grid

# Create a grid with all cells set to False (empty)
grid = np.zeros((grid_size, grid_size), dtype=bool)

grid = random_clustered_fill(grid, fill_percentage)

# # Randomly fill a set ratio of the grid with True values
# num_cells_to_fill = int(fill_percentage * grid_size * grid_size)
# indices = random.choice(grid_size * grid_size, num_cells_to_fill, replace=False)
# np.put(grid, indices, True)

prev_square = None
tested_coords = 0

# Function to count filled cells in a specified square region
def count_filled_cells_in_square(x, y, size, grid):
    # Ensure the square does not go out of bounds
    x_end = min(x + size, grid.shape[1])  # Max column index
    y_end = min(y + size, grid.shape[0])  # Max row index

    # Extract the sub-grid for the square region
    sub_grid = grid[x:x_end, y:y_end]

    # Count the number of True values (filled cells) in the sub-grid
    filled_count = int(np.sum(sub_grid))

    # print(f"Number of filled cells in the {square_size}x{square_size} square starting at ({x}, {y}): {filled_count}")

    #draw_square(x, y, size)
    
    return filled_count

def mutate_coordinate(parent, delta=1):
    while True:
        # Randomly change x and y by a value between -delta and delta
        new_x = parent[0] + random.randint(-delta, delta + 1)
        new_y = parent[1] + random.randint(-delta, delta + 1)

        # Check that both x and y are not the same as in the beggining
        if new_x != parent[0] or new_y != parent[1]:
            break

    return new_x, new_y

def draw_square(x, y, size):
    global prev_square
    if prev_square: prev_square.remove()

    # Draw the square outline
    square_outline = plt.Rectangle((x - 0.5, y - 0.5), size, size, edgecolor='red', facecolor='none', lw=0.5)
    plt.gca().add_patch(square_outline)  # Add the rectangle to the plot
    prev_square = square_outline
    plt.draw()  # Update the plot to show the rectangle
    plt.show(block=False)
    plt.pause(0.1)  # Allow the plot to render for a short period

# Function to visualize the grid
def visualize_grid(grid):
    plt.imshow(grid.T, cmap='binary', interpolation='none')  # Transpose for x-horizontal, y-vertical
    plt.title("2D Grid")
    plt.xlabel("X (Columns)")
    plt.ylabel("Y (Rows)")
    
    # Show the plot
    plt.show(block=False)
    plt.pause(0.1)  # Allow the plot to render for a short period


def evolutionary_algorithm(grid, pop_size=100, generations=10, square_size=10):
    # Initialize population with random coordinates
    population = [(random.randint(0, grid.shape[1] - square_size), 
                    random.randint(0, grid.shape[0] - square_size)) 
                   for _ in range(pop_size)]
    
    global tested_coords
    tested_coords = population.copy()

    # Measure the start time
    start_time = time.time()

    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_values = [count_filled_cells_in_square(x, y, square_size, grid) for x, y in population]

        avg_fitness = np.sum(fitness_values)/pop_size
        # print(f"Average fitness for generation {generation+1}: {avg_fitness}")

        parents = [
            population[i] for i in range(len(fitness_values)) if fitness_values[i] >= avg_fitness
        ]

        # Select top 50% of population
        # parents_sort = [x for x, _ in sorted(zip(population, fitness_values), key=lambda pair: pair[1])]
        # parents = parents_sort[-pop_size//2:]

        # Create offspring through mutation
        offspring = []
        while (len(parents) + len(offspring) < pop_size): 
            # Mutation
            delta=1
            while delta<square_size:
                rand_range = random.randint(0, len(parents)-1)
                new_x, new_y = mutate_coordinate(parents[rand_range], delta)
                delta = delta + 1
                if (new_x, new_y) not in tested_coords: break # Exit the loop if there is a new coordinate that was not used before
            tested_coords.append((new_x, new_y))
            
            # Check to see that the position will not be out of bounds
            if (new_x < 0 or new_y < 0 or (new_x > grid.shape[1] - square_size) or (new_y > grid.shape[0] - square_size)): continue
            offspring.append((new_x, new_y))

        # Create new population
        population = list(parents) + offspring

    # Measure the end time
    end_time = time.time()
    print("End of iterations")
    # Calculate the execution time
    execution_time = end_time - start_time

    print(f"Function execution time: {execution_time} seconds")
    # Find the best solution in the final population
    best_index = np.argmax([count_filled_cells_in_square(x, y, square_size, grid) for x, y in population])
    best_solution = population[best_index]
    best_filled_count = count_filled_cells_in_square(best_solution[0], best_solution[1], square_size, grid)

    return best_solution, best_filled_count


# Example usage:
population = 200
generations = 30
square_size = 20  # Size of the square

# Visualize the grid
visualize_grid(grid)


best_coordinates, filled_count = evolutionary_algorithm(grid, population, generations, square_size)
print(f"Best coordinates: {best_coordinates} with {filled_count} filled cells.")

draw_square(best_coordinates[0], best_coordinates[1], square_size)


print(f"Checked {len(tested_coords)} coordinates out of {grid_size*grid_size} ({float(len(tested_coords))/(grid_size*grid_size)*100}%)")
plt.show()