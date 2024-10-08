import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 100
fill_percentage = 0.2
seed = 42  # Set your desired seed here
prev_square = None
tested_coords = 0

# Step 1: Create a 100x100 grid with all cells set to False (empty)
grid = np.zeros((grid_size, grid_size), dtype=bool)

# Step 2: Set the seed for reproducibility
np.random.seed(seed)

# Step 3: Randomly fill about 20% of the grid with True values
num_cells_to_fill = int(fill_percentage * grid_size * grid_size)
indices = np.random.choice(grid_size * grid_size, num_cells_to_fill, replace=False)
np.put(grid, indices, True)

# Function to count filled cells in a specified square region
def count_filled_cells_in_square(x, y, size, grid):
    # Ensure the square does not go out of bounds
    x_end = min(x + size, grid.shape[1])  # Max column index
    y_end = min(y + size, grid.shape[0])  # Max row index

    # Extract the sub-grid for the square region
    sub_grid = grid[x:x_end, y:y_end]

    # Count the number of True values (filled cells) in the sub-grid
    filled_count = int(np.sum(sub_grid))

    print(f"Number of filled cells in the {square_size}x{square_size} square starting at ({x}, {y}): {filled_count}")

    #draw_square(x, y, size)
    
    return filled_count

def mutate_coordinate(parent, delta=1):
    while True:
        # Randomly change x and y by a value between -delta and delta
        new_x = parent[0] + np.random.randint(-delta, delta + 1)
        new_y = parent[1] + np.random.randint(-delta, delta + 1)

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

def tournament_selection(population, fitness_values, num_parents, tournament_size=2):
    parents = []
    
    # Repeat until the required number of parents are selected
    for _ in range(num_parents):
        # Randomly select `tournament_size` individuals from the population
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        
        # Find the individual with the highest fitness in this tournament
        best_individual_index = tournament_indices[0]
        for i in tournament_indices:
            if fitness_values[i] > fitness_values[best_individual_index]:
                best_individual_index = i
                
        # Append the best individual (the one with the highest fitness) to parents list
        parents.append(population[best_individual_index])
    
    return parents
    

def evolutionary_algorithm(grid, pop_size=10, generations=5, square_size=10):
    # Initialize population with random coordinates
    population = [(np.random.randint(0, grid.shape[1] - square_size), 
                    np.random.randint(0, grid.shape[0] - square_size)) 
                   for _ in range(pop_size)]
    
    global tested_coords
    tested_coords = population.copy()

    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_values = [count_filled_cells_in_square(x, y, square_size, grid) for x, y in population]
        # Select parents based on fitness
        avg_fitness = np.sum(fitness_values)/pop_size
        print(f"Average fitness for generation {generation}: {avg_fitness}")

        # parents = [
        #     population[i] for i in range(len(fitness_values)) if fitness_values[i] >= avg_fitness
        # ]

        parents = tournament_selection(population, fitness_values, pop_size//2)

        # Create offspring through mutation
        offspring = []
        while (len(parents) + len(offspring) < pop_size): 
            # Mutation
            delta=1
            while delta<square_size:
                rand_range = np.random.randint(len(parents))
                new_x, new_y = mutate_coordinate(parents[rand_range], delta)
                delta = delta + 1
                if (new_x, new_y) not in tested_coords: break
            tested_coords.append((new_x, new_y))
            
            # Check to see that the position will not be out of bounds
            if (new_x < 0 or new_y < 0 or (new_x > grid.shape[1] - square_size) or (new_y > grid.shape[0] - square_size)): continue
            offspring.append((new_x, new_y))

        tested_num = len(tested_coords)
        parents_num = len(parents)
        off_num = len(offspring)
        # Create new population
        population = list(parents) + offspring

    print("End of iterations")
    # Find the best solution in the final population
    best_index = np.argmax([count_filled_cells_in_square(x, y, square_size, grid) for x, y in population])
    best_solution = population[best_index]
    best_filled_count = count_filled_cells_in_square(best_solution[0], best_solution[1], square_size, grid)

    return best_solution, best_filled_count


# Example usage:
x = 10  # Top-left corner column
y = 10  # Top-left corner row
square_size = 10  # Size of the square

# Visualize the grid
visualize_grid(grid)

# Pause briefly to let the grid show
# plt.pause(2)  # Wait for 2 seconds

best_coordinates, filled_count = evolutionary_algorithm(grid)
print(f"Best coordinates: {best_coordinates} with {filled_count} filled cells.")

draw_square(best_coordinates[0], best_coordinates[1], 10)
# Show the final plot with the square outline
# plt.show(block=False)
# plt.pause(0.1)

# count_filled_cells_in_square(50, 50, square_size, grid)
# plt.show(block=False)
# plt.pause(0.1)
print(f"Checked {len(tested_coords)} coordinates")
plt.show()