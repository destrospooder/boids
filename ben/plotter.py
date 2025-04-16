import pandas as pd
import plotly.express as px

# Load the CSV file
df = pd.read_csv("random_search_results.csv")

# Create the 3D scatter plot
fig = px.scatter_3d(
    df,
    x='k_coh',
    y='k_ali',
    z='average',
    color='k_col',
    size='k_col',
    title='3D Scatter Plot of Gains vs Average Coverage',
    labels={'k_coh': 'Cohesion (k_coh)', 'k_ali': 'Alignment (k_ali)', 'average': 'Avg Coverage %'},
)

fig.show()

# Create the 3D scatter plot
fig = px.scatter_3d(
    df,
    x='k_col',
    y='k_ali',
    z='average',
    color='k_coh',
    size='k_coh',
    title='3D Scatter Plot of Gains vs Average Coverage',
    labels={'k_col': 'Collision Avoidance (k_col)', 'k_ali': 'Alignment (k_ali)', 'average': 'Avg Coverage %'},
)

fig.show()

# Create the 3D scatter plot
fig = px.scatter_3d(
    df,
    x='k_col',
    y='k_coh',
    z='average',
    color='k_ali',
    size='k_ali',
    title='3D Scatter Plot of Gains vs Average Coverage',
    labels={'k_col': 'Collision Avoidance (k_col)', 'k_coh': 'Cohesion (k_coh)', 'average': 'Avg Coverage %'},
)

fig.show()