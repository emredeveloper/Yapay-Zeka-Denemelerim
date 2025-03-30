import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

# Default dataset as fallback
default_headers = ["Name", "Population", "Size (min cm)", "Size (max cm)", "Weight (min kg)", "Weight (max kg)", "Lifespan (min years)", "Lifespan (max years)"]
default_data = [
    ["Irish Red Fox", 185000, 48, 92, 4.2, 6.8, 3, 5],
    ["Irish Badger", 95000, 62, 88, 8.5, 13.5, 6, 8],
    ["Irish Otter", 13500, 58, 98, 5.5, 11.5, 9, 13],
    ["Red Deer", 42000, 160, 220, 80, 190, 10, 15],
    ["Irish Hare", 233000, 45, 65, 2.5, 3.8, 3, 7]
]

# Initialize with default data
df_headers = default_headers
df_data = default_data
df_pandas = pd.DataFrame(df_data, columns=df_headers)

def load_csv(csv_file):
    """Load data from a CSV file and update the dataframe."""
    global df_pandas, df_headers, df_data
    
    try:
        if csv_file is None:
            return "No file selected. Using default data.", df_data, df_headers
        
        # Read the CSV file
        new_df = pd.read_csv(csv_file.name)
        
        # Update the global dataframe and data
        df_pandas = new_df
        df_headers = new_df.columns.tolist()
        df_data = new_df.values.tolist()
        
        return f"Successfully loaded data from {os.path.basename(csv_file.name)}. Found {len(df_data)} records.", df_data, df_headers
    except Exception as e:
        return f"Error loading CSV: {str(e)}. Using default data.", default_data, default_headers

def filter_data(min_population=0, min_lifespan=0):
    """Filter the dataframe based on population and lifespan."""
    try:
        population_col = [col for col in df_pandas.columns if "population" in col.lower()][0]
        lifespan_col = [col for col in df_pandas.columns if "lifespan" in col.lower() and "min" in col.lower()][0]
        
        filtered_df = df_pandas[
            (df_pandas[population_col] >= min_population) & 
            (df_pandas[lifespan_col] >= min_lifespan)
        ]
        return filtered_df.values.tolist()
    except (IndexError, KeyError):
        # If columns not found, return all data
        return df_data

def generate_chart(chart_type):
    """Generate a chart based on the selected type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        name_col = df_pandas.columns[0]  # Assume first column is animal name
        population_col = [col for col in df_pandas.columns if "population" in col.lower()][0]
        size_min_col = [col for col in df_pandas.columns if "size" in col.lower() and "min" in col.lower()][0]
        size_max_col = [col for col in df_pandas.columns if "size" in col.lower() and "max" in col.lower()][0]
        weight_min_col = [col for col in df_pandas.columns if "weight" in col.lower() and "min" in col.lower()][0]
        weight_max_col = [col for col in df_pandas.columns if "weight" in col.lower() and "max" in col.lower()][0]
        lifespan_max_col = [col for col in df_pandas.columns if "lifespan" in col.lower() and "max" in col.lower()][0]
        
        if chart_type == "Population":
            ax.bar(df_pandas[name_col], df_pandas[population_col])
            ax.set_ylabel("Population")
            ax.set_title("Population of Wildlife")
        elif chart_type == "Size Range":
            ax.barh(df_pandas[name_col], df_pandas[size_max_col] - df_pandas[size_min_col])
            ax.set_xlabel("Size Range (cm)")
            ax.set_title("Size Range of Wildlife")
        elif chart_type == "Weight Range":
            ax.barh(df_pandas[name_col], df_pandas[weight_max_col] - df_pandas[weight_min_col])
            ax.set_xlabel("Weight Range (kg)")
            ax.set_title("Weight Range of Wildlife")
        elif chart_type == "Lifespan":
            ax.barh(df_pandas[name_col], df_pandas[lifespan_max_col])
            ax.set_xlabel("Maximum Lifespan (years)")
            ax.set_title("Maximum Lifespan of Wildlife")
    except (IndexError, KeyError) as e:
        # Fallback to using default column names if the dynamic detection fails
        if chart_type == "Population":
            ax.bar(df_pandas.iloc[:, 0], df_pandas.iloc[:, 1])
            ax.set_ylabel("Population")
            ax.set_title("Population of Wildlife")
        elif chart_type == "Size Range":
            ax.barh(df_pandas.iloc[:, 0], df_pandas.iloc[:, 3] - df_pandas.iloc[:, 2])
            ax.set_xlabel("Size Range (cm)")
            ax.set_title("Size Range of Wildlife")
        elif chart_type == "Weight Range":
            ax.barh(df_pandas.iloc[:, 0], df_pandas.iloc[:, 5] - df_pandas.iloc[:, 4])
            ax.set_xlabel("Weight Range (kg)")
            ax.set_title("Weight Range of Wildlife")
        elif chart_type == "Lifespan":
            ax.barh(df_pandas.iloc[:, 0], df_pandas.iloc[:, 7])
            ax.set_xlabel("Maximum Lifespan (years)")
            ax.set_title("Maximum Lifespan of Wildlife")
    
    # Adjust for long labels and rotate x-axis labels if needed
    if chart_type == "Population":
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot to a temporary file and return the file path
    temp_file = "temp_chart.png"
    plt.savefig(temp_file)
    plt.close(fig)  # Close the figure to free memory
    
    return temp_file

def export_data(format_type):
    """Export the dataframe to the selected format."""
    try:
        if format_type == "CSV":
            output_file = "wildlife_export.csv"
            df_pandas.to_csv(output_file, index=False)
            return output_file
        elif format_type == "JSON":
            output_file = "wildlife_export.json"
            with open(output_file, "w") as f:
                f.write(df_pandas.to_json(orient="records"))
            return output_file
        elif format_type == "Excel":
            output_file = "wildlife_export.xlsx"
            df_pandas.to_excel(output_file, index=False)
            return output_file
        return "Unknown format type"
    except Exception as e:
        return f"Error during export: {str(e)}"

def get_animal_index(animal_name):
    """Get the index of an animal by name."""
    for i, animal in enumerate(df_data):
        if animal[0] == animal_name:
            return i
    return None

def display_animal_info(animal_idx):
    """Display detailed information about the selected animal."""
    if animal_idx is None or animal_idx < 0 or animal_idx >= len(df_pandas):
        return "Select an animal to see details"
    
    try:
        animal = df_pandas.iloc[animal_idx]
        info = f"## {animal.iloc[0]}\n\n"
        
        # Try to detect column names dynamically
        try:
            population_col = [col for col in df_pandas.columns if "population" in col.lower()][0]
            size_min_col = [col for col in df_pandas.columns if "size" in col.lower() and "min" in col.lower()][0]
            size_max_col = [col for col in df_pandas.columns if "size" in col.lower() and "max" in col.lower()][0]
            weight_min_col = [col for col in df_pandas.columns if "weight" in col.lower() and "min" in col.lower()][0]
            weight_max_col = [col for col in df_pandas.columns if "weight" in col.lower() and "max" in col.lower()][0]
            lifespan_min_col = [col for col in df_pandas.columns if "lifespan" in col.lower() and "min" in col.lower()][0]
            lifespan_max_col = [col for col in df_pandas.columns if "lifespan" in col.lower() and "max" in col.lower()][0]
            
            info += f"**Population:** {animal[population_col]:,}\n\n"
            info += f"**Size:** {animal[size_min_col]} - {animal[size_max_col]} cm\n\n"
            info += f"**Weight:** {animal[weight_min_col]} - {animal[weight_max_col]} kg\n\n"
            info += f"**Lifespan:** {animal[lifespan_min_col]} - {animal[lifespan_max_col]} years"
        except (IndexError, KeyError):
            # Fallback to using numeric indices if the dynamic detection fails
            info += f"**Population:** {animal.iloc[1]:,}\n\n"
            info += f"**Size:** {animal.iloc[2]} - {animal.iloc[3]} cm\n\n"
            info += f"**Weight:** {animal.iloc[4]} - {animal.iloc[5]} kg\n\n"
            info += f"**Lifespan:** {animal.iloc[6]} - {animal.iloc[7]} years"
        
        return info
    except Exception as e:
        return f"Error displaying animal info: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), css="""
    #wildlife-df .table-wrap {
        max-height: 400px;
        overflow-y: auto;
    }
""") as demo:
    gr.Markdown("# Wildlife Database")
    gr.Markdown("An interactive database of wildlife with population estimates, physical characteristics, and lifespan information.")
    
    with gr.Tabs():
        with gr.TabItem("Data Input"):
            with gr.Row():
                with gr.Column():
                    csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                    load_btn = gr.Button("Load Data")
                
                with gr.Column():
                    load_status = gr.Textbox(label="Status", value="Using default data")
        
        with gr.TabItem("Data Table"):
            with gr.Row():
                with gr.Column(scale=1):
                    pop_slider = gr.Slider(minimum=0, maximum=1000000, value=0, step=10000, label="Minimum Population")
                    life_slider = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Minimum Lifespan (years)")
                    filter_btn = gr.Button("Filter Data")
                
                with gr.Column(scale=3):
                    df = gr.Dataframe(
                        label="Wildlife Data",
                        value=df_data,
                        headers=df_headers,
                        interactive=True,
                        show_search=True,
                        show_copy_button=True,
                        show_fullscreen_button=True,
                        show_row_numbers=True,
                        wrap=True,
                        elem_id="wildlife-df"
                    )
        
        with gr.TabItem("Visualization"):
            with gr.Row():
                chart_type = gr.Radio(
                    ["Population", "Size Range", "Weight Range", "Lifespan"], 
                    label="Chart Type", 
                    value="Population"
                )
                generate_btn = gr.Button("Generate Chart")
            
            chart_output = gr.Image(label="Chart")
        
        with gr.TabItem("Animal Details"):
            with gr.Row():
                with gr.Column(scale=1):
                    animal_selector = gr.Dropdown(
                        choices=[animal[0] for animal in df_data],
                        label="Select Animal"
                    )
                
                with gr.Column(scale=2):
                    animal_info = gr.Markdown("Select an animal to see details")
        
        with gr.TabItem("Export Data"):
            with gr.Row():
                export_format = gr.Radio(
                    ["CSV", "JSON", "Excel"], 
                    label="Export Format", 
                    value="CSV"
                )
                export_btn = gr.Button("Export")
            
            export_result = gr.Textbox(label="Export Result")
    
    # Set up event handlers
    load_btn.click(
        load_csv, 
        inputs=[csv_input], 
        outputs=[load_status, df, animal_selector]
    )
    
    filter_btn.click(
        filter_data, 
        inputs=[pop_slider, life_slider], 
        outputs=[df]
    )
    
    generate_btn.click(
        generate_chart, 
        inputs=[chart_type], 
        outputs=[chart_output]
    )
    
    animal_selector.change(
        lambda name: display_animal_info(get_animal_index(name)),
        inputs=[animal_selector],