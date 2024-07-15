import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))

import gradio as gr
from core import model
import config

state_districts = config.state_districts

with gr.Blocks() as iface:
    with gr.Tabs():
        with gr.TabItem("ASSAM"):
            district_assam = gr.Dropdown(label="Select District", choices=state_districts.get("ASSAM", []))
            output_assam_accuracy = gr.Textbox(label="Accuracy")
            output_assam_report = gr.Textbox(label="Classification Report")
            output_assam_flood_percentage = gr.Textbox(label="Flood Percentage")
            output_assam_plot = gr.Image(label="Rainfall Data")
            output_assam_conf_matrix = gr.Image(label="Confusion Matrix")

            district_assam.change(
                model.load_and_analyze,
                inputs=[gr.State("ASSAM"), district_assam],
                outputs=[output_assam_accuracy, output_assam_report, output_assam_flood_percentage, output_assam_plot, output_assam_conf_matrix]
            )

        with gr.TabItem("UTTAR PRADESH"):
            district_up = gr.Dropdown(label="Select District", choices=state_districts.get('UP', []))
            output_up_accuracy = gr.Textbox(label="Accuracy")
            output_up_report = gr.Textbox(label="Classification Report")
            output_up_flood_percentage = gr.Textbox(label="Flood Percentage")
            output_up_plot = gr.Image(label="Rainfall Data")
            output_up_conf_matrix = gr.Image(label="Confusion Matrix")

            district_up.change(
                model.load_and_analyze,
                inputs=[gr.State("UP"), district_up],
                outputs=[output_up_accuracy, output_up_report, output_up_flood_percentage, output_up_plot, output_up_conf_matrix]
            )

iface.launch()



