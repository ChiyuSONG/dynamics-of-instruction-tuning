import plotly.graph_objects as go
import os
import sys
sys.path.append(".")
from pathlib import Path
import json
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter1d
from utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--plot_type", type=str, default=None,
        choices=["overall", "curated_vs_synthetic-13b", "ood", "curated_vs_synthetic-7b"]
    )
    args = parser.parse_args()

    plot_type = args.plot_type
    result_files = []
    root_path = Path("evaluate/results/test")
    sz_dirs = os.listdir(root_path)
    sz_dirs = [os.path.join(root_path, sz_dir) for sz_dir in sz_dirs if os.path.isdir(os.path.join(root_path, sz_dir))]

    for sz_dir in sz_dirs:
        runs_dirs = os.listdir(sz_dir)
        runs_dirs = [runs_dir for runs_dir in runs_dirs if os.path.isdir(os.path.join(sz_dir, runs_dir))]

        if plot_type == "curated_vs_synthetic-7b":
            runs_dirs = [d for d in runs_dirs if "curated" in d or "7b" in sz_dir]

        elif plot_type == "curated_vs_synthetic-13b":
            runs_dirs = [d for d in runs_dirs if "curated" in d or "13b" in sz_dir]

        elif plot_type == "ood":
            runs_dirs = [d for d in runs_dirs if "curated" in d or "13b" in sz_dir]

        for runs_dir in runs_dirs:
            runs_path = os.path.join(sz_dir, runs_dir)
            result_file = Path(runs_path).glob("*.json")
            result_file = [str(f) for f in result_file]
            assert len(result_file) == 1
            result_files.extend(result_file)

    classes = {}
    for file in result_files:
        base = os.path.split(file)[0]
        base, run_name = os.path.split(base)
        base, model_sz = os.path.split(base)
        run_name = model_sz+"-" + run_name
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                assert len(data) == 1
                for k, v in data.items():
                    k = k.split(".")[0].replace("pred_", "").replace("_valid", "").replace("_test", "")
                    if k not in classes:
                        if plot_type == "ood" and ("ceval" not in k):
                            continue
                        elif plot_type != "ood" and "ceval" in k and plot_type != "overall":
                            continue
                        classes[k] = {}
                    classes[k][run_name] = v

    def log_base_4(x):
        return math.log(x) / math.log(4)

    for c in classes:
        if "expected_" in c:
            continue
        if plot_type=="overall" and "average" not in c:
            continue
        if plot_type!="overall" and "average" in c:
            continue
        fig = go.Figure()

        traces = {}
        for k,v in classes[c].items():
            trace_name = ("-").join(k.split("-")[:-1])
            if trace_name not in traces:
                traces[trace_name] = {"x":[], "y":[]}
            x = int(k.split("-")[-1])
            if "synthetic" in trace_name:
                x /= 10
            x = math.ceil(log_base_4(x))
            traces[trace_name]["x"].append(x)
            traces[trace_name]["y"].append(float(v))

        def modify_title(c):
            title = str(c).capitalize()
            if "Math" in title:
                title = "COT for Grad-Math"
            elif "Average" in title:
                title = "Overall"
            elif "Biology" in title:
                title = "STEM - Biology"
            elif "History" in title:
                title = "Humanity - History"
            elif "Code" in title:
                title = "Code Generation"
            elif "Reasoning" in title:
                title = "Logical Reasoning"
            elif "Understanding" in title:
                title = "Dialogue Understanding"
            elif "Role_play" in title:
                title = "Role-play Chat"
            elif "Creative_writing" in title:
                title = "Creative Writing"
            elif "Ceval_physician" in title:
                title = "Physician Qualification"
            elif "Ceval_teacher_qualification" in title:
                title = "Teacher Qualification"
            elif "Ceval_urban_and_rural_planner" in title:
                title = "Urban & Rural Planner"
            return title

        title = modify_title(c)

        y_ranges = {
            "Overall": [10, 25],
            "STEM - Biology": [10, 60],
            "Chinese": [10, 60],
            "Code Generation": [10, 80],
            "Creative Writing": [30, 90],
            "Ethics": [30, 90],
            "Humanity - History": [10, 60],
            "COT for Grad-Math": [-5, 10],
            "Logical Reasoning": [10, 60],
            "Role-play Chat": [0, 40],
            "Dialogue Understanding": [10, 100],
            "Physician Qualification": [10, 80],
            "Teacher Qualification": [10, 80],
            "Urban & Rural Planner": [10, 80],
        }

        def extract_num(string):
            # Extract the numerical part from the string
            match = re.search(r'\d+', string)
            if match:
                return int(match.group())
            return 0

        sorted_traces_names = sorted(list(traces.keys()), key=extract_num, reverse=False)

        for trace_name in sorted_traces_names:
            xys = [(traces[trace_name]["x"][i], traces[trace_name]["y"][i]) for i in range(len(traces[trace_name]["x"]))]
            xys = sorted(xys)
            if "curated" in trace_name:
                dash = None
            else:
                dash = "dash"

            if "33b" in trace_name:
                symbol = None
                color = "red"
            elif "13b" in trace_name:
                symbol = "circle-open"
                color = "blue"
            else:
                symbol = 'triangle-up'
                color = "green"

            fig.add_trace(go.Scatter(x=[ele[0] for ele in xys], y = gaussian_filter1d([ele[1] for ele in xys], 0.8),
                                     name=trace_name,
                                     marker=dict(symbol=symbol),
                                     line=dict(shape="spline", smoothing=0.2, width=1.3, dash=dash, color=color)))


        fig.update_layout(xaxis_range=[-0.5, 6.5])
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5, 6],
                ticktext=['10', '40', '160', '640', '2,560', '10k', "41k"]
            )
        )

        if title=="Creative Writing":
            fig.update_layout(legend=dict(
                yanchor="bottom",
                y=0.1,
                xanchor="right",
                x=0.99
            ))
        elif title=="Ethics":
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
        else:
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))

        # fig.update_layout(showlegend = False)

        fig.update_layout(
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            ))

        assert title in y_ranges
        fig.update_layout(yaxis_range=y_ranges[title])
        fig.add_hline(y=list(classes["expected_"+c].values())[0], line_width=3, line_dash="dot", line_color="grey")

        # Edit the layout
        fig.update_layout(
            title=title.center(30),
            title_x=0.55,
            title_y=0.99,
            xaxis_title='# Training Samples',
            yaxis_title='Ability Score (%)',
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_layout(
            autosize=False,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=30,
                pad=4
            )
        )

        config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'height': 350,
                'width': 300,
                'scale': 6
            }
        }

        # fig.show(config=config)

        out_name = os.path.join("evaluate", "plots", str(c) + "." + config["toImageButtonOptions"]['format'])
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        fig.write_image(out_name, **config["toImageButtonOptions"])


if __name__ == "__main__":
    main()
    print("Done!")
