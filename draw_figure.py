import matplotlib.pyplot as plt
import matplotlib.patches as patches
from embpred.config import REPORTS_DIR

def draw_clean_horizontal_custom_resnet50_diagram():
    FONT_SIZE = 16
    TITLE_FONT_SIZE = 18

    # Increase default font size globally
    plt.rcParams.update({'font.size': FONT_SIZE})

    fig, ax = plt.subplots(figsize=(20, 3))

    # Component definitions (label, center_x, color)
    components = [
        ("Input\n(224×224×3)",    1.0,  "lightblue"),
        ("ResNet-50\n(Pretrained)\nFrozen/Unfrozen", 3.0, "lightgray"),
        ("Global Avg\nPool",       5.0,  "lightgray"),
        ("Flatten\n2048x1",       7.0,  "lightblue"),
        ("Dense Layer\n1",         9.0,  "orange"),
        ("Dense Layer\n2",         11.0, "orange"),
        ("...",                   13.0, "white"),  # placeholder for variable layers
        ("Dense Layer\nN",         15.0, "orange"),
        ("Output\n(14×1)",        17.0, "red")
    ]

    box_width = 1.2
    box_height = 0.8

    # Draw components as rounded boxes
    for label, center_x, color in components:
        left = center_x - box_width / 2
        bottom = 0.5 - box_height / 2
        rect = patches.FancyBboxPatch(
            (left, bottom), box_width, box_height,
            boxstyle="round,pad=0.2", edgecolor="black", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(center_x, 0.5, label, ha="center", va="center",
                fontsize=FONT_SIZE, fontweight="bold")

    # Draw arrows (skip the "..." placeholder)
    for i in range(len(components) - 1):
        label_i, x_i, _ = components[i]
        label_next, x_next, _ = components[i + 1]
        if label_i == "...":
            continue
        start_x = x_i + box_width / 2
        end_x = x_next - box_width / 2
        ax.annotate(
            "",
            xy=(end_x, 0.5), xytext=(start_x, 0.5),
            arrowprops=dict(arrowstyle="->", lw=2)
        )

    ax.set_xlim(0, 19)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.title("ResNet50 with Variable Size MLP", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "resnet50_mlp_diagram.png")
    plt.show()

draw_clean_horizontal_custom_resnet50_diagram()
