import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_two_model_diagrams():
    FONT_SIZE = 14
    TITLE_FONT_SIZE = 16
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 6))
    
    # Common coordinates for both diagrams
    image_branch_y = 0.75    # y-position for image branch
    scalar_branch_y = 0.25   # y-position for scalar branch
    fusion_y = 0.5           # y-position for the fusion box
    pos = {
        "image_input": 1,
        "resnet": 3,
        "gap": 5,
        "flatten": 7,
        "fusion": 9,
        "dense": 11,
        "output": 13
    }
    box_width = 1.8
    box_height = 0.8

    # ====== Diagram for ResNet50TIndexBasic (Naive Concatenation) ======
    ax = ax1
    ax.set_title("ResNet50TIndexBasic\n(Naive Concatenation)", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    
    # Image branch components
    image_components = [
        ("Image Input\n(224×224×3)", pos["image_input"], image_branch_y, "lightblue"),
        ("ResNet-50\nPretrained", pos["resnet"], image_branch_y, "lightgray"),
        ("Global Avg Pool", pos["gap"], image_branch_y, "lightgray"),
        ("Flatten\n(2048)", pos["flatten"], image_branch_y, "lightblue")
    ]
    for label, center_x, center_y, color in image_components:
        left = center_x - box_width/2
        bottom = center_y - box_height/2
        rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                      boxstyle="round,pad=0.2", edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(center_x, center_y, label, ha="center", va="center", fontsize=FONT_SIZE, fontweight="bold")
        
    # Scalar branch components
    scalar_components = [
        ("Scalar Input\n(normalized time)", pos["image_input"], scalar_branch_y, "lightgreen"),
        ("Scalar MLP\n(1→128)", pos["resnet"], scalar_branch_y, "orange")
    ]
    for label, center_x, center_y, color in scalar_components:
        left = center_x - box_width/2
        bottom = center_y - box_height/2
        rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                      boxstyle="round,pad=0.2", edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(center_x, center_y, label, ha="center", va="center", fontsize=FONT_SIZE, fontweight="bold")
        
    # Draw arrows along image branch
    for i in range(len(image_components) - 1):
        start_x = image_components[i][1] + box_width/2
        end_x = image_components[i+1][1] - box_width/2
        ax.annotate("", xy=(end_x, image_branch_y), xytext=(start_x, image_branch_y),
                    arrowprops=dict(arrowstyle="->", lw=2))
    
    # Draw arrows along scalar branch
    for i in range(len(scalar_components) - 1):
        start_x = scalar_components[i][1] + box_width/2
        end_x = scalar_components[i+1][1] - box_width/2
        ax.annotate("", xy=(end_x, scalar_branch_y), xytext=(start_x, scalar_branch_y),
                    arrowprops=dict(arrowstyle="->", lw=2))
    
    # Draw Fusion Box for Basic Model
    fusion_center_x = pos["fusion"]
    left = fusion_center_x - box_width/2
    bottom = fusion_y - box_height/2
    fusion_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                          boxstyle="round,pad=0.2", edgecolor="black", facecolor="violet")
    ax.add_patch(fusion_rect)
    ax.text(fusion_center_x, fusion_y, "Naive Concatenation", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    # Arrows converging to Fusion Box
    ax.annotate("", xy=(fusion_center_x - box_width/2, fusion_y + 0.3),
                xytext=(pos["flatten"] + box_width/2, image_branch_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(fusion_center_x - box_width/2, fusion_y - 0.3),
                xytext=(pos["resnet"] + box_width/2, scalar_branch_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    # Dense Layers and Output for Basic Model
    dense_center_x = pos["dense"]
    left = dense_center_x - box_width/2
    bottom = fusion_y - box_height/2
    dense_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                         boxstyle="round,pad=0.2", edgecolor="black", facecolor="orange")
    ax.add_patch(dense_rect)
    ax.text(dense_center_x, fusion_y, "Dense Layers", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    output_center_x = pos["output"]
    left = output_center_x - box_width/2
    bottom = fusion_y - box_height/2
    output_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                          boxstyle="round,pad=0.2", edgecolor="black", facecolor="red")
    ax.add_patch(output_rect)
    ax.text(output_center_x, fusion_y, "Output\n(14 classes)", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    # Arrows from Fusion to Dense, and Dense to Output
    ax.annotate("", xy=(dense_center_x - box_width/2, fusion_y),
                xytext=(fusion_center_x + box_width/2, fusion_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(output_center_x - box_width/2, fusion_y),
                xytext=(dense_center_x + box_width/2, fusion_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # ====== Diagram for ResNet50TIndexAttention (Attention Gate & Concatenation) ======
    ax = ax2
    ax.set_title("ResNet50TIndexAttention\n(Attention Gate & Concatenation)", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    
    # Draw image branch for Attention model
    for label, center_x, center_y, color in image_components:
        left = center_x - box_width/2
        bottom = center_y - box_height/2
        rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                      boxstyle="round,pad=0.2", edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(center_x, center_y, label, ha="center", va="center", fontsize=FONT_SIZE, fontweight="bold")
    
    # Draw scalar branch for Attention model
    for label, center_x, center_y, color in scalar_components:
        left = center_x - box_width/2
        bottom = center_y - box_height/2
        rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                      boxstyle="round,pad=0.2", edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(center_x, center_y, label, ha="center", va="center", fontsize=FONT_SIZE, fontweight="bold")
    
    # Draw arrows along image branch
    for i in range(len(image_components) - 1):
        start_x = image_components[i][1] + box_width/2
        end_x = image_components[i+1][1] - box_width/2
        ax.annotate("", xy=(end_x, image_branch_y), xytext=(start_x, image_branch_y),
                    arrowprops=dict(arrowstyle="->", lw=2))
    
    # Draw arrows along scalar branch
    for i in range(len(scalar_components) - 1):
        start_x = scalar_components[i][1] + box_width/2
        end_x = scalar_components[i+1][1] - box_width/2
        ax.annotate("", xy=(end_x, scalar_branch_y), xytext=(start_x, scalar_branch_y),
                    arrowprops=dict(arrowstyle="->", lw=2))
    
    # Show the attention gate from scalar branch to image branch
    ax.annotate("Attention Gate", xy=(pos["flatten"], image_branch_y),
                xytext=(pos["resnet"], image_branch_y - 0.1),
                arrowprops=dict(arrowstyle="->", lw=2, color="purple"),
                fontsize=FONT_SIZE, ha="center")
    
    # Draw Fusion Box for Attention model
    fusion_center_x = pos["fusion"]
    left = fusion_center_x - box_width/2
    bottom = fusion_y - box_height/2
    fusion_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                          boxstyle="round,pad=0.2", edgecolor="black", facecolor="violet")
    ax.add_patch(fusion_rect)
    ax.text(fusion_center_x, fusion_y, "Attention &\nConcatenation", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    # Arrows converging into Fusion Box for Attention model
    ax.annotate("", xy=(fusion_center_x - box_width/2, fusion_y + 0.3),
                xytext=(pos["flatten"] + box_width/2, image_branch_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(fusion_center_x - box_width/2, fusion_y - 0.3),
                xytext=(pos["resnet"] + box_width/2, scalar_branch_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    # Draw Dense Layers and Output for Attention model
    dense_center_x = pos["dense"]
    left = dense_center_x - box_width/2
    bottom = fusion_y - box_height/2
    dense_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                         boxstyle="round,pad=0.2", edgecolor="black", facecolor="orange")
    ax.add_patch(dense_rect)
    ax.text(dense_center_x, fusion_y, "Dense Layers", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    output_center_x = pos["output"]
    left = output_center_x - box_width/2
    bottom = fusion_y - box_height/2
    output_rect = patches.FancyBboxPatch((left, bottom), box_width, box_height,
                                          boxstyle="round,pad=0.2", edgecolor="black", facecolor="red")
    ax.add_patch(output_rect)
    ax.text(output_center_x, fusion_y, "Output\n(14 classes)", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold")
    
    ax.annotate("", xy=(dense_center_x - box_width/2, fusion_y),
                xytext=(fusion_center_x + box_width/2, fusion_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(output_center_x - box_width/2, fusion_y),
                xytext=(dense_center_x + box_width/2, fusion_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

draw_two_model_diagrams()
