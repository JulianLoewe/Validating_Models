from ..frequency_distribution_table import FrequencyDistributionTable
from ..groupings.classification import group_by_gt_class, group_by_predicted_class
from ..visualizations import graphviz_helper 
from ..colors import adjust_colors
# from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def confusion_matrix_decomposition(model, checker, constraint, non_applicable_counts = False, indices = None, fontsize= 12, colors = None):
    colors = adjust_colors(colors)
    if indices == None:
        indices = list(range(len(checker.dataset)))

    fdt = FrequencyDistributionTable(checker, [constraint], indices, [group_by_gt_class, group_by_predicted_class], non_applicable_counts=non_applicable_counts)
    nodes = []

    categories = ['invalid','valid']

    if non_applicable_counts:
        categories = ['not applicable'] + categories

    # Plot Confusion Matrix with Valid Instances
    for category in categories:
        plot = fdt.matrix(category=category)
        node_name = f'confusion_matrix_{category.replace(" ","_")}'
        img_path = graphviz_helper.get_image_path(node_name)
        plot.save(img_path)
        nodes.append(graphviz_helper.node_stmt(node_name, graphviz_helper.html_image(graphviz_helper.html_label(f'{category}'.capitalize(), colors['text'], fontsize), img_path), False, colors))
    
    # Plot regular Confusion Matrix
    # labels, display_labels = zip(*checker.dataset.class_names.items())
    # ConfusionMatrixDisplay.from_estimator(model, checker.dataset.x_data()[indices,:], checker.dataset.y_data()[indices,:], labels=labels, display_labels=display_labels)
    plot = fdt.matrix(category='all')
    img_path = graphviz_helper.get_image_path('confusion_matrix')
    plot.save(img_path)
    nodes.append(graphviz_helper.node_stmt('confusion_matrix', graphviz_helper.html_image(graphviz_helper.html_label(f'Confusion Matrix', colors['text'], fontsize), img_path), False, colors))

    edges = ['confusion_matrix -- confusion_matrix_valid','confusion_matrix -- confusion_matrix_invalid']

    if non_applicable_counts:
        edges.append('confusion_matrix -- confusion_matrix_not_applicable')

    cluster = graphviz_helper.cluster_nodes('validation_results', "Confusion Matrix Decomposition", nodes)    

    plt.close()
    return graphviz_helper.grid_layout('', [cluster], edges, None, colors, size=(1,1), orientation='LR')


