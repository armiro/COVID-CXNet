from vis.visualization import visualize_cam
from lime import lime_image
import numpy as np, matplotlib.cm as cm


def generate_heatmap(model, input_image):
    grads = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=input_image, backprop_modifier=None,
                          grad_modifier=None, penultimate_layer_idx=None)
    jet_heatmap = np.uint8(cm.jet(grads)[:, :, :, 0] * 255)
    return jet_heatmap


def generate_explanation(model, input_image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image=input_image, classifier_fn=model.predict, hide_color=0,
                                             num_samples=1000, random_seed=18)
    temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    return temp, mask
