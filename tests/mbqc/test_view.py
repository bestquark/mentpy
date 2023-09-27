# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
"""Tests for the MBQC view module."""
import pytest
import mentpy as mp

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex


def get_node_colors_from_plot(ax):
    """Extract node colors from a matplotlib axis containing a networkx plot."""
    node_colors = []

    # Loop over collections (there could be more than one if nodes are drawn multiple times)
    for collection in ax.collections:
        colors = collection.get_facecolor()
        for color in colors:
            # Convert RGBA to hex
            hex_color = rgb2hex(color)
            node_colors.append(hex_color)

    return node_colors


def test_return_types():
    """Test that the draw function returns a figure and axes."""
    state = mp.templates.muta(2, 1)
    fig, ax = mp.draw(state)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("style", ["default", "black_and_white", "blue_inputs"])
def test_draw_styles(style):
    state = mp.templates.muta(2, 1)
    fig, ax = mp.draw(state, style=style)

    if style == "default":
        colors = get_node_colors_from_plot(ax)
        assert "#FFBD59".lower() in colors

    elif style == "black_and_white":
        colors = get_node_colors_from_plot(ax)
        # Ensure only black and white are used, for example
        assert all(col in ["#FFFFFF".lower()] for col in colors)

    elif style == "blue_inputs":
        colors = get_node_colors_from_plot(ax)
        assert "#FFBD59".lower() in colors

    # Close the plot after checking
    plt.close(fig)
