"""Visualization functions

Example usage (given some problem object `o`):

    import wflopg.visualization as vis
    fig, ax = plt.subplots()
    vis.draw_boundaries(ax, o)
    vis.draw_zones(ax, o)
    vis.draw_turbines(ax, o)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

"""

import matplotlib.pyplot as plt


def draw_boundaries(axes, owflop):
    """Draw the boundaries of a site"""
    def draw_boundary(boundary):
        if 'polygon' in boundary:
            axes.add_patch(
                plt.Polygon(boundary['polygon'], edgecolor='k', fill=False))
        if 'circle' in boundary:
            axes.add_patch(
                plt.Circle(boundary['circle'], boundary['circle'].radius,
                           edgecolor='k', fill=False))
        if 'exclusions' in boundary:
            for exclusion_boundary in boundary['exclusions']:
                draw_boundary(exclusion_boundary)

    for boundary in owflop.boundaries:
        draw_boundary(boundary)


def draw_zones(axes, owflop):
    """Draw the parcels of a site, i.e., all enclaves and exclaves"""
    def draw_zone(zone, exclusion=True):
        c = 'r' if exclusion else 'b'
        l = '--' if exclusion else '-'
        if 'vertices' in zone:
            axes.add_patch(plt.Polygon(zone['vertices'],
                                       edgecolor=c, linestyle=l, fill=False))
        if 'circle' in zone:
            axes.add_patch(plt.Circle(zone['circle'], zone['circle'].radius,
                                      edgecolor=c, linestyle=l, fill=False))
        if 'exclusions' in zone:
            for exclusion_zone in zone['exclusions']:
                draw_zone(exclusion_zone, not exclusion)

    draw_zone(owflop.parcels)


def draw_turbines(axes, owflop, layout=None):
    """Draw the turbines and their proximity exclusion zones"""
    if layout is None:
        layout = owflop._ds['layout']
    turbine_size = owflop.rotor_radius / owflop.site_radius
    for position in layout.values:
        axes.add_patch(
            plt.Circle(position, owflop.minimal_proximity / 2,
                       color='r', linestyle=':', fill=False))
        axes.add_patch(plt.Circle(position, turbine_size, color='k'))
