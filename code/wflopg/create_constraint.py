import numpy as np
import xarray as xr
import collections as cl

from wflopg.constants import COORDS
from wflopg.create_site import xy_to_monomial


# NOTE: We work with double and need this to deal with round-off issues.
#       The multiplier has been determined experimentally,
#       i.e., by trial-and-error.
ε = np.finfo(np.double).eps * 64


def distance(turbine_distance):
    """Return a generator of steps that can fix turbine distance constraints

    The turbine_distance is assumed to be site-distance adimensional.

    """
    def proximity_violation(distance):
        """Check whether a pair of turbines are too close"""
        return ((distance < turbine_distance)  # too close
                & (distance.target != distance.source))
                    # but turbines are never too close to themselves

    def proximity_repulsion(violation, unit_vector, distance):
        """Return steps that can fix turbine constraints

        Both vector and distance must be DataArrays with source and
        target as dimensions, of adimensional xy-coordinate pairs and
        distances, respectively.

        """
        # before we can correct the proximity violations, we need to make sure
        # that for any pair of nonidentical turbines there is a nonzero unit
        # vector; this may happen, e.g., if site constraint correction places
        # more than one turbine at the same parcel vertex
        collision = violation & (distance == 0)
        if collision.any():
            # TODO: it may be more efficient to try and generate just enough
            #       random unit vectors
            random_angle = xr.DataArray(
                np.random.uniform(0, 2 * np.pi, distance.shape),
                coords=distance.coords
            )
            random_unit_vector = xr.concat(
                [np.cos(random_angle), np.sin(random_angle)], 'xy')
            random_unit_vector.coords['xy'] = COORDS['xy']
            unit_vector = xr.where(collision, random_unit_vector, unit_vector)
        # Now that we have appropriate unit vectors everywhere, we can correct.
        # We take thrice the minimally required step, as in case a turbine is
        # pushed outside of the site, half the step can be undone by the site
        # constraint correction procedure; we take thrice and not twice because
        # this leaves some extra room that might be squandered in the sum
        # (from experience we know this is necessary)
        return (
            3 * violation * (turbine_distance - distance) / 2 * unit_vector
        ).sum(dim='source')

    return proximity_violation, proximity_repulsion


def inside_site(site):
    """Check which turbines in the layout are outside the site

    site is a nested dict of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    """
    def inside(layout):

        layout_mon = xy_to_monomial(layout)
        def inside_polygon(constraints):
            # calculate signed distance from constraint
            distance = constraints.dot(layout_mon)
            # turbines with a positive constraint evaluation value
            # lie outside the polygon, so violate the constraint
            inside = (distance <= 0).all(dim='constraint')
            return {'distance': distance, 'inside': inside}

        def inside_disc(circle):
            distance = np.sqrt(np.square(layout - circle).sum(dim='xy'))
            inside = distance <= circle.radius
            return {'distance': distance, 'inside': inside}

        def in_site(site, undecided, exclusion=True):
            """Return which turbines are inside the given site

            This recursive function walks over the site and its exclusions.
            Whether or not it is currently operating in an exclusion is tracked
            by the exclusion variable. The xarray DataArray undecided is a
            boolean array with the same shape of the layout, but removing the
            xy-dimension. It contains the turbines to consider.

            """
            info = {}
            if 'constraints' in site:
                info['constraints'] = inside_polygon(site['constraints'])
                inside = info['constraints']['inside']
            elif 'circle' in site:
                info['circle'] = inside_disc(site['circle'])
                inside = info['circle']['inside']
            ##
            is_in_site = undecided & (~inside if exclusion else inside)
            if 'exclusions' in site:
                # recurse to evaluate the exclusions or inclusions (which, that
                # depends on the status of the exclusion variable)
                info['exclusions'] = []
                for subsite in site['exclusions']:
                    subinfo = in_site(subsite, inside, not exclusion)
                    info['exclusions'].append(subinfo)
                    if exclusion:
                        is_in_site |= subinfo['in_site']
                    else:
                        is_in_site &= subinfo['in_site']
            info['in_site'] = is_in_site

            return info

        undecided = xr.DataArray(np.full(len(layout), True), dims=['target'])
        return in_site(site, undecided)

    return inside


def site(parcels):
    """Create function to move turbines outside the site onto the parcel border

    parcels is a nested dict of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles.

    """
    def _constraint_common(e_clave, layout, scrutinize):
        layout_mon = xy_to_monomial(layout)
        # calculate signed distance
        distance = scrutinize * e_clave['constraints'].dot(layout_mon)
            # TODO: is a value of 0 for unscrutinized turbines safe here?
        # turbines with a nonpositive constraint evaluation value
        # satisfy that constraint
        satisfies = scrutinize & (distance <= 0)
        return distance, satisfies

    def _circle_common(e_clave, layout, scrutinize):
        layout_centered = layout - e_clave['circle']
        dist_sqr = np.square(layout_centered).sum(dim='xy')
        radius_sqr = np.square(e_clave['circle'].radius)
        inside = scrutinize & (dist_sqr <= radius_sqr)
        return layout_centered, dist_sqr, radius_sqr, inside

    def process_enclave(enclave, layout, scrutinize):
        # determine step to enclave border
        if 'constraints' in enclave:
            distance, satisfies = _constraint_common(
                enclave, layout, scrutinize)
            # turbines are inside the enclave if all of the constraints are
            # satisfied
            inside = scrutinize & satisfies.all(dim='constraint')
            steps = (
                ~satisfies * enclave['border_seeker']
                * (distance * (1 + ε) + ε) # …+ε to avoid round-off ‘outsides’
            )
            step = steps.isel(constraint=distance.argmax(dim='constraint'))
            # now check if correction lies on the border;
            # if not, move to the closest vertex
            distance, satisfies = _constraint_common(
                enclave, layout + step, scrutinize)
            still_outside = scrutinize & ~satisfies.all(dim='constraint')
            # TODO: ideally, we only check the relevant vertices,
            #       now we brute-force it by checking all
            vertex_dist_sqr = np.square(
                layout - enclave['vertices']
            ).sum(dim='xy').where(still_outside, np.inf)
            step = xr.where(
                still_outside,
                enclave['vertices'].isel(
                    vertex=vertex_dist_sqr.argmin(dim='vertex')) - layout,
                step
            )
        elif 'circle' in enclave:
            layout_centered, dist_sqr, radius_sqr, inside = _circle_common(
                enclave, layout, scrutinize)
            step = (
                ~inside * layout_centered
                * (np.sqrt(radius_sqr / dist_sqr) - 1)
                * (1 + ε)  # …+ε to avoid round-off ‘outsides’
            )
            enclave = None
        else:
            ValueError("An enclave should consist of at least constraints or "
                       "a circle.")
        # determine exclaves to be treated
        exclaves = enclave.get('exclusions', []) if enclave else []
        # return step to update layout and exclaves to be treated
        return step, exclaves, inside, enclave

    def process_exclave(exclave, layout, scrutinize, enclave):
        # determine step to exclave border
        if 'constraints' in exclave:
            distance, satisfies = _constraint_common(
                exclave, layout, scrutinize)
            # turbines are inside the exclave if all of the constraints are
            # satisfied
            inside = scrutinize & satisfies.all(dim='constraint')
            closest = (-distance).argmin(dim='constraint')
                    # NOTE: argmin decides ties by picking first of minima
                    # TODO: we're choosing the closest too soon here;
                    #       it must be done after discarding the ones that do
                    #       not lie inside the enclosing enclave (if any)
            step = (
                exclave['border_seeker'].isel(constraint=closest)
                * inside * (  # …+ε to avoid round-off ‘outsides’
                   -distance.isel(constraint=closest) * (1 + ε) + ε
                )
            )
            if enclave is not None:
                # now check if correction lies in the encompassing enclave;
                # if not, move to the closest vertex of this exclave
                distance, satisfies = _constraint_common(
                    enclave, layout + step, inside)
                outside_enclave = inside & ~satisfies.all(dim='constraint')
                # TODO: ideally, we only check the relevant vertices,
                #       now we brute-force it by checking all (non-violating)
                vertices = exclave['vertices'][~exclave['violates']]
                vertex_dist_sqr = np.square(
                    layout - vertices
                ).sum(dim='xy').where(outside_enclave, np.inf)
                step = xr.where(
                    outside_enclave,
                    vertices.isel(vertex=vertex_dist_sqr.argmin(dim='vertex'))
                    - layout,
                    step
                )
                enclave = None
        elif 'circle' in exclave:
            layout_centered, dist_sqr, radius_sqr, inside = _circle_common(
                exclave, layout, scrutinize)
            step = xr.where(
                dist_sqr > 0,
                layout_centered * inside * (np.sqrt(radius_sqr / dist_sqr) - 1)
                * (1 + ε),  # …+ε to avoid round-off ‘outsides’
                [np.sqrt(radius_sqr), 0]  # arbitrarily break symmetry
            )
        else:
            ValueError("An exclave should consist of at least constraints or "
                       "a circle.")
        if 'exclusions' in exclave:
            # determine step to exclave border or some enclave border
            steps, exclaves, insides, enclaves = list(
                zip(*(process_enclave(enclave, layout, scrutinize)
                      for enclave in exclave['exclusions']))
            )
            steps = xr.concat([step] + list(steps), 'border')
            dists_sqr = np.square(steps).sum(dim='xy')
            borders = dists_sqr.argmin(dim='border')
                # NOTE: argmin decides ties by picking first of minima
            step = xr.where(scrutinize, steps.isel(border=borders), step)
            # update scrutinize in exclaves depending on chosen border
            borders -= 1  # we want indices for enclaves only
            insides = xr.concat(insides, 'enclave')
            outside = scrutinize & ~insides.any(dim='enclave')
            # TODO: I suspect that the rest of this if-branch is inefficient
            # TODO: It is also very flakey in view of arbitrary dimensionality
            for target, border in enumerate(borders.values):
                if (border < 0).all() or not outside[target].any():
                    continue
                insides[border][{'target': target}] = True
            exclaves = [(exclave, inside, enclaves[i])
                        for i, inside in enumerate(insides)
                        for exclave in exclaves[i]]
        else:  # end of recursion
            exclaves = []
        # return step to update layout and exclaves to be treated
        return step, exclaves

    def to_border(layout):
        """Move turbines outside the site onto the closest parcel border

        layout is an xarray DataArray of xy-coordinates for the turbines.

        """
        old_layout = layout
        layout = old_layout.copy()
        scrutinize = xr.full_like(layout, True).all(dim='xy')
        todo = cl.deque([(parcels, scrutinize, None)])
        while todo:
            exclave, scrutinize, enclave = todo.popleft()
            step, deeper_todo = process_exclave(
                exclave, layout, scrutinize, enclave)
            layout = xr.where(scrutinize, layout + step, layout)
            todo.extend(deeper_todo)

        return layout - old_layout

    return to_border
