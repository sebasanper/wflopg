import numpy as _np
import xarray as _xr
import collections as _cl

from wflopg.create_site import xy_to_monomial


# NOTE: We work with double and need this to deal with round-off issues.
#       The multiplier has been determined experimentally,
#       i.e., by trial-and-error.
ε = _np.finfo(_np.double).eps * 64


def distance(turbine_distance):
    """Return a generator of steps that can fix turbine distance constraints

    The turbine_distance is assumed to be site-distance adimensional.

    """
    def proximity_repulsion(distance, unit_vector):
        """Check whether turbines are too close & return step to fix this

        Both vector and distance must be DataArrays with source and
        target as dimensions, of adimensional xy-coordinate pairs and
        distances, respectively.

        """
        violation = (
            # too close
            (distance < turbine_distance)
            # but turbines are never too close to themselves
            & (distance.target != distance.source)
        )
        if violation.any():
            dims_to_stack = list(distance.dims)
            distance_flat = distance.stack(pair=dims_to_stack)
            violation_flat = violation.stack(pair=dims_to_stack)
            unit_vector_flat = (
                unit_vector.stack(pair=dims_to_stack).transpose('pair', 'xy'))
            # before we can correct the proximity violations, we need to make
            # sure that for any pair of nonidentical turbines there is a
            # nonzero unit vector; this may happen, e.g., if site constraint
            # correction places more than one turbine at the same parcel vertex
            collision = violation_flat & (distance_flat == 0)
            collisions = collision.sum().values.item()
            if collisions:
                random_angle = _np.random.uniform(0, 2 * _np.pi, collisions)
                unit_vector_flat[collision] = _np.vstack(
                    [_np.cos(random_angle), _np.sin(random_angle)]).T
            # Now that we have appropriate unit vectors everywhere, we can
            # correct. We take twice the minimally required step, as in case a
            # turbine is pushed outside of the site, half the step can be
            # undone by the site constraint correction procedure. Furthermore,
            # we add a constant extra distance of 1/8 (chosen from experience)
            # the turbine distance to reduce the probability immediate later
            # conflict
            step_flat = _xr.zeros_like(unit_vector_flat)
            step_flat = violation_flat * (
                (1.125 * turbine_distance - distance_flat) * unit_vector_flat)
            step = step_flat.unstack('pair').sum(dim='source')
            step.attrs['violations'] = violation_flat.sum().values.item()
            return step
        else:
            return None

    return proximity_repulsion


def inside_site(site):
    """Check which turbines in the layout are outside the site

    site is a nested dict of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    """
    def inside(layout):
        dims_to_stack = list(layout.dims)
        dims_to_stack.remove('xy')
        if len(dims_to_stack) == 1:
            # see https://github.com/pydata/xarray/issues/2802
            layout_flat = layout
            position_name = dims_to_stack[0]
        else:  # len(dims_to_stack) > 1
            layout_flat = layout.stack(
                position=dims_to_stack
            ).transpose('position', 'xy')
            position_name = 'position'
        layout_flat_mon = xy_to_monomial(layout_flat)

        def inside_polygon(constraints):
            # calculate signed distance from constraint
            distance = constraints.dot(layout_flat_mon)
            # turbines with a positive constraint evaluation value
            # lie outside the polygon, so violate the constraint
            inside = (distance <= 0).all(dim='constraint')
            return {'distance': distance, 'inside': inside}

        def inside_disc(circle):
            distance = _np.sqrt(_np.square(layout_flat - circle).sum(dim='xy'))
            inside = distance <= circle.radius
            return {'distance': distance, 'inside': inside}

        def in_site(site, undecided, exclusion):
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

        undecided = _xr.full_like(
            layout_flat.coords[position_name], True, 'bool')
        return in_site(site, undecided, True)

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
        # TODO: is a value of 0 for unscrutinized turbines safe here?
        distance = scrutinize * e_clave['constraints'].dot(layout_mon)
        # turbines with a nonpositive constraint evaluation value
        # satisfy that constraint
        satisfies = scrutinize & (distance <= 0)
        return distance, satisfies

    def _circle_common(e_clave, layout, scrutinize):
        layout_centered = layout - e_clave['circle']
        dist_sqr = _np.square(layout_centered).sum(dim='xy')
        radius_sqr = _np.square(e_clave['circle'].radius)
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
                * (distance * (1 + ε) + ε)  # …+ε to avoid round-off ‘outsides’
            )
            step = steps.isel(constraint=distance.argmax(dim='constraint'))
            # now check if correction lies on the border;
            # if not, move to the closest vertex
            distance, satisfies = _constraint_common(
                enclave, layout + step, scrutinize)
            still_outside = scrutinize & ~satisfies.all(dim='constraint')
            # TODO: ideally, we only check the relevant vertices,
            #       now we brute-force it by checking all
            vertex_dist_sqr = _np.square(
                layout - enclave['vertices']
            ).sum(dim='xy').where(still_outside, _np.inf)
            step = _xr.where(
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
                * (_np.sqrt(radius_sqr / dist_sqr) - 1)
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

    def process_exclave(exclave, layout_flat, scrutinize, enclave):
        # determine step to exclave border
        if 'constraints' in exclave:
            distance, satisfies = _constraint_common(
                exclave, layout_flat, scrutinize)
            # turbines are inside the exclave if all of the constraints are
            # satisfied
            inside = scrutinize & satisfies.all(dim='constraint')
            # NOTE: argmin decides ties by picking first of minima
            # TODO: we're choosing the closest too soon here; it must be done
            #       after discarding the ones that do not lie inside the
            #       enclosing enclave (if any)
            closest = (-distance).argmin(dim='constraint')
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
                    enclave, layout_flat + step, inside)
                outside_enclave = inside & ~satisfies.all(dim='constraint')
                # TODO: ideally, we only check the relevant vertices,
                #       now we brute-force it by checking all (non-violating)
                vertices = exclave['vertices'][~exclave['violates']]
                vertex_dist_sqr = _np.square(
                    layout_flat - vertices
                ).sum(dim='xy').where(outside_enclave, _np.inf)
                step = _xr.where(
                    outside_enclave,
                    vertices.isel(vertex=vertex_dist_sqr.argmin(dim='vertex'))
                    - layout_flat,
                    step
                )
                enclave = None
        elif 'circle' in exclave:
            layout_centered, dist_sqr, radius_sqr, inside = _circle_common(
                exclave, layout_flat, scrutinize)
            step = _xr.where(
                dist_sqr > 0,
                layout_centered * inside
                * (_np.sqrt(radius_sqr / dist_sqr) - 1)
                * (1 + ε),  # …+ε to avoid round-off ‘outsides’
                [_np.sqrt(radius_sqr), 0]  # arbitrarily break symmetry
            )
        else:
            ValueError("An exclave should consist of at least constraints or "
                       "a circle.")
        if 'exclusions' in exclave:
            # determine step to exclave border or some enclave border
            steps, exclaves, insides, enclaves = list(
                zip(*(process_enclave(enclave, layout_flat, scrutinize)
                      for enclave in exclave['exclusions']))
            )
            steps = _xr.concat([step] + list(steps), 'border')
            dists_sqr = _np.square(steps).sum(dim='xy')
            # NOTE: argmin decides ties by picking first of minima
            borders = dists_sqr.argmin(dim='border')
            step = _xr.where(scrutinize, steps.isel(border=borders), step)
            # update scrutinize in exclaves depending on chosen border
            borders -= 1  # we want indices for enclaves only
            insides = _xr.concat(insides, 'enclave')
            outside = scrutinize & ~insides.any(dim='enclave')
            # TODO: I suspect that the rest of this if-branch is inefficient
            for target, border in enumerate(borders.values):
                if border < 0 or not outside[target]:
                    continue
                insides[border][target] = True
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
        dims_to_stack = list(layout.dims)
        dims_to_stack.remove('xy')
        if len(dims_to_stack) > 1:
            layout_flat = (
                # but see https://github.com/pydata/xarray/issues/2802
                layout.stack(position=dims_to_stack)
                      .transpose('position', 'xy')
            )
            position_name = 'position'
        else:
            layout_flat = layout
            position_name = dims_to_stack[0]
        old_layout_flat = layout_flat
        layout_flat = old_layout_flat.copy()
        scrutinize = _xr.full_like(
            layout_flat.coords[position_name], True, 'bool')
        todo = _cl.deque([(parcels, scrutinize, None)])
        while todo:
            exclave, scrutinize, enclave = todo.popleft()
            step_flat, deeper_todo = process_exclave(
                exclave, layout_flat, scrutinize, enclave)
            layout_flat = _xr.where(
                scrutinize, layout_flat + step_flat, layout_flat)
            todo.extend(deeper_todo)

        step_flat = layout_flat - old_layout_flat
        if len(dims_to_stack) > 1:
            return step_flat.unstack('position')
        else:
            return step_flat

    return to_border
