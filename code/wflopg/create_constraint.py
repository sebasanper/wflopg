import numpy as np
import xarray as xr
import collections as cl

from wflopg.constants import COORDS


def _xy_to_monomial(xy):
    """Return monomial ‘coordinates’ for the given xy-coordinates

    This function works for any xarray DataArray with xy as a dimension.

    """
    x = xy.sel(xy='x', drop=True)
    y = xy.sel(xy='y', drop=True)
    one = xr.ones_like(x)
    mon = xr.concat([one, x, y], 'monomial').transpose()
    mon.coords['monomial'] = COORDS['monomial']
    return mon


def distance(turbine_distance):
    """Return a generator of steps that can fix turbine distance constraints

    The turbine_distance is assumed to be site-distance adimensional.

    """
    def proximity_repulsion(vector, distance):
        """Return steps that can fix turbine constraints

        Both vector and distance must be DataArrays with source and
        target as dimensions, of adimensional xy-coordinate pairs and
        distances, respectively.

        """
        violation = (0 < distance) & (distance < turbine_distance)
            # NOTE: 0 excluded for distance-to-self,
            #       so try to avoid turbines at the same location
        return xr.where( # just enough to fix the issue
            violation, (turbine_distance - distance) / 2 * vector, [0, 0]
        ).sum(dim='source')

    return proximity_repulsion


def inside_site(parcels):
    """Check which turbines in the layout are outside the parcels

    parcels is a nested dict of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles.

    layout is an xarray DataArray of xy-coordinates for the turbines.

    """
    def inside(layout):
        def parcel_walker(parcels, undecided, exclusion):
            insides = [inside_recursive(parcel, undecided, not exclusion)
                       for parcel in parcels]
            insides = xr.concat(insides, 'parcel')
            if exclusion:
                return insides.any(dim='parcel')
            else:
                return insides.all(dim='parcel')

        def inside_recursive(parcel, undecided, exclusion=True):
            """Return which turbines are inside the given parcel

            This recursive function walks over the parcel and its exclusions.
            Whether or not it is currently operating in an exclusion is tracked
            by the exclusion variable. The xarray DataArray undecided is a
            boolean array with the same shape of the layout, but removing the
            xy-dimension. It contains the turbines to consider.

            """
            ##
            undecided = undecided.copy()
            if 'constraints' in parcel:
                distance = xr.where(  # signed distance
                    undecided, parcel['constraints'].dot(layout_mon), np.nan)
                # turbines with a nonpositive constraint evaluation value
                # satisfy that constraint
                satisfies = xr.where(undecided, distance <= 0, False)
                if exclusion:
                    # if no (not any) constraint is satisfied, then the
                    # turbine is excluded (at this recursion level)
                    outside = ~satisfies.any(dim='constraint')
                else:
                    # excluded turbines are inside the parcel if all of the
                    # constraints at this deeper recursion level are satisfied
                    included = satisfies.all(dim='constraint')
            elif 'circle' in parcel:
                in_disc = xr.where(
                    undecided,
                    np.square(layout - parcel['circle']).sum(dim='xy')
                    <= parcel['circle'].radius_sqr,
                    False
                )
                if exclusion:
                    outside = in_disc
                else:
                    included = in_disc
            ##
            if exclusion:
                inside = undecided.copy()
                undecided &= outside
            else:
                inside = undecided & included
                undecided = inside

            if 'exclusions' in parcel:
                # recurse to evaluate an exclusion (which may be an inclusion
                # if its inside an exclusion, so we flip the exclusion
                # variable's truth value in the called parcel_walker function)
                inside = xr.where(
                    undecided,
                    parcel_walker(
                        parcel['exclusions'], undecided, exclusion),
                    inside
                )
            else:  # end of recursion
                if exclusion:
                    inside = xr.where(undecided, False, inside)
            return inside

        layout_mon = _xy_to_monomial(layout)
        undecided = xr.DataArray(np.full(len(layout), True), dims=['target'])
        return inside_recursive(parcels, undecided)

    return inside


def site(parcels):
    """Create function to move turbines outside the site onto the parcel border

    parcels is a nested dict of constraints and exclusions, where the
    constraints are formulated as linear expressions that evaluate to a
    positive number on their ‘outside’ side or as circles.

    """
    def _constraint_common(e_clave, layout, scrutinize):
        layout_mon = _xy_to_monomial(layout)
        distance = xr.where(  # signed distance
            scrutinize, e_clave['constraints'].dot(layout_mon), 0)
            # TODO: is a value of 0 for unscrutinized turbines safe here?
        # turbines with a nonpositive constraint evaluation value
        # satisfy that constraint
        satisfies = xr.where(scrutinize, distance <= 0, False)
        return distance, satisfies

    def _circle_common(e_clave, layout, scrutinize):
        layout_centered = layout - e_clave['circle']
        dist_sqr = np.square(layout_centered).sum(dim='xy')
        dist_sqr = xr.where(dist_sqr, dist_sqr, np.nan)
            # dist_sqr is used as a divisor, NaN instead of zero gives warnings
        radius_sqr = e_clave['circle'].radius_sqr
        inside = xr.where(scrutinize, dist_sqr <= radius_sqr, False)
        return layout_centered, dist_sqr, radius_sqr, inside

    def process_enclave(enclave, layout, scrutinize):
        # determine step to enclave border
        if 'constraints' in enclave:
            distance, satisfies = _constraint_common(
                enclave, layout, scrutinize)
            # turbines are inside the enclave if all of the constraints are
            # satisfied
            inside = satisfies.all(dim='constraint')
            steps = xr.where(
                satisfies,
                [0, 0],
                distance * (1+1e-6) * enclave['border_seeker']
            )  # 1+1e-6 to avoid round-off ‘outsides’ TODO: more elegantly
            step = steps.isel(constraint=distance.argmax(dim='constraint'))
            # now check if correction lies on the border;
            # if not, move to the closest vertex
            distance, satisfies = _constraint_common(
                enclave, layout + step, scrutinize)
            still_outside = ~satisfies.all(dim='constraint')
            # TODO: ideally, we only check the relevant vertices,
            #       now we brute-force it by checking all
            vertex_dist_sqr = xr.where(
                still_outside,
                np.square(layout - enclave['vertices']).sum(dim='xy'),
                np.inf
            )
            step = xr.where(
                still_outside,
                enclave['vertices'].isel(
                    vertex=vertex_dist_sqr.argmin(dim='vertex')) - layout,
                step
            )
        elif 'circle' in enclave:
            layout_centered, dist_sqr, radius_sqr, inside = _circle_common(
                enclave, layout, scrutinize)
            step = xr.where(
                inside,
                [0, 0],
                layout_centered * (np.sqrt(radius_sqr / dist_sqr) - 1)
            )
        else:
            ValueError("An enclave should consist of at least constraints or "
                       "a circle.")
        # determine exclaves to be treated
        exclaves = enclave.get('exclusions', [])
        # return step to update layout and exclaves to be treated
        return step, exclaves, inside

    def process_exclave(exclave, layout, scrutinize):
        # determine step to exclave border
        if 'constraints' in exclave:
            distance, satisfies = _constraint_common(
                exclave, layout, scrutinize)
            # turbines are inside the exclave if none of the constraints are
            # satisfied
            inside = ~satisfies.any(dim='constraint')
            closest = distance.argmin(dim='constraint')
                    # NOTE: argmin decides ties by picking first of minima
            step = xr.where(
                inside,
                distance.isel(constraint=closest)
                * exclave['border_seeker'].isel(constraint=closest),
                [0, 0]
            )
        elif 'circle' in exclave:
            layout_centered, dist_sqr, radius_sqr, inside = _circle_common(
                exclave, layout, scrutinize)
            step = xr.where(
                inside,
                xr.where(
                    dist_sqr > 0,
                    layout_centered * (np.sqrt(radius_sqr / dist_sqr) - 1),
                    [np.sqrt(radius_sqr), 0]  # arbitrarily break symmetry
                ),
                [0, 0]
            )
        else:
            ValueError("An exclave should consist of at least constraints or "
                       "a circle.")
        if 'exclusions' in exclave:
            # determine step to exclave border or some enclave border
            steps, exclaves, insides = list(
                zip(*(process_enclave(enclave, layout, scrutinize)
                      for enclave in exclave['exclusions']))
            )
            steps = xr.concat([step] + list(steps), 'border')
            dists_sqr = np.square(steps).sum(dim='xy')
            borders = dists_sqr.argmin(dim='border')
                # NOTE: argmin decides ties by picking first of minima
            step = xr.where(scrutinize, steps[borders], step)
            # update scrutinize in exclaves depending on chosen border
            borders -= 1  # we want indices for enclaves only
            insides = xr.concat(insides, 'enclave')
            outside = ~insides.any(dim='enclave')
            # TODO: I suspect that the rest of this if-branch is inefficient
            for target, border in enumerate(borders.values):
                if border < 0 or not outside[target]:
                    continue
                insides[border][target] = True
            exclaves = [(exclave, inside)
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
        scrutinize = xr.DataArray(np.full(len(layout), True), dims=['target'])
        todo = cl.deque([(parcels, scrutinize)])
        while todo:
            exclave, scrutinize = todo.popleft()
            step, deeper_todo = process_exclave(exclave, layout, scrutinize)
            layout = xr.where(scrutinize, layout + step, layout)
            todo.extend(deeper_todo)

        return layout - old_layout

    return to_border
